#!/usr/bin/env python

import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

sys.path.append(".")
print(sys.path)
print(os.getcwd())

from barcodebert import utils
from baselines.datasets import labels_from_df, representations_from_df
from baselines.io import load_baseline_model


def run(config):
    r"""
    Run linear probing job, using a single GPU worker to create the embeddings.

    Parameters
    ----------
    config: argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    t_start = time.time()
    timing_stats = {}

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # LOGGING =================================================================
    # Setup logging and saving

    # If we're using wandb, initialize the run, or resume it if the job was preempted.
    if config.log_wandb:
        wandb_run_name = config.run_name
        EXCLUDED_WANDB_CONFIG_KEYS = [
            "log_wandb",
            "wandb_entity",
            "wandb_project",
            "global_rank",
            "local_rank",
            "run_name",
            "run_id",
            "model_output_dir",
        ]
        job_type = "linearprobe"
        config.pretrained_run_id = "from_pretrained"
        wandb.init(
            name=wandb_run_name,
            id=config.run_id,
            resume="allow",
            group=config.pretrained_run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type=job_type,
            tags=["evaluate", job_type],
        )

    # LOAD PRE-TRAINED CHECKPOINT =============================================
    # Map model parameters to be load to the specified gpu.

    embedder = load_baseline_model(config.backbone)
    embedder.name = config.backbone

    # Ensure model is in eval mode
    embedder.model.eval()

    # DATASET =================================================================
    if config.taxon.lower() == "bin":
        target_level = "bin_uri"
    else:
        target_level = f"{config.taxon}_index"

    timing_stats["preamble"] = time.time() - t_start
    t_start_embed = time.time()

    # Data files
    train_filename = os.path.join(config.data_dir, "supervised_train.csv")
    validation_filename = os.path.join(config.data_dir, "supervised_val.csv")
    test_filename = os.path.join(config.data_dir, "supervised_test.csv")

    # Get pipeline for reference labels:
    df = pd.read_csv(train_filename, sep="\t" if train_filename.endswith(".tsv") else ",", keep_default_na=False)
    labels = df[target_level].to_list()
    label_set = sorted(set(labels))
    label_pipeline = lambda x: label_set.index(x)

    # Generate emebddings for the training, test and validation sets
    print("Generating embeddings for test set", flush=True)
    X_test = representations_from_df(test_filename, embedder, batch_size=128)
    y_test = labels_from_df(test_filename, target_level, label_pipeline)
    print(X_test.shape, y_test.shape)

    print("Generating embeddings for validation set", flush=True)
    X_val = representations_from_df(validation_filename, embedder, batch_size=128)
    y_val = labels_from_df(validation_filename, target_level, label_pipeline)
    print(X_test.shape, y_test.shape)

    print("Generating embeddings for train set", flush=True)
    X = representations_from_df(train_filename, embedder, batch_size=128)
    y = labels_from_df(train_filename, target_level, label_pipeline)
    print(X.shape, y.shape)

    timing_stats["embed"] = time.time() - t_start_embed

    # Normalize the features
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    X = torch.tensor(X).float()
    X_val = torch.tensor(X_val).float()
    X_test = torch.tensor(X_test).float()

    y = torch.tensor(y)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    print("Feature shapes:", X.shape, X_val.shape, X_test.shape)
    print("Labels shapes", y.shape, y_val.shape, y_test.shape)

    train = torch.utils.data.TensorDataset(X, y)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)

    val = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = DataLoader(val, batch_size=1024, shuffle=False, drop_last=False)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = DataLoader(test, batch_size=1024, shuffle=False, drop_last=False)

    # Linear Probing =====================================================================
    print("Training the Linear Classifier", flush=True)

    # Define the model
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], torch.unique(y).shape[0]))
    print(clf)

    print(y.min(), y.max())
    print(torch.unique(y).shape[0])

    # TRAIN ===================================================================
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

    clf.to(device)
    timing_stats = {}

    num_epochs = 200
    for epoch in range(num_epochs):
        loss_epoch = 0
        acc_epoch = 0

        for _batch_idx, (X_train, y_train) in enumerate(train_loader):

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            # Forward pass
            y_pred = clf(X_train)
            loss = criterion(y_pred, y_train)
            loss_epoch += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                is_correct = y_pred.argmax(dim=1) == y_train
                # Accuracy
                acc = is_correct.sum() / is_correct.numel()
                acc = 100.0 * acc.item()
                acc_epoch += acc

        results = {"loss": loss_epoch / (_batch_idx + 1), "accuracy": acc_epoch / (_batch_idx + 1)}

        val_results = evaluate(val_loader, clf, device, partition_name="Val", verbosity=0, is_distributed=False)

        if config.log_wandb:
            wandb.log(
                {
                    "Training/epochwise/epoch": epoch,
                    **{f"Training/epochwise/Train/{k}": v for k, v in results.items()},
                    **{f"Training/epochwise/Val/{k}": v for k, v in val_results.items()},
                },
                step=_batch_idx * epoch,
            )

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], \
                Loss: {results['loss']:.4f}, Training Accuracy: {results['accuracy']:.4f}",
                flush=True,
            )

    # Test the model after training
    test_results = evaluate(test_loader, clf, device, partition_name="Test", verbosity=1, is_distributed=False)

    wandb.log({**{f"Eval/Test/{k}": v for k, v in test_results.items()}}, step=_batch_idx * num_epochs)


def evaluate(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
    is_distributed=False,
):
    r"""
    Evaluate model performance on a dataset.

    Adapted from: https://github.com/Kari-Genomics-Lab/BIOSCAN_5M_DNA_experiments/blob/main/barcodebert/evaluation.py#L13

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset to evaluate on.
    model : torch.nn.Module
        Model to evaluate.
    device : torch.device
        Device to run the model on.
    partition_name : str, default="Val"
        Name of the partition being evaluated.
    verbosity : int, default=1
        Verbosity level.
    is_distributed : bool, default=False
        Whether the model is distributed across multiple GPUs.

    Returns
    -------
    results : dict
        Dictionary of evaluation results.
    """
    model.eval()

    y_true_all = []
    y_pred_all = []
    xent_all = []

    for sequences, y_true in dataloader:
        sequences = sequences.to(device)
        y_true = y_true.to(device)

        with torch.no_grad():
            logits = model(sequences)
            xent = F.cross_entropy(logits, y_true, reduction="none")
            y_pred = torch.argmax(logits, dim=-1)

        xent_all.append(xent.cpu().numpy())
        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())

    # Concatenate the targets and predictions from each batch
    xent = np.concatenate(xent_all)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    # If the dataset size was not evenly divisible by the world size,
    # DistributedSampler will pad the end of the list of samples
    # with some repetitions. We need to trim these off.
    n_samples = len(dataloader.dataset)
    xent = xent[:n_samples]
    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]
    # Create results dictionary
    results = {}
    results["count"] = len(y_true)
    results["cross-entropy"] = np.mean(xent)
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_true, y_pred)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    # Could expand to other metrics too

    if verbosity >= 1:
        print(f"\n{partition_name} evaluation results:")
        for k, v in results.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif "entropy" in k:
                print(f"  {k + ' ':.<24s} {v:9.5f} nat")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")

    return results


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import sys

    from barcodebert.pretraining import get_parser as get_pretraining_parser

    parser = get_pretraining_parser()

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser.prog = prog
    parser.description = "Evaluate with k-nearest neighbors for BarcodeBERT."

    # Model args --------------------------------------------------------------
    group = parser.add_argument_group("Input model type")
    group.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        required=False,
        help=" Model checkpoint path for new BarcodeBERT.",
    )
    group.add_argument(
        "--backbone",
        "--model_type",
        dest="backbone",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="Architecture of the Encoder one of [DNABERT-2, Hyena_DNA, DNABERT-S, \
              BarcodeBERT, NT]",
    )
    # kNN args ----------------------------------------------------------------
    group = parser.add_argument_group("kNN parameters")
    group.add_argument(
        "--taxon",
        type=str,
        default="species",
        help="Taxonomic level to evaluate on. Default: %(default)s",
    )
    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    return run(config)


if __name__ == "__main__":
    cli()
