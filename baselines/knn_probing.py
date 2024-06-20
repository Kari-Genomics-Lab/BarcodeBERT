#!/usr/bin/env python

import os
import resource
import sys
import time
from itertools import product

import pandas as pd
import sklearn.metrics
import torch
import torch.optim
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchtext.vocab import build_vocab_from_iterator

sys.path.append(".")
print(sys.path)
print(os.getcwd())

from barcodebert import utils
from barcodebert.datasets import KmerTokenizer
from barcodebert.io import load_pretrained_model
from baselines.datasets import labels_from_df, representations_from_df
from baselines.io import load_baseline_model


def run(config):
    r"""
    Run kNN job, using a single GPU worker to create the embeddings.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
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
    test_filename = os.path.join(config.data_dir, "unseen.csv")

    # Get pipeline for reference labels:
    df = pd.read_csv(train_filename, sep="\t" if train_filename.endswith(".tsv") else ",", keep_default_na=False)
    labels = df[target_level].to_list()
    label_set = sorted(set(labels))
    label_pipeline = lambda x: label_set.index(x)

    # Generate emebddings for the training and test sets
    print("Generating embeddings for test set", flush=True)
    X_unseen = representations_from_df(test_filename, embedder, batch_size=128)
    y_unseen = labels_from_df(test_filename, f"{config.taxon}_index", label_pipeline)
    print(X_unseen.shape, y_unseen.shape)

    print("Generating embeddings for train set", flush=True)
    X = representations_from_df(train_filename, embedder, batch_size=128)
    y = labels_from_df(train_filename, f"{config.taxon}_index", label_pipeline)
    print(X.shape, y.shape)

    timing_stats["embed"] = time.time() - t_start_embed

    # kNN =====================================================================
    print("Computing Nearest Neighbors", flush=True)

    # Fit ---------------------------------------------------------------------
    t_start_train = time.time()
    clf = KNeighborsClassifier(n_neighbors=config.n_neighbors, metric=config.metric)
    clf.fit(X, y)
    timing_stats["train"] = time.time() - t_start_train

    # Evaluate ----------------------------------------------------------------
    t_start_test = time.time()

    # Create a results dictionary
    results = {}
    for partition_name, X_part, y_part in [("Train", X, y), ("Unseen", X_unseen, y_unseen)]:
        y_pred = clf.predict(X_part)
        res_part = {}
        res_part["count"] = len(y_part)
        # Note that these evaluation metrics have all been converted to percentages
        res_part["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred)
        res_part["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_part, y_pred)
        res_part["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="micro")
        res_part["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="macro")
        res_part["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="weighted")
        results[partition_name] = res_part
        print(f"\n{partition_name} evaluation results:")
        for k, v in res_part.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")
    acc = results["Unseen"]["accuracy"]
    timing_stats["test"] = time.time() - t_start_test

    # Save results -------------------------------------------------------------
    dt = time.time() - t_start
    hour = dt // 3600
    minutes = (dt - (3600 * hour)) // 60
    seconds = dt - (hour * 3600) - (minutes * 60)
    print(f"The code finished after: {int(hour)}:{int(minutes):02d}:{seconds:02.0f} (hh:mm:ss)\n")

    with open("KNN_RESULTS.txt", "a") as f:
        model_name = config.backbone
        f.write(f"\n{model_name} \t {acc:.4f}")

    timing_stats["overall"] = time.time() - t_start

    # LOGGING =================================================================
    # Setup logging and saving

    if config.log_wandb:
        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
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
        job_type = "knn"
        config.pretrained_run_id = "from_pretrained"
        wandb.init(
            name=wandb_run_name,
            id=config.run_id,
            group=config.pretrained_run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type=job_type,
            tags=["evaluate", job_type],
        )

    # Log results to wandb ----------------------------------------------------
    wandb.log(
        {
            **{f"knn/duration/{k}": v for k, v in timing_stats.items()},
            **{f"knn/{partition}/{k}": v for partition, res in results.items() for k, v in res.items()},
        },
    )


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
        help="Architecture of the Encoder one of [DNABERT-2, HyenaDNA, DNABERT-S, \
              BarcodeBERT, NT]",
    )
    # kNN args ----------------------------------------------------------------
    group = parser.add_argument_group("kNN parameters")
    group.add_argument(
        "--taxon",
        type=str,
        default="genus",
        help="Taxonomic level to evaluate on. Default: %(default)s",
    )
    group.add_argument(
        "--n-neighbors",
        "--n_neighbors",
        default=1,
        type=int,
        help="Neighborhood size for kNN. Default: %(default)s",
    )
    group.add_argument(
        "--metric",
        default="cosine",
        type=str,
        help="Distance metric to use for kNN. Default: %(default)s",
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
