#!/usr/bin/env python

import os
import resource
import time
from itertools import product

import pandas as pd
import sklearn.metrics
import torch
import torch.optim
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchtext.vocab import build_vocab_from_iterator

from barcodebert import utils
from barcodebert.datasets import KmerTokenizer, representations_from_df
from barcodebert.io import get_project_root, load_pretrained_model


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
    model, pre_checkpoint = load_pretrained_model(config.pretrained_checkpoint_path, device=device)
    # Override the classifier with an identity function as we only want the embeddings
    model.classifier = nn.Identity()
    model = model.to(device)

    keys_to_reuse = [
        "k_mer",
        "stride",
        "max_len",
        "tokenizer",
        "use_unk_token",
        "n_layers",
        "n_heads",
    ]
    default_kwargs = vars(get_parser().parse_args(["--pretrained_checkpoint=dummy.pt"]))
    for key in keys_to_reuse:
        if not hasattr(config, key) or getattr(config, key) == getattr(pre_checkpoint["config"], key):
            pass
        elif getattr(config, key) == default_kwargs[key]:
            print(
                f"  Overriding default config value {key}={getattr(config, key)}"
                f" with {getattr(pre_checkpoint['config'], key)} from pretained checkpoint."
            )
        elif getattr(config, key) != getattr(pre_checkpoint["config"], key):
            raise ValueError(
                f"config value for {key} differs from pretrained checkpoint:"
                f" {getattr(config, key)} (ours) vs {getattr(pre_checkpoint['config'], key)} (pretrained checkpoint)"
            )
        setattr(config, key, getattr(pre_checkpoint["config"], key, None))

    config.pretrained_run_name = pre_checkpoint["config"].run_name
    config.pretrained_run_id = pre_checkpoint["config"].run_id

    # DATASET =================================================================

    if not config.use_unk_token:
        kmer_iter = (["".join(kmer)] for kmer in product("ACGTN", repeat=config.k_mer))
        vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>"])
        vocab.set_default_index(vocab["N" * config.k_mer])  # <UNK> and <CLS> do not exist anymore
    else:
        kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=config.k_mer))
        vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])  # <UNK> is necessary in the hard case

    tokenizer = KmerTokenizer(config.k_mer, vocab, stride=config.k_mer, padding=True, max_len=config.max_len)

    if config.data_dir is None:
        config.data_dir = os.path.join(get_project_root(), "data")

    df_train = pd.read_csv(os.path.join(config.data_dir, "supervised_train.csv"))
    df_test = pd.read_csv(os.path.join(config.data_dir, "unseen.csv"))

    if config.taxon.lower() == "bin":
        config.target_level = "bin_uri"
    else:
        config.target_level = config.taxon + "_name"

    timing_stats["preamble"] = time.time() - t_start

    # Ensure model is in eval mode
    model.eval()
    t_start_embed = time.time()
    # Generate emebddings for the training and test sets
    print("Generating embeddings for test set", flush=True)
    X_unseen, y_unseen, orders = representations_from_df(df_test, config.target_level, model, tokenizer)
    print("Generating embeddings for train set", flush=True)
    X, y, train_orders = representations_from_df(df_train, config.target_level, model, tokenizer)
    timing_stats["embed"] = time.time() - t_start_embed

    c = 0
    for label in y_unseen:
        if label not in y:
            c += 1
    print(f"There are {c} genus that are not present during training")

    running_info = resource.getrusage(resource.RUSAGE_SELF)
    dt = time.time() - t_start_embed
    hour = dt // 3600
    minutes = (dt - (3600 * hour)) // 60
    seconds = dt - (hour * 3600) - (minutes * 60)
    memory = running_info.ru_maxrss / 1e6
    print(f"Creating embeddings took: {int(hour)}:{int(minutes):02d}:{seconds:02.0f} (hh:mm:ss)\n")
    print(f"Max memory usage: {memory} (GB)")

    # kNN =====================================================================
    print("Computing Nearest Neighbors", flush=True)

    # Fit ---------------------------------------------------------------------
    t_start_train = time.time()
    clf = KNeighborsClassifier(n_neighbors=config.n_neighbors, metric=config.metric)
    clf.fit(X, y)
    timing_stats["train"] = time.time() - t_start_train

    # Evaluate ----------------------------------------------------------------
    t_start_test = time.time()
    # Create results dictionary
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
        model_name = os.path.join(*os.path.split(config.pretrained_checkpoint_path)[-2:])
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
    group = parser.add_argument_group("Input model")
    group.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to pretrained model checkpoint (required).",
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
