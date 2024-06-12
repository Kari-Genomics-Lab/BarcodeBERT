#!/usr/bin/env python

import os
import pickle
import time
from itertools import product

import pandas as pd
import torch
import torch.optim
import torchtext
from torch import nn

torchtext.disable_torchtext_deprecation_warning()

from torchtext.vocab import build_vocab_from_iterator

from barcodebert import utils
from barcodebert.datasets import KmerTokenizer, inference_from_df, inference_from_fasta
from barcodebert.io import load_inference_model


def inference_on_file(file, model, tokenizer):
    if file.endswith((".csv", ".tsv")):
        print(f"Generating embeddings for {file}", flush=True)
        df = pd.read_csv(file, sep="\t" if file.endswith(".tsv") else ",", keep_default_na=False)
        embeddings = inference_from_df(df, model, tokenizer)

    elif file.endswith((".fas", ".fa")):
        print(f"Generating embeddings for {file}", flush=True)
        embeddings = inference_from_fasta(file, model, tokenizer)
    else:
        print(f"Skipping {file}. Extension not supported")
    return embeddings


def run(config):
    r"""
    Run an inference job, using a single GPU worker to create the embeddings.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    t_start = time.time()
    timing_stats = {}

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

    # create embeddings folder
    if not os.path.isdir("embeddings"):
        os.mkdir("embeddings")

    # LOAD PRE-TRAINED CHECKPOINT =============================================
    # Map model parameters to be load to the specified gpu.
    model, pre_checkpoint = load_inference_model(config.pretrained_checkpoint_path, config, device=device)
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
    if not config.from_paper:
        default_kwargs = vars(get_parser().parse_args(["--input=dummy.pt"]))
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
        if config.from_paper:
            vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
        else:
            vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])  # <UNK> is necessary in the hard case

    tokenizer = KmerTokenizer(config.k_mer, vocab, stride=config.k_mer, padding=True, max_len=config.max_len)

    if os.path.isdir(config.input):
        print("Computing the embeddings for all files in the directory ... \n\n")
        input_files = os.listdir(config.input)
        print(input_files)
        for file in input_files:
            filename = os.path.join(config.input, file)
            prefix = file.split(".")[0]
            out_fname = f'{os.path.join("embeddings", prefix)}.pickle'

            if os.path.exists(out_fname):
                print(f"Skipping {file}, it seems that we have computed the embeddings ... \n")
            else:
                embeddings = inference_on_file(filename, model, tokenizer)
                # Save Embeddings
                print(f"Saving embeddings for file {file} ... \n \n")
                prefix = file.split(".")[0]
                with open(out_fname, "wb") as handle:
                    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        prefix = config.input.split(".")[0].split("/")[-1]
        out_fname = f'{os.path.join("embeddings", prefix)}.pickle'

        if os.path.exists(out_fname):
            print(f"Skipping {file}, it seems that we have computed the embeddings ... \n")
        else:

            embeddings = inference_on_file(config.input, model, tokenizer)

            with open(out_fname, "wb") as handle:
                pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dt = time.time() - t_start
    hour = dt // 3600
    minutes = (dt - (3600 * hour)) // 60
    seconds = dt - (hour * 3600) - (minutes * 60)
    print(f"The code finished after: {int(hour)}:{int(minutes):02d}:{seconds:02.0f} (hh:mm:ss)\n")
    timing_stats["overall"] = time.time() - t_start

    print("Timing Stats")
    print(timing_stats)


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
    parser.description = "Use BarcodeBERT to compute embeddings"

    # Model args --------------------------------------------------------------
    group = parser.add_argument_group("Input model")
    group.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default=None,
        type=str,
        metavar="PATH",
        required=False,
        help="Path to pretrained model checkpoint.",
    )
    group = parser.add_argument_group("Checkpoint type parameters (provisional)")
    group.add_argument(
        "--from_paper",
        "--from-paper",
        "--from-workshop",
        "--from_workshop",
        dest="from_paper",
        action="store_true",
        help="Loads the architectures published in the workshop paper Default: %(default)s",
    )
    # Inference args ----------------------------------------------------------------
    group.add_argument(
        "--input",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="PATH containing the files it can be a single file or a folder \
              with multiple files \
              Extensions supported: .csv, .tsv, .fas, .fa. Default: %(default)s",
    )
    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()

    return run(config)


if __name__ == "__main__":
    cli()
