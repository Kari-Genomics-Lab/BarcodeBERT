import argparse
import random

import scipy.io as sio

from barcodebert.bzsl.feature_extraction import (
    extract_clean_barcode_list,
    extract_clean_barcode_list_for_aligned,
    extract_dna_features,
)
from barcodebert.bzsl.models import load_model

random.seed(10)


def load_data(args):
    x = sio.loadmat(args.input_path)

    if args.using_aligned_barcode:
        barcodes = extract_clean_barcode_list_for_aligned(x["nucleotides_aligned"])
    else:
        barcodes = extract_clean_barcode_list(x["nucleotides"])
    labels = x["labels"].squeeze()

    return barcodes, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--model", choices=["barcodebert", "dnabert", "dnabert2"], default="barcodebert")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output", type=str, default="../data/INSECT/dna_embedding.csv")
    parser.add_argument("--using_aligned_barcode", "--alignment", default=False, action="store_true")
    parser.add_argument("-k", "--kmer", default=6, type=int, dest="k", help="k-mer value for tokenization")
    parser.add_argument(
        "-s",
        "--save-all",
        action="store_true",
        dest="extract_all_embeddings",
        help="if specified, save all embeddings rather than average class embedding",
    )
    args = parser.parse_args()

    model, sequence_pipeline = load_model(args, k=args.k)
    model.eval()

    barcodes, labels = load_data(args)

    extract_dna_features(
        args.model,
        model,
        sequence_pipeline,
        barcodes,
        labels,
        args.output,
        extract_class_embeddings=not args.extract_all_embeddings,
    )
