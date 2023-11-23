import argparse
import random

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from barcodebert.bzsl.models import load_model

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
random.seed(10)


def extract_clean_barcode_list(barcodes):
    barcode_list = []

    for i in barcodes:
        barcode_list.append(str(i[0][0]))

    return barcode_list


def extract_clean_barcode_list_for_aligned(barcodes):
    barcodes = barcodes.squeeze().T
    barcode_list = []
    for i in barcodes:
        barcode_list.append(str(i[0]))

    return barcode_list


def load_data(args):
    x = sio.loadmat(args.input_path)

    if args.using_aligned_barcode:
        barcodes = extract_clean_barcode_list_for_aligned(x["nucleotides_aligned"])
    else:
        barcodes = extract_clean_barcode_list(x["nucleotides"])
    labels = x["labels"].squeeze()

    return barcodes, labels


def extract_and_save_class_level_feature(
    args, model, sequence_pipeline, barcodes, labels, extract_class_embeddings=True
):
    all_label = np.unique(labels)
    all_label.sort()
    dict_emb = {}

    with torch.no_grad():
        all_embeddings = []
        pbar = tqdm(enumerate(labels), total=len(labels))
        for i, label in pbar:
            pbar.set_description("Extracting features: ")
            _barcode = barcodes[i]
            if args.model == "dnabert2":
                x = sequence_pipeline(_barcode).to(device)
                x = model(x)[0]
                # x = torch.mean(x[0], dim=0)  # mean pooling
                x = torch.max(x[0], dim=0)[0]  # max pooling
            else:
                x = torch.tensor(sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
                x = model(x).hidden_states[-1]
                x = x.mean(1)  # Global Average Pooling excluding CLS token
            x = x.cpu().numpy()

            if extract_class_embeddings:
                if str(label) not in dict_emb.keys():
                    dict_emb[str(label)] = []
                dict_emb[str(label)].append(x)
            else:
                all_embeddings.append(x)
    if extract_class_embeddings:
        class_embed = []
        for i in all_label:
            class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
        class_embed = np.array(class_embed, dtype=object)
        class_embed = class_embed.T.squeeze()
        np.savetxt(args.output, class_embed, delimiter=",")
    else:
        all_embeddings = np.array(all_embeddings).squeeze()

        np.savetxt(args.output, all_embeddings, delimiter=",")
    print("DNA embeddings is saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument(
        "--model", choices=["barcodebert", "dnabert", "dnabert2"], default="barcodebert"
    )
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

    extract_and_save_class_level_feature(
        args, model, sequence_pipeline, barcodes, labels, extract_class_embeddings=not args.extract_all_embeddings
    )
