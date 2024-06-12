import argparse
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from model import load_model
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
random.seed(10)


import matplotlib.pyplot as plt
import seaborn as sns
import umap


class DNADataset(Dataset):
    def __init__(self, barcodes, labels, tokenizer, pre_tokenize=False):
        # Vocabulary
        self.barcodes = barcodes
        self.labels = labels
        self.pre_tokenize = pre_tokenize
        self.tokenizer = tokenizer

        self.tokenized = tokenizer(self.barcodes.tolist()) if self.pre_tokenize else None

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        tokens = self.tokenized[idx] if self.pre_tokenize else self.tokenizer(self.barcodes[idx])
        if not isinstance(tokens, torch.Tensor):
            processed_barcode = torch.tensor(tokens, dtype=torch.int64)
        else:
            processed_barcode = tokens.clone().detach().to(dtype=torch.int64)
        return processed_barcode, self.labels[idx]


def extract_features(args, file, model, sequence_pipeline):

    df = pd.read_csv(f"{args.input_path}/{file}", sep=None)
    target_level = "genus_name"

    barcodes = df["nucleotides"]
    targets = df[target_level]

    label_set = sorted(set(targets.tolist()))
    targets = np.array([label_set.index(x) for x in targets])

    dna_embeddings = []
    labels = []

    with torch.no_grad():
        for i, _barcode in enumerate(barcodes):
            x = torch.tensor(sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
            x = model(x).hidden_states[-1]
            x = x.mean(1)  # Global Average
            # print(x.shape)
            dna_embeddings.extend(x.cpu().numpy())
            labels.append(targets[i])

    print(f"There are {len(dna_embeddings)} points in the dataset")
    latent = np.array(dna_embeddings).reshape(-1, 768)
    y = np.array(labels)
    # print(latent.shape)
    return latent, y


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--model", choices=["bioscanbert", "dnabert", "dnabert2"], default="bioscanbert")
    parser.add_argument("--checkpoint", default="bert_checkpoint/5-mer/model_41.pth", type=str)
    parser.add_argument("--output_dir", type=str, default="../data/INSECT/")
    parser.add_argument("--using_aligned_barcode", default=False, action="store_true")
    parser.add_argument("--n_epoch", default=12, type=int)
    parser.add_argument("-k", "--kmer", default=6, type=int, dest="k", help="k-mer value for tokenization")
    parser.add_argument(
        "--batch-size", default=8, type=int, dest="batch_size", help="batch size for supervised training"
    )
    parser.add_argument(
        "--model-output", default=None, type=str, dest="model_out", help="path to save model after training"
    )

    args = parser.parse_args()

    train_file = f"embeddings/{args.k}_train.pkl"
    test_file = f"embeddings/{args.k}_unseen.pkl"

    if os.path.isfile(train_file):
        print(f"Representations found  after {time.time()-start} seconds . . .")
        with open(train_file, "rb") as f:
            X, y = pickle.load(f)

    else:

        print("Loading the model.....")
        model, sequence_pipeline = load_model(args, k=args.k)

        X, y = extract_features(args, "supervised_train.csv", model, sequence_pipeline)
        file = open(train_file, "wb")
        pickle.dump((X, y), file)
        file.close()

    if os.path.isfile(test_file):
        print(f"Representations found  after {time.time()-start} seconds . . .")
        with open(test_file, "rb") as f:
            X_unseen, y_unseen = pickle.load(f)

    else:

        print("Loading the model.....")
        model, sequence_pipeline = load_model(args, k=args.k)

        X_unseen, y_unseen = extract_features(args, "unseen.csv", model, sequence_pipeline)
        file = open(test_file, "wb")
        pickle.dump((X_unseen, y_unseen), file)
        file.close()

    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "serif"
    embedding = umap.UMAP(random_state=42).fit_transform(X)

    train = pd.read_csv(f"{args.input_path}/supervised_train.csv")

    plt.title("DNABERT representation space of training sequences \n colored by order")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    # sns.scatterplot(x=embedding[:,0], y=embedding[:, 1], hue=train_orders, s=2, legend='auto')
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=train["order_name"].to_list(), s=2, legend="auto")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    plt.savefig("BERT_embeddings.png", dpi=150)

    # 1 - Nearest Neighbors
    print("Computing Nearest Neighbors")

    # X = normalize(X, axis=1, norm='l2')
    # X_unseen = normalize(X_unseen, axis=1, norm='l2')

    # print(y)
    # print(y_unseen)

    # metrics = ['manhattan', 'cosine', 'minkowski']
    metrics = ["cosine"]
    scores = []
    neighbour_size = 1
    for metric in metrics:
        neigh = KNeighborsClassifier(n_neighbors=neighbour_size, metric=metric)
        neigh.fit(X, y)
        # scores.append(neigh.score(X_unseen, y_unseen))
        scores.append(neigh.score(X_unseen, y_unseen))

    print(scores)
    print("Best Metric: ", metrics[scores.index(max(scores))], "Accuracy: ", max(scores))
    best_metric = metrics[scores.index(max(scores))]

    # torch.save(model.bert_model.state_dict(), '../../model_checkpoints/dnabert.pth')
