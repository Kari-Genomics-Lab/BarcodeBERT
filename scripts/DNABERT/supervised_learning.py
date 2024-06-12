import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from bert_with_prediction_head import train_and_eval
from model import load_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
random.seed(10)


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


def load_data(args):
    train = pd.read_csv(f"{args.input_path}/supervised_train.csv")
    target_level = "species_name"

    x_train = train["nucleotides"]
    y_train = train[target_level]

    test = pd.read_csv(f"{args.input_path}/supervised_test.csv")

    x_val = test["nucleotides"]
    y_val = test[target_level]

    label_set = sorted(set(y_train.tolist()))
    y_train = np.array([label_set.index(y) for y in y_train])
    y_val = np.array([label_set.index(y) for y in y_val])

    num_class = len(label_set)

    return x_train, y_train, x_val, y_val, num_class


def extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels):
    all_label = np.unique(labels)
    all_label.sort()
    dict_emb = {}

    with torch.no_grad():
        model.eval()
        pbar = tqdm(enumerate(labels), total=len(labels))
        for i, label in pbar:
            pbar.set_description("Extracting features: ")
            _barcode = barcodes[i]
            if args.model == "dnabert2":
                x = sequence_pipeline(_barcode).to(device)
                x = model(x)[-1]
            else:
                x = torch.tensor(sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
                _, x = model(x)
                x = x.squeeze()

            x = x.cpu().numpy()

            if str(label) not in dict_emb.keys():
                dict_emb[str(label)] = []
            dict_emb[str(label)].append(x)

    class_embed = []
    for i in all_label:
        class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
    class_embed = np.array(class_embed, dtype=object)
    class_embed = class_embed.T.squeeze()

    # save results
    os.makedirs(args.output_dir, exist_ok=True)

    if args.using_aligned_barcode:
        np.savetxt(
            os.path.join(args.output_dir, "dna_embedding_supervised_aligned.csv"),
            class_embed,
            delimiter=",",
        )
    else:
        np.savetxt(
            os.path.join(args.output_dir, "dna_embedding_supervised.csv"),
            class_embed,
            delimiter=",",
        )

    print("DNA embeddings is saved.")


def construct_dataloader(X_train, X_val, y_train, y_val, batch_size, tokenizer, pre_tokenize):
    train_dataset = DNADataset(X_train, y_train, tokenizer, pre_tokenize)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = DNADataset(X_val, y_val, tokenizer, pre_tokenize)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--model", choices=["bioscanbert", "dnabert", "dnabert2"], default="bioscanbert")
    parser.add_argument("--checkpoint", default="bert_checkpoint/5-mer/model_41.pth", type=str)
    parser.add_argument("--output_dir", type=str, default="../data/INSECT/")
    parser.add_argument("--using_aligned_barcode", default=False, action="store_true")
    parser.add_argument("--n_epoch", default=5, type=int)
    parser.add_argument("-k", "--kmer", default=6, type=int, dest="k", help="k-mer value for tokenization")
    parser.add_argument(
        "--batch-size", default=16, type=int, dest="batch_size", help="batch size for supervised training"
    )
    parser.add_argument(
        "--model-output", default=None, type=str, dest="model_out", help="path to save model after training"
    )

    args = parser.parse_args()

    x_train, y_train, x_val, y_val, num_classes = load_data(args)

    model, sequence_pipeline = load_model(args, k=args.k, classification_head=True, num_classes=num_classes)

    train_loader, val_loader = construct_dataloader(
        x_train,
        x_val,
        y_train,
        y_val,
        args.batch_size,
        sequence_pipeline,
        pre_tokenize=args.model in {"dnabert", "dnabert2"},
    )

    train_and_eval(model, train_loader, val_loader, device=device, n_epoch=args.n_epoch)

    # extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels)

    torch.save(model.bert_model.state_dict(), f"../../model_checkpoints/dnabert_{args.k}.pth")
