# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import random
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertConfig, BertForTokenClassification

from barcodebert.datasets import KmerTokenizer, representations_from_df
from barcodebert.io import get_project_root
from barcodebert.utils import remove_extra_pre_fix

random.seed(10)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

start = time.time()


def run(config):
    start_time = time.time()
    k = config.k_mer

    extra = ""
    representation_folder = os.path.join(get_project_root(), "data", f"{extra}{config.n_layers}_{config.n_layers}")
    if not os.path.exists(representation_folder):
        os.makedirs(representation_folder, exist_ok=True)

    fname_base = f"{config.k_mer}_{extra}"

    # Vocabulary
    max_len = config.max_len
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])  # <UNK> is necessary in the hard case
    vocab_size = len(vocab)

    tokenizer = KmerTokenizer(k, vocab, stride=k, padding=True, max_len=max_len)

    print("Initializing the model . . .")
    configuration = BertConfig(
        vocab_size=vocab_size,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_labels=4**config.k_mer,
        output_hidden_states=True,
    )

    model = BertForTokenClassification(configuration)
    state_dict = remove_extra_pre_fix(torch.load(config.Pretrained_checkpoint_path, map_location="cuda:0"))
    model.load_state_dict(state_dict, strict=False)

    print(f"The model has been succesfully loaded . . . after {time.time()-start} seconds")
    model.to(device)
    model.eval()

    train_file = f"{representation_folder}/{fname_base}_train.pkl"
    test_file = f"{representation_folder}/{fname_base}_test.pkl"

    target_level = "species_name"

    if os.path.isfile(train_file):
        print(f"Representations found  after {time.time()-start} seconds . . .")
        with open(train_file, "rb") as f:
            X, y = pickle.load(f)
            train = pd.read_csv(os.path.join(config.data_path, "supervised_train.csv"), sep=",")

            targets = train[target_level].to_list()
            label_set = sorted(set(targets))
            y = [label_set.index(t) for t in targets]
    else:
        train = pd.read_csv(os.path.join(config.data_path, "supervised_train.csv"), sep=",")
        X, y, train_orders = representations_from_df(train, target_level, model, tokenizer)
        file = open(f"{representation_folder}/{fname_base}_train.pkl", "wb")
        pickle.dump((X, y), file)
        file.close()

    if os.path.isfile(test_file):
        print(f"Representations found  after {time.time()-start} seconds . . .")
        with open(test_file, "rb") as f:
            X_test, y_test = pickle.load(f)
            test = pd.read_csv(os.path.join(config.data_path, "supervised_test.csv"), sep=",")
            targets = test[target_level].to_list()
            label_set = sorted(set(targets))
            y_test = [label_set.index(t) for t in targets]

    else:
        test = pd.read_csv(os.path.join(config.data_path, "supervised_test.csv"), sep=",")
        X_test, y_test, orders = representations_from_df(test, target_level, model, tokenizer)
        file = open(f"{representation_folder}/{fname_base}_test.pkl", "wb")
        pickle.dump((X_test, y_test), file)
        file.close()

    # Normalize the features
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    X = torch.tensor(X).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y)
    y_test = torch.tensor(y_test)

    print("Train shapes:", X.shape, X_test.shape)

    train = torch.utils.data.TensorDataset(X, y)
    train_loader = DataLoader(train, batch_size=1024, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    # test_loader = DataLoader(test, batch_size=1024, shuffle=False, drop_last=False)

    # Define the model
    clf = torch.nn.Sequential(torch.nn.Linear(768, np.unique(y).shape[0]))

    clf.to(device)

    # Train the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

    num_epochs = 200
    for epoch in range(num_epochs):

        for X_train, y_train in train_loader:

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            # Forward pass
            y_pred = clf(X_train)
            loss = criterion(y_pred, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    with torch.no_grad():
        y_pred = clf(X_test)
        print(y_pred.shape)
        _, predicted = torch.max(y_pred, dim=1)
        print(predicted, y_test)
        accuracy = (predicted == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item():.4f}")

    _time = time.time() - start_time  # running_info.ru_utime + running_info.ru_stime
    hour = _time // 3600
    minutes = (_time - (3600 * hour)) // 60
    seconds = _time - (hour * 3600) - (minutes * 60)
    print(f"The code finished after: {int(hour)}:{int(minutes)}:{round(seconds)} (hh:mm:ss)\n")

    with open("LINEAR_RESULTS.txt", "a") as f:
        model_name = config.Pretrained_checkpoint_path.split(".")[-2].split("/")[-1]
        f.write(f"\n{model_name} \t {accuracy.item():.4f}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--Pretrained_checkpoint_path", action="store", type=str)
    parser.add_argument("--k_mer", action="store", type=int, default=4)
    parser.add_argument("--stride", action="store", type=int, default=4)
    parser.add_argument("--max_len", action="store", type=int, default=660)
    parser.add_argument("--n_layers", action="store", type=int, default=12)
    parser.add_argument("--n_heads", action="store", type=int, default=12)
    return parser


def main():
    parser = get_parser()
    config = parser.parse_args()

    # sys.stdout.write("\Evaluation Parameters:\n")
    # for key in config:
    #    sys.stdout.write(f'{key} \t -> {getattr(config, key)}\n')

    run(config)


if __name__ == "__main__":
    main()
