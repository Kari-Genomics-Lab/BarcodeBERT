import argparse
import os
import random
from typing import Callable, Union

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from barcodebert.bzsl.feature_extraction import (
    extract_clean_barcode_list,
    extract_clean_barcode_list_for_aligned,
    extract_dna_features,
)
from barcodebert.bzsl.models import load_model

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
random.seed(10)


class DNADataset(Dataset):
    """Load tokenized barcode samples for training and validation."""

    def __init__(
        self,
        barcodes: np.ndarray,
        labels: np.ndarray,
        tokenizer: Callable[[Union[list[str], str]], torch.Tensor],
        pre_tokenize: bool = False,
    ):
        # Vocabulary
        self.barcodes = barcodes
        self.labels = labels
        self.pre_tokenize = pre_tokenize
        self.tokenizer = tokenizer

        # if self.pre_tokenize, applies tokenizer to entire list of barcodes up front rather than in __getitem__
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


def load_data(
    data_path: str, aligned: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, int]:
    """_summary_

    :param data_path: path to .mat data input file
    :param aligned: if True, uses aligned barcodes, else uses raw unaligned barcodes
    :return: tuple of
     * x_train - array of barcodes for training
     * y_train - class labels as integers for training
     * x_val - array of barcodes for testing
     * y_val - class labels as integers for testing
     * barcodes - list of all barcodes
     * labels - array of all class labels, corresponding to barcodes
     * num_classes - number of class labels
    """
    x = sio.loadmat(data_path)

    if aligned:
        barcodes = extract_clean_barcode_list_for_aligned(x["nucleotides_aligned"])
    else:
        barcodes = extract_clean_barcode_list(x["nucleotides"])
    labels = x["labels"].squeeze() - 1

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index = None
    val_index = None
    for train_split, val_split in stratified_split.split(barcodes, labels):
        train_index = train_split
        val_index = val_split

    x_train = np.array([barcodes[i] for i in train_index])
    x_val = np.array([barcodes[i] for i in val_index])
    y_train = np.array([labels[i] for i in train_index])
    y_val = np.array([labels[i] for i in val_index])

    num_classes = np.unique(labels).shape[0]

    return x_train, y_train, x_val, y_val, barcodes, labels, num_classes


def construct_dataloader(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    tokenizer: Callable[[Union[list[str], str]], torch.Tensor],
    pre_tokenize: bool,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = DNADataset(X_train, y_train, tokenizer, pre_tokenize)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    val_dataset = DNADataset(X_val, y_val, tokenizer, pre_tokenize)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_dataloader, val_dataloader


def categorical_cross_entropy(outputs, target, num_classes=1213):
    m = nn.Softmax(dim=1)
    pred_label = torch.log(m(outputs))
    target_label = F.one_hot(target, num_classes=num_classes)

    loss = (-pred_label * target_label).sum(dim=1).mean()
    return loss


def train_and_eval(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    device,
    lr: float = 0.005,
    n_epoch: int = 12,
    num_classes: int = 1213,
):
    """Train and evaluate model on supervised task.

    :param model: model to train
    :param trainloader: training dataloader
    :param testloader: testing dataloader
    :param device: device on which to run model
    :param lr: learning rate, defaults to 0.005
    :param n_epoch: number of epochs to train, defaults to 12
    :param num_classes: number of classes in output, defaults to 1213 (INSECT dataset)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    loss = None
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        # model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for _, (inputs, labels) in pbar:
            if loss is not None:
                pbar.set_description("Epoch: " + str(epoch) + " || loss: " + str(loss.item()))
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, _ = model(inputs)

            loss = categorical_cross_entropy(outputs, labels, num_classes=num_classes)

            loss.backward()

            optimizer.step()
            # print statistics
            running_loss += loss.item()

        with torch.no_grad():
            # model.eval()
            train_correct = 0
            train_total = 0
            for data in trainloader:
                inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                labels = labels.int()
                # calculate outputs by running images through the network
                outputs, _ = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.int()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            correct = 0
            total = 0
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                labels = labels.int()
                # calculate outputs by running images through the network
                outputs, _ = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.int()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            "Epoch: "
            + str(epoch)
            + "|| loss: "
            + str(running_loss / len(trainloader))
            + "|| Accuracy: "
            + str(train_correct / train_total)
            + "|| Val Accuracy: "
            + str(correct / total)
            + "|| lr: "
            + str(scheduler.get_last_lr())
        )
        running_loss = 0
        scheduler.step()

    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--model", choices=["bioscanbert", "barcodebert", "dnabert", "dnabert2"], default="barcodebert")
    parser.add_argument("--checkpoint", default="bert_checkpoint/5-mer/model_41.pth", type=str)
    parser.add_argument("--output_dir", type=str, default="../data/INSECT/")
    parser.add_argument("--using_aligned_barcode", default=False, action="store_true")
    parser.add_argument("--n_epoch", default=12, type=int)
    parser.add_argument("-k", "--kmer", default=6, type=int, dest="k", help="k-mer value for tokenization")
    parser.add_argument(
        "--batch-size", default=32, type=int, dest="batch_size", help="batch size for supervised training"
    )
    parser.add_argument(
        "--model-output", default=None, type=str, dest="model_out", help="path to save model after training"
    )
    parser.add_argument(
        "-s",
        "--save-all",
        action="store_true",
        dest="extract_all_embeddings",
        help="if specified, save all embeddings rather than average class embedding",
    )

    args = parser.parse_args()

    x_train, y_train, x_val, y_val, barcodes, labels, num_classes = load_data(
        args.input_path, args.using_aligned_barcode
    )

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

    train_and_eval(model, train_loader, val_loader, device=device, n_epoch=args.n_epoch, num_classes=num_classes)

    output = os.path.join(args.output_dir, "dna_embedding_supervised.csv")
    extract_dna_features(
        args.model, model, sequence_pipeline, barcodes, labels, output, not args.extract_all_embeddings
    )

    if args.model_out:
        torch.save(model.bert_model.state_dict(), args.model_out)
