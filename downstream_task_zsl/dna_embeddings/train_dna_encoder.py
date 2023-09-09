import os
import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from CNN import Model, train_and_eval
from sklearn.preprocessing import LabelEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUCLEOTIDE_DICT = {'A': 0, 'C':1, 'G':2, 'T':3, 'N':4}

def dna_barcode_to_one_hot(args, barcodes):

    one_hot_array = np.zeros((barcodes.shape[0], args.pad_to, len(NUCLEOTIDE_DICT)), dtype=int)

    for i, barcode in enumerate(barcodes):
        barcode = barcode[:args.pad_to].ljust(args.pad_to, '-')
        for j, char in enumerate(barcode):
            if char in NUCLEOTIDE_DICT.keys():
                one_hot_array[i, j, NUCLEOTIDE_DICT[char]] = 1

    
    return one_hot_array
    

def load_data(args):
    seen_train_df = pd.read_csv(os.path.join(args.input_dir, args.taxnomy_level + "_seen_train.tsv") ,sep = '\t')
    
    
    x_seen_train = seen_train_df['nucraw'].values
    x_seen_train = np.array([s.upper() for s in x_seen_train])
    y_seen_train = seen_train_df[args.taxnomy_level].values
    
    
    number_of_classes = len(np.unique(y_seen_train))
    
    x_seen_train = dna_barcode_to_one_hot(args, x_seen_train)
    label_encoder = LabelEncoder()

    y_seen_train = label_encoder.fit_transform(y_seen_train)
    x_seen_train = np.expand_dims(x_seen_train, axis=3)
    
    X_train, X_test, y_train, y_test = train_test_split(x_seen_train, y_seen_train, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, number_of_classes


def construct_dataloader(X_train, X_test, y_train, y_test, batch_size):
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    
    return train_dataloader, test_dataloader


def get_embedding(model, all_X):
    embedding = None
    with torch.no_grad():
        for inputs in all_X:
            _, feature = model(torch.unsqueeze(inputs.to(device), 0))
            if embedding is None:
                embedding = feature
            else:
                embedding = torch.cat((embedding, feature), 0)
    embedding = embedding.to('cpu')
    return embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="../data/bioscan_1m", help="path to the directory that contains the split data.")
    parser.add_argument('--model_output_dir', type=str, default="../ckpt/dna_encoder")
    parser.add_argument('--taxnomy_level', type=str, default="genus")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--max_epochs', type=int, default=12)
    parser.add_argument('--pad_to', type=int, default=936)
    # 936
    parser.add_argument('--output_dir', type=str, default="", help="The path used to store the checkpoint.")
    parser.add_argument('--number_of_workers', type=int, default=4)
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, total_number_of_classes = load_data(args)
    trainloader, testloader = construct_dataloader(X_train, X_test, y_train, y_test, args.batch_size)
    model = Model(1, total_number_of_classes, dim=2640).to(device)
    train_and_eval(args, model, trainloader, testloader, device=device, n_epoch=args.max_epochs, lr=args.learning_rate, num_classes=total_number_of_classes)
    