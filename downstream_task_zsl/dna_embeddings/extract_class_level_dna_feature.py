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
from tqdm import tqdm
import h5py
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
    seen_train_df = pd.read_csv(os.path.join(args.input_dir, args.taxonomy_level + "_seen_train.tsv"), sep ='\t')
    
    seen_test_df = pd.read_csv(os.path.join(args.input_dir, args.taxonomy_level + "_seen_test.tsv"), sep ='\t')
    easy_unseen_test_df = pd.read_csv(os.path.join(args.input_dir, args.taxonomy_level + "_easy_unseen_test.tsv"), sep ='\t')
    hard_unseen_test_df = pd.read_csv(os.path.join(args.input_dir, args.taxonomy_level + "_hard_unseen_test.tsv"), sep ='\t')
    
    x_seen_train = seen_train_df['nucraw'].values
    x_seen_test = seen_test_df['nucraw'].values
    x_easy_unseen_test = easy_unseen_test_df['nucraw'].values
    x_hard_unseen_test = hard_unseen_test_df['nucraw'].values
    all_X = np.concatenate((x_seen_train, x_seen_test, x_easy_unseen_test, x_hard_unseen_test), axis=0)
    
    
    all_X = np.array([s.upper() for s in all_X])
    
    
    all_X = dna_barcode_to_one_hot(args, all_X)
    
    
    
    all_X = np.expand_dims(all_X, axis=3)
    
    y_seen_train = seen_train_df[args.taxonomy_level].values
    y_seen_test = seen_test_df[args.taxonomy_level].values
    y_easy_unseen_test = easy_unseen_test_df[args.taxonomy_level].values
    y_hard_unseen_test = hard_unseen_test_df[args.taxonomy_level].values
    labels = np.concatenate((y_seen_train, y_seen_test, y_easy_unseen_test, y_hard_unseen_test), axis=0)
    
    number_of_classes = len(np.unique(y_seen_train))
    

    return torch.Tensor(all_X), labels, number_of_classes


def get_embedding(model, all_X):
    embedding = None
    with torch.no_grad():
        pbar = tqdm(all_X)
        for inputs in pbar:
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
    parser.add_argument('--ckpt_path', type=str, default="../ckpt/dna_encoder/genus/2023_09_07_10_59_43/model.pth")
    # for species level: '../ckpt/dna_encoder/species/2023_09_07_09_18_55/model.pth'
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--pad_to', type=int, default=936)
    parser.add_argument('--taxnomy_level', type=str, default="genus")
    parser.add_argument('--output_dir', type=str, default="../data/bioscan_1m/extracted_feature", help="The path used to store the checkpoint.")
    args = parser.parse_args()

    all_X, labels, total_number_of_classes = load_data(args)
    model = Model(1, total_number_of_classes, dim=2640, embedding_dim=768).to(device)
    
    model.load_state_dict(torch.load(args.ckpt_path))
    dna_embeddings = get_embedding(model, all_X)
    dict_emb = {}
    
    
    for index, label in enumerate(labels):
        if label not in dict_emb.keys():
            dict_emb[label] = []
        dict_emb[label].append(np.array(dna_embeddings[index]))
    
    class_embed = {}
    for i in dict_emb.keys():
        class_embed[i] = np.sum(dict_emb[i], axis=0) / len(dict_emb[i])
    
    print(class_embed.keys())
    print(len(list(class_embed.keys())))
    print(class_embed[list(class_embed.keys())[0]].shape)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.taxnomy_level + "_dna_feature.hdf5")
    feature_name = args.taxnomy_level + "_class_level_dna_feature"
    
    
    
    with h5py.File(output_path, "a") as hdf5:
        if feature_name in hdf5:  # feature already exists; remove it
            del hdf5[feature_name]
        group = hdf5.create_group(feature_name)
        
        for class_name in class_embed.keys():
            group.create_dataset(class_name, data=class_embed[class_name])
            
    print(args.taxnomy_level + ' level DNA feature is saved in: ' + output_path)
    