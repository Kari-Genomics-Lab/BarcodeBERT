# -*- coding: utf-8 -*-
import pandas as pd
import os
import scipy.io as sio
import numpy as np
import random

from itertools import product

import re
import sys
import argparse

import math
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import argparse

import umap
import seaborn as sns
from transformers import BertForMaskedLM
from transformers import BertConfig

random.seed(10)




"""# DNA vocab tools """
class kmer_tokenizer(object):
    
    def __init__(self, k, stride=1):
        self.k=k
        self.stride = stride

    def __call__(self, dna_sequence):
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i:i + self.k]
            tokens.append(k_mer)
        return tokens

class pad_sequence(object):
    
    def __init__(self, max_len):
        self.max_len=max_len
        
    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[:self.max_len]
        else:
            return dna_sequence + 'N'*(self.max_len-len(dna_sequence))
        
        return new_sequence    
    
"""# Evaluate

"""
test = pd.read_csv("data/unseen.tsv", sep='\t')
target_level='species_name'
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
 
k = 4
max_len = 660
tokenizer = kmer_tokenizer(k, stride=k)  #Non overlapping k-mers
PAD = pad_sequence(max_len)

kmer_iter = ([''.join(kmer)] for kmer in  product('ACGT',repeat=k))
vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>","<CLS>","<UNK>"])
vocab.set_default_index(vocab["<UNK>"])


barcodes =  test['nucleotides'].to_list()
targets  =  test[target_level].to_list()
orders = test['order_name'].to_list()


label_set=sorted(list(set(targets)))
print(label_set)

label_pipeline = lambda x: label_set.index(x)
sequence_pipeline = lambda x: vocab(tokenizer(PAD(x)))

vocab_size = len(vocab)

config = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "max_len": 512

}

num_class = 10 ## 1390 Change this according to the model this is comming from 
print("Initializing the model . . .")

configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)
# Initializing a model (with random weights) from the bert-base-uncased style configuration

model = BertForMaskedLM(configuration)


model.load_state_dict(torch.load('model_checkpoints/pre_trained_model_full.pth'))
print("The model has been succesfully loaded . . .")
model.to(device)
model.eval()

dna_embeddings = []
labels=[]

with torch.no_grad():
    for i, _barcode in enumerate(barcodes):
        x = torch.tensor([0]+sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
        x = model(x).hidden_states[-1]
        x = x.mean(1)   #Global Average Pooling excluding CLS token
        
        
        #x = x[:,1:,:].mean(1)   #Global Average Pooling excluding CLS token
        
        #x = model.activ1(model.fc(x[:, 0])) #Representation of the CLS token
        
        dna_embeddings.extend(x.cpu().numpy())
        labels.append(label_pipeline(targets[i]))
    
print(f"There are {len(dna_embeddings)} points in the dataset")
latent = np.array(dna_embeddings).reshape(-1,config["d_model"])
y_unseen = np.array(labels)
print(latent.shape)
embedding = umap.UMAP(random_state=42).fit_transform(latent)

# fig, ax = plt.subplots(nrows=1, ncols=1) 
# ax.set_title("Representation of the Latent Space")
# ax.set_xlabel("UMAP 1")
# ax.set_ylabel("UMAP 2")

# ax.scatter(embedding[:, 0],
#            embedding[:, 1],
#            color=y_unseen,
#            s=1,
#            cmap='Spectral')

plt.title("Representation space \n colored by true taxa")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
sns.scatterplot(x=embedding[:,0], y=embedding[:, 1], hue=orders, s=2, legend='auto')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()

plt.savefig('Pre_Training_BERT_GAP_embeddings.png',dpi=150)

## Histogram of Nearest Neighbors
metrics = ['manhattan', 'cosine', 'minkowski']
scores = []
neighbour_size = 1
gt = y_unseen
for metric in metrics:
    nbrs = NearestNeighbors(n_neighbors=neighbour_size+1, metric=metric).fit(embedding)
    _, neighbour_indices = nbrs.kneighbors(embedding)
    neighbour_indices = neighbour_indices.astype(np.int32)[:,1:]
    #print(neighbour_indices)
    neighbour_indices = neighbour_indices.reshape(-1)
    #print(neighbour_indices)
    
    
    gt_indices = np.array([[idx]*neighbour_size for idx in range(len(gt))]).reshape(-1).astype(np.int32)
    #print(gt_indices.shape)

    gt = np.array(gt)

    neighbor_gt = gt[neighbour_indices]
    #print(np.sum(gt[gt_indices] == gt[neighbour_indices]))

    cluster_distribution = pd.Series(gt[gt_indices]).value_counts().to_dict()
    #print(cluster_distribution)
    good_neighbours = pd.Series( gt[gt_indices][gt[gt_indices] == gt[neighbour_indices]]).value_counts().to_dict()
    #print(good_neighbours)
    score = 0
    n_clusters = 0
    for cluster in cluster_distribution:
        score += good_neighbours[cluster]/cluster_distribution[cluster]
        n_clusters += 1
    scores.append(score/n_clusters)

print(scores)
print("Best Metric: ", metrics[scores.index(max(scores))], "Accuracy: ", max(scores))
best_metric = metrics[scores.index(max(scores))]

