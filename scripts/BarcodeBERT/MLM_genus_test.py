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

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

import matplotlib.pyplot as plt
import argparse

#import umap
import seaborn as sns
from transformers import BertForMaskedLM
from transformers import BertConfig
import time
import pickle
from sklearn.preprocessing import normalize




random.seed(10)
start = time.time()
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
    
def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # 去除 'module.' 前缀
        new_state_dict[key] = value
    return new_state_dict

def data_from_df(df, target_level, model):
    
    barcodes =  df['nucleotides'].to_list()
    targets  =  df[target_level].to_list()
    orders = test['order_name'].to_list()

    label_set=sorted(list(set(targets)))
    #print(label_set)

    #label_pipeline = lambda x: label_set.index(x)
    label_pipeline = lambda x: x
    sequence_pipeline = lambda x: vocab(tokenizer(PAD(x)))
        
    dna_embeddings = []
    labels=[]

    with torch.no_grad():
        for i, _barcode in enumerate(barcodes):
            x = torch.tensor(sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
            x = model(x).hidden_states[-1]
            x = x.mean(1)   #Global Average Pooling excluding CLS token

            #x = x[:,1:,:].mean(1)   #Global Average Pooling excluding CLS token

            #x = model.activ1(model.fc(x[:, 0])) #Representation of the CLS token

            dna_embeddings.extend(x.cpu().numpy())
            labels.append(label_pipeline(targets[i]))

    print(f"There are {len(dna_embeddings)} points in the dataset")
    latent = np.array(dna_embeddings).reshape(-1,768)
    y = np.array(labels)
    print(latent.shape)
    return latent, y, orders


    
"""# Evaluate

"""

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
k = int(sys.argv[1])


if os.path.isfile(f"../../data/representations/{k}_train.pkl") and os.path.isfile(f"../../data/representations/{k}_unseen.pkl"):
    print(f'Representations found  after {time.time()-start} seconds . . .')
    with open(f"../../data/representations/{k}_unseen.pkl", 'rb') as f:
        X_unseen,y_unseen = pickle.load(f)
        
    with open(f"../../data/representations/{k}_train.pkl", "rb") as f:
        X,y = pickle.load(f)
        
    print(f'Loaded representations  after {time.time()-start} seconds . . .')
    print(X_unseen.shape, X.shape)
    
    
else:
    max_len = 660
    tokenizer = kmer_tokenizer(k, stride=k)  #Non overlapping k-mers
    PAD = pad_sequence(max_len)

    kmer_iter = ([''.join(kmer)] for kmer in  product('ACGT',repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>","<CLS>","<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])



    vocab_size = len(vocab)

    print("Initializing the model . . .")

    configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)
    # Initializing a model (with random weights) from the bert-base-uncased style configuration

    model = BertForMaskedLM(configuration)

    configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)

    state_dict = torch.load(f'../../model_checkpoints/model_{k}.pth', map_location='cuda:0')
    #state_dict = torch.load(f'../../model_checkpoints/model_{k}.pth', map_location='cpu')

    adjusted_state_dict = {k: v for k, v in state_dict.items() if 'pooler' not in k}
    single_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.', '')
        single_state_dict[new_key] = state_dict[key]
    model.load_state_dict(single_state_dict, strict=False)


    print(f'The model has been succesfully loaded . . . after {time.time()-start} seconds')
    model.to(device)
    model.eval()
    
    train = pd.read_csv("../../data/supervised_train.csv")
    test = pd.read_csv("../../data/unseen.tsv", sep='\t')
    target_level='genus_name'


    X_unseen, y_unseen, orders = data_from_df(test, target_level, model)
    file = open(f"../../data/representations/{k}_unseen.pkl", "wb")
    pickle.dump((X_unseen,y_unseen), file)
    file.close()

    print(f'Unseen representations saved after {time.time()-start} seconds . . .')

    X, y, train_orders = data_from_df(train, target_level, model)
    file = open(f"../../data/representations/{k}_train.pkl", "wb")
    pickle.dump((X,y), file)
    file.close()

    print(f'Training representations saved after {time.time()-start} seconds . . .')

    c = 0
    for label in y_unseen:
        if not label in y:
            c += 1

    print(f'There are {c} genus that are not present during training')


#X_unseen, y_unseen, orders = X_unseen[:10], y_unseen[:10], orders[:10]
#X, y, train_orders = X[:10], y[:10], train_orders[:10]
#print(X_unseen.shape, X.shape)

#print(f"Getting the visual embedding after {time.time()-start} seconds")
#embedding = umap.UMAP(random_state=42).fit_transform(X_unseen)


# plt.title("Representation space of test sequences \n colored by order")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# sns.scatterplot(x=embedding[:,0], y=embedding[:, 1], hue=orders, s=2, legend='auto')
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# plt.tight_layout()

# plt.savefig('Pre_Training_BERT_GAP_embeddings.png',dpi=150)

## 1 - Nearest Neighbors
print(f"Computing Nearest Neighbors after {time.time()-start} seconds")

#X = normalize(X, axis=1, norm='l2')
#X_unseen = normalize(X_unseen, axis=1, norm='l2')

#print(y)
#print(y_unseen)

metrics = ['cosine']
scores = []
neighbour_size = 1
for metric in metrics:
    neigh = KNeighborsClassifier(n_neighbors=neighbour_size, metric=metric)
    neigh.fit(X, y)
    scores.append(neigh.score(X_unseen, y_unseen))

print(scores)
print("Best Metric: ", metrics[scores.index(max(scores))], "Accuracy: ", max(scores))
best_metric = metrics[scores.index(max(scores))]
print(f'The code finished after {time.time()-start} seconds')

