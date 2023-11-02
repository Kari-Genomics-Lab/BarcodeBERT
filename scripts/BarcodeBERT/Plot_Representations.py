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


#Load the model every time just because .... 
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

state_dict = torch.load(f'../model_checkpoints/model_{k}.pth', map_location='cuda:0')

adjusted_state_dict = {k: v for k, v in state_dict.items() if 'pooler' not in k}
single_state_dict = {}
for key in state_dict:
    new_key = key.replace('module.', '')
    single_state_dict[new_key] = state_dict[key]
model.load_state_dict(single_state_dict, strict=False)

print(f'The model has been succesfully loaded . . . after {time.time()-start} seconds')
model.to(device)
model.eval()


train_file = f"../data/representations/{k}_train.pkl"
test_file = f"../data/representations/{k}_test.pkl"

target_level='species_name'

if os.path.isfile(train_file):
    print(f'Representations found  after {time.time()-start} seconds . . .')
    with open(train_file, 'rb') as f:
        X, y = pickle.load(f)
        train = pd.read_csv('../data/supervised_train.tsv', sep='\t')
        y = train[target_level]
else:  
    train = pd.read_csv('../data/supervised_train.tsv', sep='\t')
    X, y, train_orders = data_from_df(train, target_level, model)
    file = open(f"../data/representations/{k}_train.pkl", "wb")
    pickle.dump((X,y), file)
    file.close()
    

if os.path.isfile(test_file):
    print(f'Representations found  after {time.time()-start} seconds . . .')
    with open(test_file, 'rb') as f:
        X_test, y_test = pickle.load(f)
        test= pd.read_csv('../data/supervised_test.tsv', sep='\t')
        y_test = test[target_level]

else:
    test = pd.read_csv('../data/supervised_test.tsv', sep='\t')
    X_test, y_test, orders = data_from_df(test, target_level, model)
    file = open(f"../data/representations/{k}_unseen.pkl", "wb")
    pickle.dump((X_test,y_test), file)
    file.close()

    
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import umap


# Set the font to Times New Roman
plt.rcParams['font.family'] = 'serif'
embedding = umap.UMAP(random_state=42).fit_transform(X_test)

plt.title("MLM representation space of testing sequences \n colored by order")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
#sns.scatterplot(x=embedding[:,0], y=embedding[:, 1], hue=train_orders, s=2, legend='auto')
sns.scatterplot(x=embedding[:,0], y=embedding[:, 1], hue=orders, s=2, legend='auto')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()

#plt.savefig('1D_CNN_embeddings.png',dpi=150)
plt.savefig('MLM_embeddings.pdf', format='pdf', bbox_inches='tight', dpi=150)

    
    