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
from resource import *

#import umap
import seaborn as sns
from transformers import BertForTokenClassification
from transformers import BertConfig
import time
import pickle
from sklearn.preprocessing import normalize
from sklearn.linear_model import Perceptron

random.seed(10)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
start = time.time()
"""# DNA vocab tools """
class KmerTokenizer(object):
    def __init__(self, k, vocabulary_mapper, stride=1, padding=False, max_len=660):
        self.k = k
        self.stride = stride
        self.padding = padding
        self.max_len = max_len
        self.vocabulary_mapper = vocabulary_mapper

    def __call__(self, dna_sequence) -> list:
        tokens = []
        att_mask = [1] * (self.max_len//self.stride)
        if self.padding:
            if len(dna_sequence) > self.max_len:
                x = dna_sequence[:self.max_len]
            else:
                x = dna_sequence + 'N' * (self.max_len - len(dna_sequence))
                att_mask[len(dna_sequence)//self.stride:] = [0]* (len(att_mask)-len(dna_sequence)//self.stride)
        else:
            x = dna_sequence

        for i in range(0, len(x) - self.k + 1, self.stride):
            k_mer = x[i:i + self.k]
            tokens.append(k_mer)
            
        tokens = torch.tensor(self.vocabulary_mapper(tokens), dtype=torch.int64)
        att_mask = torch.tensor(att_mask, dtype=torch.int32)
        
        return tokens, att_mask 

    
def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  
        new_state_dict[key] = value
    return new_state_dict
    
def data_from_df(df, target_level, model, tokenizer):
    
    barcodes =  df['nucleotides'].to_list()
    targets  =  df[target_level].to_list()
    orders = df['order_name'].to_list()

    label_set=sorted(list(set(targets)))
    #print(label_set)

    label_pipeline = lambda x: label_set.index(x)
           
    dna_embeddings = []
    labels=[]

    with torch.no_grad():
        for i, _barcode in enumerate(barcodes):
            
            x, att_mask  = tokenizer(_barcode)
            
            x = x.unsqueeze(0).to(device)
            att_mask = att_mask.unsqueeze(0).to(device)
            x = model(x, att_mask).hidden_states[-1]
            x = x.mean(1)   

            dna_embeddings.extend(x.cpu().numpy())
            labels.append(label_pipeline(targets[i]))

    print(f"There are {len(dna_embeddings)} points in the dataset")
    latent = np.array(dna_embeddings).reshape(-1,768)
    y = np.array(labels)
    print(latent.shape)
    return latent, y, orders


"""# Evaluate

"""

def run(args):

    start_time = time.time()
    k = args['k_mer']
    
    representation_folder=f"../data/soft_{args['n_layers']}_{args['n_layers']}"
    if not os.path.exists(representation_folder):
        os.mkdir(representation_folder)
        
    # Vocabulary
    max_len = args['max_len']
    kmer_iter = ([''.join(kmer)] for kmer in product('ACGTN', repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>"])  
    vocab.set_default_index(vocab['N'*k])  ### <UNK> and <CLS> do not exist anymore
    vocab_size = len(vocab)
    
    tokenizer = KmerTokenizer(k, vocab, stride=k, padding=True, max_len=max_len)
    
    print("Initializing the model . . .")
    configuration = BertConfig(vocab_size=vocab_size,
                               num_hidden_layers=args['n_layers'],
                               num_attention_heads=args['n_heads'],
                               num_labels = 4**args['k_mer'],
                               output_hidden_states=True)

    #model = BertForMaskedLM(configuration)
    model =  BertForTokenClassification(configuration)   
    state_dict = remove_extra_pre_fix(torch.load(args['Pretrained_checkpoint_path'], map_location="cuda:0"))
    model.load_state_dict(state_dict, strict=True)

    print(f'The model has been succesfully loaded . . . after {time.time()-start} seconds')
    model.to(device)
    model.eval()


    train_file = f"{representation_folder}/{k}_train.pkl"
    test_file = f"{representation_folder}/{k}_test.pkl"

    target_level='species_name'

    if os.path.isfile(train_file):
        print(f'Representations found  after {time.time()-start} seconds . . .')
        with open(train_file, 'rb') as f:
            X, y = pickle.load(f)
            train = pd.read_csv(f'{args["input_path"]}/supervised_train.csv', sep=',')
            
            targets = train[target_level].to_list()
            label_set = sorted(list(set(targets)))
            label_pipeline = lambda x: label_set.index(x)
            y = list(map(label_pipeline, targets))
    else:  
        train = pd.read_csv(f'{args["input_path"]}/supervised_train.csv', sep=',')
        X, y, train_orders = data_from_df(train, target_level, model, tokenizer)
        file = open(f"{representation_folder}/{k}_train.pkl", "wb")
        pickle.dump((X,y), file)
        file.close()


    if os.path.isfile(test_file):
        print(f'Representations found  after {time.time()-start} seconds . . .')
        with open(test_file, 'rb') as f:
            X_test, y_test = pickle.load(f)
            test= pd.read_csv(f'{args["input_path"]}/supervised_test.csv', sep=',')
            targets = test[target_level].to_list()
            label_set = sorted(list(set(targets)))
            label_pipeline = lambda x: label_set.index(x)
            y_test = list(map(label_pipeline, targets))

    else:
        test = pd.read_csv(f'{args["input_path"]}/supervised_test.csv', sep=',')
        X_test, y_test, orders = data_from_df(test, target_level, model,tokenizer)
        file = open(f"{representation_folder}/{k}_test.pkl", "wb")
        pickle.dump((X_test,y_test), file)
        file.close()


    ## Training Linear Classifier
    #from sklearn.neural_network import MLPClassifier
    #print("training the classifier")
    #clf = MLPClassifier(random_state=1, max_iter=100).fit(X, y)
    #print(f'Test Accuracy: {clf.score(X_test, y_test)}')
    
    running_info = getrusage(RUSAGE_SELF)
    _time = (time.time() - start_time) #running_info.ru_utime + running_info.ru_stime
    hour = _time // 3600
    minutes = (_time  - (3600 * hour)) // 60
    seconds = _time - (hour * 3600) - (minutes * 60)
    memory = (running_info.ru_maxrss/1e6)
    print(f'Learning the representations took: {int(hour)}:{int(minutes)}:{round(seconds)} (hh:mm:ss)\n')
    
    print(f'Max memory usage: {memory} (GB)')
    print(np.unique(y).shape[0], np.unique(y_test).shape[0])


    print("training the classifier")
    #clf = MLPClassifier(random_state=1, max_iter=1, 
    #                    verbose=True, early_stopping=True,
    #                    learning_rate='adaptive').fit(X, y)
    clf = Perceptron(tol=1e-3, random_state=0, verbose=False, early_stopping=True,max_iter=100).fit(X, y)
    print(f'Test Accuracy: {clf.score(X_test, y_test)}')
    
    
    
    # Normalize the features 
    mean = X.mean() 
    std = X.std() 
    X = (X - mean) / std 
    X_test = (X_test - mean) / std 

    # Define the model 
    clf = torch.nn.Sequential( 
        torch.nn.Linear(768, np.unique(y).shape[0]), 
        torch.nn.Softmax(dim=1) 
    ) 

    clf.to(device)
    X_train = torch.tensor(X).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    y_train = torch.tensor(y).to(device)
    y_test = torch.tensor(y_test).to(device)
    
    # Train the model 
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

    num_epochs = 1000
    for epoch in range(num_epochs): 
        # Forward pass 
        y_pred = clf(X_train) 
        loss = criterion(y_pred, y_train) 

        # Backward pass and optimization 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        # Print the loss every 100 epochs 
        if (epoch+1) % 100 == 0: 
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 

    # Evaluate the model 
    with torch.no_grad(): 
        y_pred = clf(X_test) 
        _, predicted = torch.max(y_pred, dim=1) 
        accuracy = (predicted == y_test).float().mean() 
        print(f'Test Accuracy: {accuracy.item():.4f}') 

        _time = (time.time() - start_time) #running_info.ru_utime + running_info.ru_stime
        hour = _time // 3600
        minutes = (_time  - (3600 * hour)) // 60
        seconds = _time - (hour * 3600) - (minutes * 60)
        print(f'The code finished after: {int(hour)}:{int(minutes)}:{round(seconds)} (hh:mm:ss)\n')

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)
    parser.add_argument('--Pretrained_checkpoint_path', action='store', type=str)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--stride', action='store', type=int, default=4)
    parser.add_argument('--max_len', action='store', type=int, default=720)
    parser.add_argument('--n_layers', action='store', type=int, default=12)
    parser.add_argument('--n_heads', action='store', type=int, default=12)

    args = vars(parser.parse_args())

    #sys.stdout.write("\Evaluation Parameters:\n")
    #for key in args:
    #    sys.stdout.write(f'{key} \t -> {args[key]}\n')

    run(args)

    
if __name__ == '__main__':
    main()