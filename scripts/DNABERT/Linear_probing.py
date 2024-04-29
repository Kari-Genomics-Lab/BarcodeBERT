import argparse
import os
import random

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from model import load_model
from bert_extract_dna_feature import extract_clean_barcode_list, extract_clean_barcode_list_for_aligned
from bert_with_prediction_head import train_and_eval
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle
import time

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


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


def extract_features(args, file, model, sequence_pipeline):
    
    df = pd.read_csv(f"{args.input_path}/{file}")
    target_level='species_name'

    barcodes =  df['nucleotides']
    targets  =  df[target_level]

    label_set=sorted(list(set(targets.tolist())))
    targets = np.array(list(map(lambda x: label_set.index(x), targets)))  
    
    dna_embeddings = []
    labels=[]

    with torch.no_grad():
        for i, _barcode in enumerate(barcodes):
            x = torch.tensor(sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
            x = model(x).hidden_states[-1]
            x = x.mean(1)   #Global Average
            #print(x.shape)
            dna_embeddings.extend(x.cpu().numpy())
            labels.append(targets[i])

                
    print(f"There are {len(dna_embeddings)} points in the dataset")
    latent = np.array(dna_embeddings).reshape(-1,768)
    y = np.array(labels)
    #print(latent.shape)
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
    test_file = f"embeddings/{args.k}_test.pkl"
    
    if os.path.isfile(train_file):
        print(f'Representations found  after {time.time()-start} seconds . . .')
        with open(train_file, 'rb') as f:
            X, y = pickle.load(f)
            train = pd.read_csv('../../data/supervised_train.csv')
            y = train['species_name']

    else: 

        print("Loading the model.....")
        model, sequence_pipeline = load_model(args, k=args.k)

        X, y = extract_features(args,'supervised_train.csv', model, sequence_pipeline)
        file = open(train_file, "wb")
        pickle.dump((X,y), file)
        file.close()

        
    if os.path.isfile(test_file):
        print(f'Representations found  after {time.time()-start} seconds . . .')
        with open(test_file, 'rb') as f:
            X_test, y_test = pickle.load(f)
            test= pd.read_csv('../../data/supervised_test.csv')
            y_test = test['species_name']
            

    else: 

        print("Loading the model.....")
        model, sequence_pipeline = load_model(args, k=args.k)

        X_test, y_test = extract_features(args,'supervised_test.csv', model, sequence_pipeline)
        file = open(test_file, "wb")
        pickle.dump((X_test,y_test), file)
        file.close()
    
    ## Training Linear Classifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import Perceptron
    print("training the classifier")
    # clf = MLPClassifier(random_state=1, max_iter=1, 
    #                     verbose=True, early_stopping=True,
    #                     learning_rate='adaptive').fit(X, y)
    clf = Perceptron(tol=1e-3, random_state=0, verbose=False, early_stopping=True).fit(X, y)
    print(f'Test Accuracy: {clf.score(X_test, y_test)}')
