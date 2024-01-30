# -*- coding: utf-8 -*-
import pandas as pd
import os
import scipy.io as sio
import numpy as np
import random
from random import randrange, shuffle, random, randint
from itertools import product
import sys
import argparse
from transformers import BertForPreTraining, BertForMaskedLM, BertForTokenClassification
from transformers import BertConfig

import torch
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
import time
from resource import *

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

"""# Load data and tokenize """


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


class DNADataset(Dataset):
    def __init__(self, file_path, k_mer=4, stride=4, max_len=256):
        self.k_mer = k_mer
        self.stride = stride
        self.max_len = max_len

        # Vocabulary
        kmer_iter = ([''.join(kmer)] for kmer in product('ACGTN', repeat=self.k_mer))
        self.vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>"])  
        self.vocab.set_default_index(self.vocab['N'*self.k_mer])  ### <UNK> and <CLS> do not exist anymore
        self.vocab_size = len(self.vocab)

        self.tokenizer = KmerTokenizer(self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len)

        train_csv = pd.read_csv(file_path, sep='\t')
        self.barcodes = train_csv['nucleotides'].to_list()

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx): 
        return self.tokenizer(self.barcodes[idx]) 

def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # 去除 'module.' 前缀
        new_state_dict[key] = value
    return new_state_dict

""" Auxiliary function to load the model"""
def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  #
        new_state_dict[key] = value
    return new_state_dict
    
""" Soft Ordinal Labels """ 
def optimized_levenshtein_distance(num1, num2, k):
    #Calculate the Levenshtein distance between two k-mers represented by integers. 
    #There are two mappins, one for the input that may contain Ns and one for the output 
    #that should not contain N.
    
    num1 -= 1 ## <MASK> token is index 0
    
    mapping_in = ['A', 'C', 'G', 'T', 'N']
    mapping_out = ['A', 'C', 'G', 'T']
    distance = 0

    for _ in range(k):
        nucleotide1 = mapping_in[num1 % 5]
        if nucleotide1 != 'N':
            nucleotide2 = mapping_out[num2 % 4]   
            if nucleotide1 != nucleotide2:
                distance += 1
                
        num1 //= 5
        num2 //= 4

    return distance

def create_optimized_levenshtein_matrix(kmers_int_array, k):
    max_kmer_int = 4**k - 1
    matrix = [[0] * (max_kmer_int + 1) for _ in range(len(kmers_int_array))]

    for i, num1 in enumerate(kmers_int_array):
        for j in range(max_kmer_int + 1):
            matrix[i][j] = optimized_levenshtein_distance(num1, j, k)

    return matrix

def _softmax_batch_levenshtein_matrices(batch_kmers_int_tensor, k):
    batch_size = batch_kmers_int_tensor.size(0)
    max_kmer_int = 4**k - 1
    batch_matrices = []

    for i in range(batch_size):
        kmers_int_array = batch_kmers_int_tensor[i].tolist()
        matrix = create_optimized_levenshtein_matrix(kmers_int_array, k)
        batch_matrices.append(matrix)

    output_tensor = torch.softmax(torch.tensor(batch_matrices, dtype=torch.float32), dim=1)
    return output_tensor


def build_lookup_table(k):
    """ Build a lookup table for Levenshtein distances considering 'N' in input k-mers. """
    input_vocab_size = 5  # A, C, G, T, N
    output_vocab_size = 4  # A, C, G, T
    input_max_kmer_int = input_vocab_size**k - 1
    output_max_kmer_int = output_vocab_size**k - 1

    # Initialize the lookup table
    lookup_table = torch.zeros((input_max_kmer_int + 1, output_max_kmer_int + 1), dtype=torch.float32)

    # Compute distances for all combinations
    for i in range(input_max_kmer_int + 1):
        for j in range(output_max_kmer_int + 1):
            lookup_table[i, j] = optimized_levenshtein_distance(i, j, k)

    return lookup_table

def optimized_levenshtein_distance(num1, num2, k):
    mapping_in = ['A', 'C', 'G', 'T', 'N']
    mapping_out = ['A', 'C', 'G', 'T']
    num1 -= 1  # Adjust for <MASK> token
    distance = 0

    for _ in range(k):
        nucleotide1 = mapping_in[num1 % 5]
        if nucleotide1 != 'N':
            nucleotide2 = mapping_out[num2 % 4]
            if nucleotide1 != nucleotide2:
                distance += 1
        num1 //= 5
        num2 //= 4

    return distance

def softmax_batch_levenshtein_matrices(batch_input_tensor, lookup_table):
    """ :( """
    
    # Expand the lookup table indices to match the input tensor shape
    batch_size, seq_len = batch_input_tensor.size()
    output_size = lookup_table.size(1)

    # Gather the rows from the lookup table based on the input tensor indices
    # Adjust indices for <MASK> token in input
    adjusted_indices = batch_input_tensor - 1
    output_tensor = lookup_table[adjusted_indices]

    # Ensure the output tensor is of the shape (batch, seq_len, output_size)
    output_tensor = output_tensor.view(batch_size, seq_len, output_size)
    output_tensor = torch.softmax(output_tensor, dim=1)

    return output_tensor



"""# Train """

def train(args, dataloader, device, model, optimizer, scheduler):
    model.train()
    epoch_loss_list = []
    training_epoch = args["epoch"]
    continue_epoch = args["continue_epoch"]
    criterion = nn.CrossEntropyLoss()
    distance_table = build_lookup_table(args['k_mer']).to(device)

    saving_path = "../model_checkpoints/"
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    

    sys.stdout.write("Training is started:\n")

    for epoch in range(continue_epoch + 1, training_epoch + 1):
        epoch_loss = 0

        dataloader.sampler.set_epoch(epoch)
        for i, (batch, att_mask) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            # Build the masking on the fly every time something different
            batch = batch.to(device)
            att_mask = att_mask.to(device)
            #print(batch.size(), mask.size())
            masked_input = batch.clone()
            random_mask = torch.rand(masked_input.shape).to(device)  # I can only do this for non-overlapping
            random_mask = (random_mask < 0.5)  # Can mask anything I want mask the [<UNK>] token
            mask_idx = (random_mask.flatten() == True).nonzero().view(-1)
            masked_input = masked_input.flatten()
            masked_input[mask_idx] = 0
            masked_input = masked_input.view(batch.size())
            
            
            out = model(masked_input, attention_mask=att_mask) # Do not send the labels anymore
            #loss = out.loss
            #soft_labels = softmax_batch_levenshtein_matrices(batch, args['k_mer'])
            soft_labels = softmax_batch_levenshtein_matrices(batch, distance_table)
            loss = criterion(out.logits, soft_labels.to(device))
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(dataloader)
        epoch_loss_list.append(epoch_loss)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: lr %f -> %f" % (epoch, before_lr, after_lr))

        sys.stdout.write(f"Epoch {epoch}, Device {device}: Loss is {epoch_loss}\n")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), saving_path + f"{args['k_mer']}_soft_model_{args['n_layers']}_{args['n_heads']}_" + str(epoch) + '.pth')
            #torch.save(optimizer.state_dict(), saving_path + "optimizer_" + str(epoch) + '.pth')
            torch.save(scheduler.state_dict(), saving_path + "scheduler_" + str(epoch) + '.pth')

        a_file = open(saving_path + f"loss_{device}.pkl", "wb")
        pickle.dump(epoch_loss_list, a_file)
        a_file.close()


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader


def main(rank: int, world_size: int, args):
    
    ddp_setup(rank, world_size)

    sys.stdout.write("Loading the dataset is started.\n")
    dataset = DNADataset(file_path=args['input_path'], k_mer=args['k_mer'], stride=args['stride'],
                         max_len=args['max_len'])

    dataloader = prepare(dataset, rank, world_size=world_size, batch_size=args['batch_size'])

    sys.stdout.write("Initializing the model ...\n")
    
    configuration = BertConfig(vocab_size=dataset.vocab_size, 
                               num_hidden_layers=args['n_layers'],
                               num_attention_heads=args['n_heads'],
                               num_labels = 4**args['k_mer'],
                               output_hidden_states=True)

    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    #model = BertForMaskedLM(configuration).to(rank)
    model = BertForTokenClassification(configuration).to(rank)
    #print(model.state_dict().keys())
    
    if args['checkpoint']:
        
        state_dict = remove_extra_pre_fix(torch.load(f"../model_checkpoints/{args['k_mer']}_soft_model_{args['n_layers']}_{args['n_heads']}_" + f"{args['continue_epoch']}" + '.pth', map_location='cuda:0'))
       
        model.load_state_dict(state_dict, strict=True)

        #a_file = open(saving_path + f"loss_{device}.pkl", "rb")
        #epoch_loss_list = pickle.load(a_file)   
    
    sys.stdout.write("The model has been successfully initialized ...\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=5)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    sys.stdout.write("Let's start the training ...\n")
    start_time = time.time()
    train(args, dataloader, rank, model, optimizer, scheduler)
    destroy_process_group()
    
    running_info = getrusage(RUSAGE_SELF)
    _time = (time.time() - start_time) #running_info.ru_utime + running_info.ru_stime
    hour = _time // 3600
    minutes = (_time  - (3600 * hour)) // 60
    seconds = _time - (hour * 3600) - (minutes * 60)
    memory = (running_info.ru_maxrss/1e6)
    print(f'training took: {int(hour)}:{int(minutes)}:{round(seconds)} (hh:mm:ss) and {memory} (GB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)
    parser.add_argument('--checkpoint', action='store', type=bool, default=False)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--stride', action='store', type=int, default=4)
    parser.add_argument('--max_len', action='store', type=int, default=720)
    parser.add_argument('--batch_size', action='store', type=int, default=16)
    parser.add_argument('--lr', action='store', type=float, default=1e-4)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)
    parser.add_argument('--n_layers', action='store', type=int, default=12)
    parser.add_argument('--n_heads', action='store', type=int, default=12)
    parser.add_argument('--epoch', action='store', type=int, default=35)
    parser.add_argument('--continue_epoch', action='store', type=int, default=0)


    args = vars(parser.parse_args())

    sys.stdout.write("\nTraining Parameters:\n")
    for key in args:
        sys.stdout.write(f'{key} \t -> {args[key]}\n')

    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
