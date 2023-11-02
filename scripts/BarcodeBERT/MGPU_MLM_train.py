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
from transformers import BertForPreTraining, BertForMaskedLM
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
        if self.padding:
            if len(dna_sequence) > self.max_len:
                x = dna_sequence[:self.max_len]
            else:
                x = dna_sequence + 'N' * (self.max_len - len(dna_sequence))
        else:
            x = dna_sequence

        for i in range(0, len(x) - self.k + 1, self.stride):
            k_mer = x[i:i + self.k]
            tokens.append(k_mer)
        return self.vocabulary_mapper(tokens)


class DNADataset(Dataset):
    def __init__(self, file_path, k_mer=4, stride=4, max_len=256):
        self.k_mer = k_mer
        self.stride = stride
        self.max_len = max_len

        # Vocabulary
        kmer_iter = ([''.join(kmer)] for kmer in product('ACGT', repeat=self.k_mer))
        self.vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
        self.vocab.set_default_index(self.vocab["<UNK>"])
        self.vocab_size = len(self.vocab)

        self.tokenizer = KmerTokenizer(self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len)

        train_csv = pd.read_csv(file_path, sep='\t')
        self.barcodes = train_csv['nucleotides'].to_list()

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        processed_barcode = torch.tensor(self.tokenizer(self.barcodes[idx]), dtype=torch.int64)
        return processed_barcode


"""# Train """
def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  #
        new_state_dict[key] = value
    return new_state_dict
    


def train(args, dataloader, device, model, optimizer, scheduler):
    model.train()
    epoch_loss_list = []
    training_epoch = 100
    continue_epoch = 0

    saving_path = "../../model_checkpoints/"
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    

    sys.stdout.write("Training is started:\n")

    for epoch in range(continue_epoch + 1, training_epoch + 1):
        epoch_loss = 0

        dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            # Build the masking on the fly every time something different
            batch = batch.to(device)
            masked_input = batch.clone()
            random_mask = torch.rand(masked_input.shape).to(device)  # I can only do this for non-overlapping
            random_mask = (random_mask < 0.5) * (masked_input != 2)  # Cannot mask the [<UNK>] token
            mask_idx = (random_mask.flatten() == True).nonzero().view(-1)
            masked_input = masked_input.flatten()
            masked_input[mask_idx] = 1
            masked_input = masked_input.view(batch.size())

            out = model(masked_input, labels=batch)
            loss = out.loss
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
        torch.save(model.state_dict(), saving_path + "4_model_" + str(epoch) + '.pth')
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
    configuration = BertConfig(vocab_size=dataset.vocab_size, output_hidden_states=True)
    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    model = BertForMaskedLM(configuration).to(rank)
    #print(model.state_dict().keys())
    
    if args['checkpoint']:
        
        state_dict = torch.load(f'../../model_checkpoints/4_model_{args["k_mer"]}.pth', map_location='cuda:0')
        adjusted_state_dict = {k: v for k, v in state_dict.items() if 'pooler' not in k}
        single_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '')
            single_state_dict[new_key] = state_dict[key]
        model.load_state_dict(single_state_dict, strict=False)

        #a_file = open(saving_path + f"loss_{device}.pkl", "rb")
        #epoch_loss_list = pickle.load(a_file)   
    
    sys.stdout.write("The model has been successfully initialized ...\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, total_iters=5)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    train(args, dataloader, rank, model, optimizer, scheduler)
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)
    parser.add_argument('--checkpoint', action='store', type=bool, default=False)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--stride', action='store', type=int, default=4)
    parser.add_argument('--max_len', action='store', type=int, default=660)
    parser.add_argument('--batch_size', action='store', type=int, default=16)
    parser.add_argument('--lr', action='store', type=float, default=1e-4)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)

    args = vars(parser.parse_args())

    sys.stdout.write("\nTraining Parameters:\n")
    for key in args:
        sys.stdout.write(f'{key} \t -> {args[key]}\n')

    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
