# -*- coding: utf-8 -*-
import pandas as pd
import os
import scipy.io as sio
import numpy as np
import random
from random import randrange, shuffle, random, randint
import re
import sys
import argparse
from transformers import BertForPreTraining
from transformers import BertConfig

import math
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

# import tfrecord
# from tfrecord.torch.dataset import TFRecordDataset


"""# Load data and tokenize"""


class PabloDNADataset:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep="\t", encoding="unicode_escape")
        # self.df = pd.read_csv(file_path, sep=",", encoding="unicode_escape")

    def clean_nan(self, col_names, replace_orig=False):
        clean_df = self.df.dropna(subset=col_names)
        clean_df = clean_df.reset_index(drop=True)
        if replace_orig:
            self.df = clean_df
        return clean_df

    def change_RXY2N(self, col_names, replace_orig=False):
        full_pattern = re.compile('[^ACGTN]')
        self.df[col_names] = self.df[col_names].apply(lambda x: re.sub(full_pattern, 'N', str(x)))
        # if replace_orig:
        #   self.df[col_names] = clean_nucleotides
        # return clean_str_df

    def generate_mini_sample(self, dataframe=None, bin_count=20, output_path="mini_sample.tsv"):
        if dataframe is None:
            dataframe = self.df
        bins = list(dataframe['bin_uri'].unique())
        rd1 = random.sample(range(0, len(bins)), bin_count)
        bins = [bins[i] for i in rd1]
        mini_df = dataframe.loc[dataframe['bin_uri'].isin(bins)]
        mini_df = mini_df.reset_index(drop=True)
        # mini_df = dataframe.iloc[0:sample_count]
        # mini_df = dataframe.take(np.random.permutation(len(dataframe))[:sample_count])
        mini_df.to_csv(output_path, sep="\t")

    def get_info(self, dataframe=None):
        if dataframe is None:
            dataframe = self.df
        print("Total data: ", len(dataframe))
        print("Number of bin clusters: ", len(dataframe['bin_uri'].unique()))


def tokenizer(dna_sentence, k_mer_dict, k_mer_length, stride=1):
    tokens = []
    for i in range(0, len(dna_sentence) - k_mer_length + 1, stride):
        k_mer = dna_sentence[i:i + k_mer_length]
        if 'N' in k_mer:
            tokens.append(k_mer_dict['[UNK]'])
        else:
            tokens.append(k_mer_dict[k_mer])
    return tokens


class SampleDNAData(Dataset):
    """Barcode Dataset"""

    @staticmethod
    def get_all_kmers(k_mer_length, alphabet=None) -> list:
        """
        :rtype: object
        """

        def base_convert(num, base, length):
            result = []
            while num > 0:
                result.insert(0, num % base)
                num = num // base
            while len(result) < length:
                result.insert(0, 0)
            return result

        if alphabet is None:
            alphabet = ["A", "T", "C", "G"]
        k_mer_counts = len(alphabet) ** k_mer_length
        all_k_mers_list = []
        for i in range(k_mer_counts):
            code = base_convert(num=i, base=len(alphabet), length=k_mer_length)
            k_mer = ""
            for j in range(k_mer_length):
                k_mer += alphabet[code[j]]
            all_k_mers_list.append(k_mer)

        return all_k_mers_list

    def _generate_processed_data(self, dna_a, dna_b, label):
        rand_len = randrange(128, 256)

        dna_a = dna_a[0:len(dna_a) // 2][0:rand_len]
        dna_b = dna_b[len(dna_b) // 2:][0:rand_len]
        tokens_a = tokenizer(dna_a, self.word_dict, self.k_mer, stride=1)
        tokens_b = tokenizer(dna_b, self.word_dict, self.k_mer, stride=1)

        input_ids = [self.word_dict['[CLS]']] + tokens_a + [self.word_dict['[SEP]']] + tokens_b + [
            self.word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM  (15 % of tokens in one sentence)
        n_pred = min(self.max_mask_count, max(1, int(round(len(input_ids) * 0.15)))) // self.k_mer

        cand_masked_pos = [i for i, token in enumerate(input_ids)
                           if token != self.word_dict['[CLS]'] and token != self.word_dict['[SEP]']]

        # remove N and gaps from cand_masked_pos
        cand_masked_pos_copy = cand_masked_pos.copy()
        for position in cand_masked_pos_copy:
            remove_flag = False
            for s in range(self.k_mer):
                if position + s < len(input_ids):
                    key = self.number_dict[input_ids[position + s]]
                    if ("N" in key) or ("-" in key):
                        remove_flag = True
                        break
            if remove_flag:
                cand_masked_pos.remove(position)

        shuffle(cand_masked_pos)

        # if the position remains is less than 15%, mask them all
        if len(cand_masked_pos) < n_pred:
            n_pred = len(cand_masked_pos)

        masked_tokens, masked_pos = [], []
        attention_mask = [1] * len(input_ids)

        for pos in cand_masked_pos[:n_pred]:
            for s in range(self.k_mer):
                if pos + s < len(input_ids):
                    masked_pos.append(pos + s)
                    masked_tokens.append(input_ids[pos + s])
                    attention_mask[pos + s] = 0
                    input_ids[pos + s] = self.word_dict['[MASK]']  # make mask

        # Zero Paddings
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        attention_mask.extend([1] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if self.max_mask_count > len(masked_pos):
            n_pad = self.max_mask_count - len(masked_pos)
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        return [input_ids, segment_ids, attention_mask, masked_tokens, masked_pos, label]

    def __init__(self, file_path, k_mer=4, max_mask_count=5, max_len=256):
        self.k_mer = k_mer
        self.max_mask_count = max_mask_count
        self.max_len = max_len
        pablo_dataset = PabloDNADataset(file_path)
        # for removing X,R,Y letters from data
        pablo_dataset.change_RXY2N("nucleotides")
        self.dna_nucleotides = list(pablo_dataset.df["nucleotides"].values)
        self.data_count = len(self.dna_nucleotides)
        word_list = SampleDNAData.get_all_kmers(self.k_mer)

        number_dict = dict()
        word_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
        for i, w in enumerate(word_list):
            word_dict[w] = i + 4
            number_dict = {i: w for i, w in enumerate(word_dict)}

        self.word_dict = word_dict
        self.number_dict = number_dict

        self.vocab_size = len(word_dict)
        self.max_len = max_len

        self.batch = []

        for data_index in tqdm(range(self.data_count)):
            dna_a = self.dna_nucleotides[data_index]

            tokens_b_index = data_index
            while data_index == tokens_b_index:
                tokens_b_index = randrange(len(self.dna_nucleotides))
            dna_b = self.dna_nucleotides[tokens_b_index]

            self.batch.append(self._generate_processed_data(dna_a=dna_a, dna_b=dna_a, label=True))
            self.batch.append(self._generate_processed_data(dna_a=dna_a, dna_b=dna_b, label=False))

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        ids = torch.Tensor(self.batch[idx][0])
        seg = torch.Tensor(self.batch[idx][1])
        mask = torch.Tensor(self.batch[idx][2])
        msk_tok = torch.Tensor(self.batch[idx][3])
        msk_pos = torch.Tensor(self.batch[idx][4])
        label = torch.Tensor([self.batch[idx][5]])

        ids, seg, msk_pos = ids.type(torch.LongTensor), seg.type(torch.LongTensor), msk_pos.type(torch.int64)
        mask = mask.type(torch.FloatTensor)
        msk_tok = msk_tok.type(torch.LongTensor)
        label = label.type(torch.LongTensor)

        return ids, seg, mask, msk_pos, msk_tok, label


"""# Train """


def train(args, dataloader, device, model, optimizer, scheduler):
    # start training
    criterion = nn.CrossEntropyLoss()

    epoch_loss_list = []
    training_epoch = 1000
    continue_epoch = 0

    saving_path = "model_checkpoints/"
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    if args['checkpoint']:
        continue_epoch = 0
        model.load_state_dict(torch.load(saving_path + f'model_{continue_epoch}.pth'))
        optimizer.load_state_dict(torch.load(saving_path + f"optimizer_{continue_epoch}.pth"))

        a_file = open(saving_path + "loss.pkl", "rb")
        epoch_loss_list = pickle.load(a_file)
        print("Trainig is countinue...")

    sys.stdout.write("Training is started:\n")

    for epoch in range(continue_epoch + 1, training_epoch + 1):
        epoch_loss = 0
        dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(tqdm(dataloader)):
            ids = batch[0].to(device)  # 'tokens'
            seg = batch[1].to(device)  # 'segment'
            mask = batch[2].to(device)
            msk_pos = batch[3].to(device)  # 'msk_pos'
            masked_tokens = batch[4].to(device)  # 'msk_tok'
            is_pos = batch[5].to(device)  # 'label'

            inputs = {
                'input_ids': ids,
                'token_type_ids': seg,
                'attention_mask': mask
            }

            optimizer.zero_grad()
            outputs = model(**inputs)

            msk_pos = msk_pos[:, :, None].expand(-1, -1, outputs['prediction_logits'].size(-1))
            masked_prediction_logits = torch.gather(outputs['prediction_logits'], 1, msk_pos)

            seq_relationship_loss = criterion(outputs['seq_relationship_logits'], torch.squeeze(is_pos))

            prediction_loss = criterion(masked_prediction_logits.transpose(1, 2), masked_tokens)

            w = args['loss_weight']  ## Weight for the importance of the masked language model
            loss = w * prediction_loss + (1 - w) * seq_relationship_loss

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss_list.append(epoch_loss)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

        # every epoch save the checkpoints and save the loss in a list
        if epoch % 1 == 0:
            sys.stdout.write(f"Epoch {epoch}, Device {device}: Loss is {epoch_loss}\n")
            torch.save(model.state_dict(), saving_path + "model_" + str(epoch) + '.pth')
            torch.save(optimizer.state_dict(), saving_path + "optimizer_" + str(epoch) + '.pth')
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
    # dataset = Your_Dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)

    sys.stdout.write("Loading the dataset is started.\n")
    dataset = SampleDNAData(file_path=args['input_path'], k_mer=args['k_mer'], max_mask_count=args['max_mask_count'],
                            max_len=args['max_len'])

    sys.stdout.write("loading the model.\n")
    # vocab_size = 4 ** args['k_mer'] + 5  # '[PAD]', '[UNK]' ,'[CLS]', '[SEP]',  '[MASK]'

    '''Default Mode'''
    # configuration = BertConfig(vocab_size=vocab_size)
    # # Initializing a model (with random weights) from the bert-base-uncased style configuration
    # model = BertForPreTraining(configuration).to(rank)

    '''DNABERT ARCHITECTURE'''
    model = BertForPreTraining.from_pretrained('zhihan1996/DNA_bert_4')  # dnabert-minilm-small'
    model = BertForPreTraining(model.config).to(rank)

    '''DNABERT ARCHITECTURE PRETRAINED'''
    # model = BertForPreTraining.from_pretrained('zhihan1996/DNA_bert_4').to(rank)  # dnabert-minilm-small'

    sys.stdout.write("Model is loaded.\n")
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(args['betas_a'], args['betas_b']), eps=args['eps'],
                           weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=20)

    dataloader = prepare(dataset, rank, world_size=world_size, batch_size=args['batch_size'])

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    train(args, dataloader, rank, model, optimizer, scheduler)
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)
    parser.add_argument('--checkpoint', action='store', type=bool, default=False)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--max_mask_count', action='store', type=int, default=80)
    parser.add_argument('--max_len', action='store', type=int, default=512)
    parser.add_argument('--batch_size', action='store', type=int, default=8)
    parser.add_argument('--loss_weight', action='store', type=float, default=0.5)
    parser.add_argument('--lr', action='store', type=float, default=0.0005)
    parser.add_argument('--betas_a', action='store', type=float, default=0.9)
    parser.add_argument('--betas_b', action='store', type=float, default=0.98)
    parser.add_argument('--eps', action='store', type=float, default=1e-06)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)

    args = vars(parser.parse_args())

    sys.stdout.write("\nTraining Parameters:\n")
    for key in args:
        sys.stdout.write(f'{key} \t -> {args[key]}\n')

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
