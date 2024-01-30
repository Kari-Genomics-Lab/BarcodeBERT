# -*- coding: utf-8 -*-
import pandas as pd
import os
from itertools import product
import sys
import argparse
from transformers import BertConfig, BertForTokenClassification

import torch
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import pickle
from tqdm import tqdm
import time
from resource import *

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.metrics import accuracy_score


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

        train_csv = pd.read_csv(file_path, sep=',')
        self.barcodes = train_csv['nucleotides'].to_list()
        self.labels = train_csv['species_name'].to_list()
   

        self.label_set=sorted(list(set(self.labels)))
        #print(label_set)

        self.label_pipeline = lambda x: self.label_set.index(x)
        self.num_labels = len(train_csv['species_name'].unique())

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        processed_barcode, att_mask  = self.tokenizer(self.barcodes[idx])        
        label = torch.tensor((self.label_pipeline(self.labels[idx])), dtype=torch.int64)
        return processed_barcode, label, att_mask 




class Classification_model(nn.Module):
    def __init__(self, args, checkpoint, num_labels, vocab_size):
        super(Classification_model, self).__init__()
        self.num_labels = num_labels
        
        configuration = BertConfig(vocab_size=int(vocab_size), 
                               num_hidden_layers=args['n_layers'],
                               num_attention_heads=args['n_heads'],    
                               num_labels = 4**args['k_mer'],
                               output_hidden_states=True)
        
        # Load Model with given checkpoint
        
        self.model = BertForTokenClassification(configuration)
        state_dict = remove_extra_pre_fix(torch.load(checkpoint, map_location="cuda:0"))
        self.model.load_state_dict(state_dict, strict=True)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids=None, mask=None ,labels=None):
        # Getting the embedding
        outputs = self.model(input_ids=input_ids, attention_mask=mask)
        embeddings = outputs.hidden_states[-1]
        GAP_embeddings = embeddings.mean(1)
        # calculate losses
        logits = self.classifier(GAP_embeddings.view(-1, 768))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


"""# Train """
def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # 去除 'module.' 前缀
        new_state_dict[key] = value
    return new_state_dict

def train(args, dataloader, device, model, optimizer, scheduler):
    epoch_loss_list = []
    eval_epoch_loss_list = []
    eval_acc_list = []
    training_epoch = args["epoch"]
    continue_epoch = 0
    dataloader_train, dataloader_dev = dataloader[0], dataloader[1]

    saving_path = args["input_path"] + "checkpoints/"
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    if args['checkpoint']:
        continue_epoch = 4
        model.load_state_dict(torch.load(saving_path + f'model_{continue_epoch}.pth'))
        optimizer.load_state_dict(torch.load(saving_path + f"optimizer_{continue_epoch}.pth"))
        scheduler.load_state_dict(torch.load(saving_path + f"scheduler_{continue_epoch}.pth"))
        a_file = open(saving_path + f"loss_{device}.pkl", "rb")
        epoch_loss_list = pickle.load(a_file)
        print("Training is continued...")

    sys.stdout.write("Training is started:\n")

    for epoch in range(continue_epoch + 1, training_epoch + 1):
        epoch_loss = 0
        eval_loss = 0
        acc = 0
        dataloader_train.sampler.set_epoch(epoch)
        model.train()
        for i, (sequences, labels, mask) in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()
            
            sequences = sequences.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            sequences = sequences.clone()
            labels = labels.clone()

            out = model(sequences, mask=mask, labels=labels)
            loss = out.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(dataloader_train)
        epoch_loss_list.append(epoch_loss)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        sys.stdout.write("Epoch %d: lr %f -> %f" % (epoch, before_lr, after_lr))

        sys.stdout.write(f"Epoch {epoch}, Device {device}: Loss is {epoch_loss}\n")

        torch.save(model.state_dict(), saving_path + "model_" + str(epoch) + '.pth')
        torch.save(optimizer.state_dict(), saving_path + "optimizer_" + str(epoch) + '.pth')
        torch.save(scheduler.state_dict(), saving_path + "scheduler_" + str(epoch) + '.pth')

        model.eval()
        for i, (eval_sequences, eval_labels, mask) in enumerate(tqdm(dataloader_dev)):
            
            eval_sequences = eval_sequences.to(device)
            eval_labels = eval_labels.to(device)
            mask = mask.to(device)
            
            with torch.no_grad():
                outputs = model(eval_sequences, mask=mask, labels=eval_labels)

            eval_loss += outputs.loss.item()
            eval_logits = outputs.logits
            predictions = torch.argmax(eval_logits, dim=-1)
            acc += accuracy_score(predictions.cpu(), eval_labels.cpu())

        eval_acc = acc / len(dataloader_dev)
        eval_acc_list.append(eval_acc)

        eval_loss = eval_loss / len(dataloader_dev)
        eval_epoch_loss_list.append(eval_loss)

        sys.stdout.write("validation set loss: %f \n" % eval_loss)
        sys.stdout.write("validation set accuracy: %f \n" % eval_acc)

        sys.stdout.write("--------------------------------------------")
        a_file = open(saving_path + f"loss_{device}.pkl", "wb")
        l = [epoch_loss_list, eval_epoch_loss_list, eval_acc_list]
        pickle.dump(l, a_file)

        a_file.close()

    return model


'''test'''


def test(dataloader_test, device, model):
    acc = 0

    model.eval()
    for i, (eval_sequences, eval_labels, mask) in enumerate(tqdm(dataloader_test)):

        eval_sequences = eval_sequences.to(device)
        eval_labels = eval_labels.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            outputs = model(eval_sequences)

        eval_logits = outputs.logits
        predictions = torch.argmax(eval_logits, dim=-1)
        acc += accuracy_score(predictions.cpu(), eval_labels.cpu())

    eval_acc = acc / len(dataloader_test)

    sys.stdout.write("test set accuracy: %f \n" % eval_acc)


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
    # Loading data
    sys.stdout.write("Loading the dataset is started.\n")
    dataset_train = DNADataset(file_path=args['input_path'] + "/train.csv", k_mer=args['k_mer'],
                               stride=args['stride'],
                               max_len=args['max_len'])
    dataloader_train = prepare(dataset_train, rank, world_size=world_size, batch_size=args['batch_size'])
    dataset_dev = DNADataset(file_path=args['input_path'] + "/dev.csv", k_mer=args['k_mer'], stride=args['stride'],
                             max_len=args['max_len'])
    dataloader_dev = prepare(dataset_dev, rank, world_size=world_size, batch_size=args['batch_size'])
    dataset_test = DNADataset(file_path=args['input_path'] + "/test.csv", k_mer=args['k_mer'],
                              stride=args['stride'],
                              max_len=args['max_len'])

    dataloader_test = prepare(dataset_test, rank, world_size=world_size, batch_size=args['batch_size'])

    # loading model
    checkpoint_path = args["Pretrained_checkpoint_path"]
    num_labels = dataset_train.num_labels
    vocab_size = dataset_train.vocab_size

    sys.stdout.write("Initializing the model ...\n")

    model = Classification_model(args, 
                                 checkpoint=checkpoint_path, 
                                 num_labels=num_labels, 
                                 vocab_size=vocab_size).to(rank)
    sys.stdout.write("The model has been successfully initialized ...\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=5)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    start_time = time.time()
    trained_model = train(args, [dataloader_train, dataloader_dev], rank, model, optimizer, scheduler)
    
    running_info = getrusage(RUSAGE_SELF)
    _time = (time.time() - start_time) #running_info.ru_utime + running_info.ru_stime
    hour = _time // 3600
    minutes = (_time  - (3600 * hour)) // 60
    seconds = _time - (hour * 3600) - (minutes * 60)
    memory = (running_info.ru_maxrss/1e6)
    sys.stdout.write(f'training took: {int(hour)}:{int(minutes)}:{round(seconds)} (hh:mm:ss)\n')
    
    start_time = time.time()
    test(dataloader_test, rank, trained_model)
    
    _time = (time.time() - start_time) #running_info.ru_utime + running_info.ru_stime
    hour = _time // 3600
    minutes = (_time  - (3600 * hour)) // 60
    seconds = _time - (hour * 3600) - (minutes * 60)
    memory = (running_info.ru_maxrss/1e6)
    print(f'Inference took: {int(hour)}:{int(minutes)}:{round(seconds)} (hh:mm:ss)\n')
    
    print(f'Max memory usage: {memory} (GB)')
    
    
    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', type=str)
    parser.add_argument('--Pretrained_checkpoint_path', action='store', type=str)
    parser.add_argument('--checkpoint', action='store', type=bool, default=False)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--stride', action='store', type=int, default=4)
    parser.add_argument('--max_len', action='store', type=int, default=720)
    parser.add_argument('--batch_size', action='store', type=int, default=64)
    parser.add_argument('--lr', action='store', type=float, default=1e-4)
    parser.add_argument('--epoch', action='store', type=int, default=35)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)
    parser.add_argument('--n_layers', action='store', type=int, default=12)
    parser.add_argument('--n_heads', action='store', type=int, default=12)

    args = vars(parser.parse_args())

    #sys.stdout.write("\nTraining Parameters:\n")
    #for key in args:
    #   sys.stdout.write(f'{key} \t -> {args[key]}\n')

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
