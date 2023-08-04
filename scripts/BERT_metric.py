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


import umap
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




"""# Model"""
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(int(vocab_size), int(d_model))  # token embedding
        self.pos_embed = nn.Embedding(int(maxlen), int(d_model))  # position embedding
        self.norm = nn.LayerNorm(int(d_model))
        self.device = device

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    pad_attn_mask = pad_attn_mask.to(device)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_k)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_model = d_model

        self.linear = nn.Linear(self.n_heads * self.d_v, self.d_model)

        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_v)  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)

        return self.layernorm(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class BERT_METRIC(nn.Module):
    def __init__(self, vocab_size, n_output, d_model, maxlen, n_layers, d_k, d_v, n_heads, device):
        super(BERT_METRIC, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, maxlen, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = GELU()
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_output)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device
        self.to(self.device)

    def forward(self, input_ids):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            # embedding layer
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        return output

    def get_embeddings(self, input_ids):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            # embedding layer
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        return output

    
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


label_set=sorted(list(set(targets)))

label_pipeline = lambda x: label_set.index(x)
sequence_pipeline = lambda x: vocab(tokenizer(PAD(x)))

vocab_size = len(vocab)

config = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "max_len": 512

}

num_class = 1390
print("Initializing the model . . .")
model = BERT_METRIC(vocab_size,  num_class, config["d_model"], max_len, config["n_layers"], 32, 32,
             config["n_heads"], device=device)
model.load_state_dict(torch.load('model_checkpoints/model_full.pth'))
print("The model has been succesfully initialized . . .")
model.eval()

dna_embeddings = []
labels=[]

with torch.no_grad():
    for i, _barcode in enumerate(barcodes):
        x = torch.tensor([0]+sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
        x = model(x)
        #x = x[:,1:,:].mean(1)   #Global Average Pooling excluding CLS token
        x = model.activ1(model.fc(x[:, 0])) #Representation of the CLS token
        dna_embeddings.extend(x.cpu().numpy())
        labels.append(label_pipeline(targets[i]))
    
print(f"There are {len(dna_embeddings)} points in the dataset")
latent = np.array(dna_embeddings).reshape(-1,config["d_model"])
y_unseen = np.array(labels)
print(latent.shape)
embedding = umap.UMAP(random_state=42).fit_transform(latent)

import matplotlib.pyplot as plt 
fig, ax = plt.subplots(nrows=1, ncols=1) 
ax.set_title("Representation of the Latent Space")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")

ax.scatter(embedding[:, 0],
           embedding[:, 1],
           c=y_unseen,
           s=1,
           cmap='Spectral')
plt.savefig('BERT_CLS_embeddings.png',dpi=150)

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

