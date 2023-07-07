import time
from itertools import product

import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import random
import math

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
    
    
k = 4
max_len = 660
tokenizer = kmer_tokenizer(k, stride=k)  #Non overlapping k-mers
PAD = pad_sequence(max_len)

kmer_iter = ([''.join(kmer)] for kmer in  product('ACGT',repeat=k))
vocab = build_vocab_from_iterator(kmer_iter, specials=["<CLS>","<MASK>","<UNK>"])
vocab.set_default_index(vocab["<UNK>"])


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


class BERT(nn.Module):
    def __init__(self, vocab_size, n_output, d_model, maxlen, n_layers, d_k, d_v, n_heads, device):
        super(BERT, self).__init__()
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

        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # classifier pos/neg (it will be decided by first token(CLS))
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        return logits_clsf

    def get_embeddings(self, input_ids):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            # embedding layer
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        return output

"""# Training """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv("data/supervised.tsv", sep='\t')
target_level='species_name'

barcodes =  train['nucleotides'].to_list()
targets  =  train[target_level].to_list()

# Splitting the training set into train and validation
barcodes_train, barcodes_test, targets_train, targets_test = train_test_split(barcodes, targets, test_size=0.3, random_state=42)
print(len(barcodes_train), len(targets_train))


label_set=sorted(list(set(targets)))
num_class= len(label_set)

label_pipeline = lambda x: label_set.index(x)
sequence_pipeline = lambda x: vocab(tokenizer(PAD(x)))


def collate_batch(batch):
    features, labels = [], []
    
    for _label, _barcode in batch:
        labels.append(label_pipeline(_label))
        processed_barcode = torch.tensor([0]+sequence_pipeline(_barcode), dtype=torch.int64) #CLS token always at the beginning
        features.append(processed_barcode)

    labels = torch.tensor(labels, dtype=torch.int64)
    features = torch.stack(features)

    return  features.to(device), labels.to(device)

vocab_size = len(vocab)

config = {
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "max_len": 512

}
    
print("Initializing the model . . .")
model = BERT(vocab_size, num_class, config["d_model"], max_len, config["n_layers"], 32, 32,
             config["n_heads"], device=device)
print("The model has been succesfully initialized . . .")


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Convert the data to PyTorch tensors and create DataLoader if needed
#train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
dataloader = DataLoader(list(zip(targets_train,barcodes_train)), batch_size=128, shuffle=True, collate_fn=collate_batch)

#test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(list(zip(targets_test,barcodes_test)), batch_size=128, shuffle=True, collate_fn=collate_batch)

log_interval=200
# Training
for epoch in range(20):
    epoch_start_time = time.time()
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(inputs)
        loss = criterion(predicted_label, labels)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (predicted_label.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()
    
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            predicted_label = model(inputs)
            loss = criterion(predicted_label, labels)
            total_acc += (predicted_label.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    accu_val = total_acc / total_count
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)
    
torch.save(model.state_dict(), 'model_checkpoints/model_full.pth')