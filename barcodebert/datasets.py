"""
Datasets.
"""

from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoTokenizer


class KmerTokenizer(object):
    def __init__(self, k, vocabulary_mapper, stride=1, padding=False, max_len=660):
        self.k = k
        self.stride = stride
        self.padding = padding
        self.max_len = max_len
        self.vocabulary_mapper = vocabulary_mapper

    def __call__(self, dna_sequence, offset=0) -> (list, list):
        tokens = []
        att_mask = [1] * (self.max_len // self.stride)
        x = dna_sequence[offset:]
        if self.padding:
            if len(x) > self.max_len:
                x = x[: self.max_len]
            else:
                x = x + "N" * (self.max_len - len(x))
                att_mask[len(x) // self.stride :] = [0] * (len(att_mask) - len(x) // self.stride)

        for i in range(0, len(x) - self.k + 1, self.stride):
            k_mer = x[i : i + self.k]
            tokens.append(k_mer)

        tokens = torch.tensor(self.vocabulary_mapper(tokens), dtype=torch.int64)
        att_mask = torch.tensor(att_mask, dtype=torch.int32)

        return tokens, att_mask


class DnaBertBPETokenizer(object):
    def __init__(self, padding=False, max_tokenized_len=128):
        self.padding = padding
        self.max_tokenized_len = max_tokenized_len
        self.bpe = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    def __call__(self, dna_sequence, offset=0) -> (list, list):
        x = dna_sequence[offset:]
        tokens = self.bpe(x, padding=True, return_tensors="pt")["input_ids"]
        tokens[tokens == 0] = 1

        tokens = tokens[0].tolist()

        if len(tokens) > self.max_tokenized_len:
            att_mask = [1] * self.max_tokenized_len
            tokens = tokens[: self.max_tokenized_len]
        else:
            att_mask = [1] * (len(tokens)) + [0] * (self.max_tokenized_len - len(tokens))
            tokens = tokens + [1] * (self.max_tokenized_len - len(tokens))

        att_mask = torch.tensor(att_mask, dtype=torch.int32)
        return tokens, att_mask


class DNADataset(Dataset):
    def __init__(
        self, file_path, k_mer=4, stride=None, max_len=256, randomize_offset=False, use_unk_token=True, tokenizer="kmer"
    ):
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset

        # Vocabulary
        letters = "ACGT"
        specials = ["<MASK>"]
        if use_unk_token:
            # Encode all kmers which contain at least one N as <UNK>
            UNK_TOKEN = "<UNK>"
            specials.append(UNK_TOKEN)
        else:
            # Encode kmers which contain N differently depending on where it is
            letters += "N"

        if tokenizer == "kmer":
            kmer_iter = (["".join(kmer)] for kmer in product(letters, repeat=self.k_mer))
            self.vocab = build_vocab_from_iterator(kmer_iter, specials=specials)
            if use_unk_token:
                self.vocab.set_default_index(self.vocab.lookup_indices([UNK_TOKEN])[0])

            self.vocab_size = len(self.vocab)
            self.tokenizer = KmerTokenizer(
                self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len
            )
        elif tokenizer == "DnaBertBPE":
            self.tokenizer = DnaBertBPETokenizer(padding=True, max_tokenized_len=self.max_len)
            self.vocab_size = self.tokenizer.bpe.vocab_size
        else:
            raise ValueError(f'Tokenizer "{tokenizer}" not recognized.')
        df = pd.read_csv(file_path, sep="\t" if file_path.endswith(".tsv") else ",", keep_default_na=False)
        self.barcodes = df["nucleotides"].to_list()
        self.label_names = df["species_name"].to_list()
        self.labels = df["species_index"].to_list()

        self.num_labels = 22_622

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0
        processed_barcode, att_mask = self.tokenizer(self.barcodes[idx], offset=offset)
        label = torch.tensor((self.label_pipeline(self.labels[idx])), dtype=torch.int64)
        return processed_barcode, label, att_mask


def single_inference(model, tokenizer, barcode):
    with torch.no_grad():
        x, att_mask = tokenizer(barcode)

        x = x.unsqueeze(0).to(model.device)
        att_mask = att_mask.unsqueeze(0).to(model.device)
        x = model(x, att_mask).hidden_states[-1]
        x = x.mean(1)
    return x


def representations_from_df(df, target_level, model, tokenizer):

    orders = df["order_name"].to_numpy()

    _label_set, y = np.unique(df[target_level], return_inverse=True)

    dna_embeddings = []
    for barcode in df["nucleotides"]:
        x = single_inference(model, tokenizer, barcode)
        dna_embeddings.append(x.cpu().numpy())

    print(f"There are {len(df)} points in the dataset")
    latent = np.array(dna_embeddings)
    latent = np.squeeze(latent, 1)
    print(latent.shape)
    return latent, y, orders


def inference_from_df(df, model, tokenizer):

    assert "processid" in df.columns  # Check that processid column is present in your dataframe
    assert "nucleotides" in df.columns  # Check that nucleotide column is present in your dataframe

    dna_embeddings = {}

    for _i, row in df.iterrows():
        barcode = row["nucleotides"]
        id = row["processid"]

        x = single_inference(model, tokenizer, barcode)

        dna_embeddings[id] = x.cpu().numpy()

    return dna_embeddings


def check_sequence(header, seq):
    """
    Adapted from VAMB: https://github.com/RasmussenLab/vamb

    Check that there're no invalid characters or bad format
    in the file.

    Note: The GAPS ('-') that are introduced from alignment
    are considered valid characters.
    """

    if len(header) > 0 and (header[0] in (">", "#") or header[0].isspace()):
        raise ValueError("Bad character in sequence header")
    if "\t" in header:
        raise ValueError("tab included in header")

    basemask = bytearray.maketrans(b"acgtuUswkmyrbdhvnSWKMYRBDHV-", b"ACGTTTNNNNNNNNNNNNNNNNNNNNNN")

    masked = seq.translate(basemask, b" \t\n\r")
    stripped = masked.translate(None, b"ACGTN")
    if len(stripped) > 0:
        bad_character = chr(stripped[0])
        msg = "Invalid DNA byte in sequence {}: '{}'"
        raise ValueError(msg.format(header, bad_character))
    return masked


def inference_from_fasta(fname, model, tokenizer):

    dna_embeddings = {}
    lines = []
    seq_id = ""

    for line in open(fname, "rb"):
        if line.startswith(b"#"):
            pass

        elif line.startswith(b">"):
            if seq_id != "":
                seq = bytearray().join(lines)

                # Check entry is valid
                seq = check_sequence(seq_id, seq)

                # Compute embedding
                x = single_inference(model, tokenizer, seq.decode())

                lines = []
                seq_id = line[1:-1].decode()  # Modify this according to your labels.
                dna_embeddings[seq_id] = x.cpu().numpy()
            seq_id = line[1:-1].decode()
        else:
            lines += [line.strip()]

    seq = bytearray().join(lines)
    seq = check_sequence(seq_id, seq)
    # Compute embedding
    x = single_inference(model, tokenizer, seq.decode())
    dna_embeddings[seq_id] = x.cpu().numpy()

    return dna_embeddings
