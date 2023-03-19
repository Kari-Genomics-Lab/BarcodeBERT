import random
import re
from random import randrange, shuffle, random

import pandas as pd
import torch
from torch.utils.data import Dataset


class PabloDNADataset:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep="\t", encoding="unicode_escape")

    def clean_nan(self, col_names, replace_orig=False):
        clean_df = self.df.dropna(subset=col_names)
        clean_df = clean_df.reset_index(drop=True)
        if replace_orig:
            self.df = clean_df
        return clean_df

    def change_RXY2N(self, col_names, replace_orig=False):
        full_pattern = re.compile('[^ACGTN\-]')
        self.df[col_names] = self.df[col_names].apply(lambda x: re.sub(full_pattern, 'N', x))
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
            alphabet = ["A", "C", "G", "T", "-", "N"]
        k_mer_counts = len(alphabet) ** k_mer_length
        all_k_mers_list = []
        for i in range(k_mer_counts):
            code = base_convert(num=i, base=len(alphabet), length=k_mer_length)
            k_mer = ""
            for j in range(k_mer_length):
                k_mer += alphabet[code[j]]
            all_k_mers_list.append(k_mer)

        return all_k_mers_list

    def __init__(self, file_path, k_mer=4, data_count=8, max_mask_count=5, max_len=256):
        pablo_dataset = PabloDNADataset(file_path)
        # for removing X,R,Y letters from data
        pablo_dataset.change_RXY2N("nucleotides")
        self.dna_nucleotides = list(pablo_dataset.df["nucleotides"].values)
        word_list = SampleDNAData.get_all_kmers(k_mer)

        word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i, w in enumerate(word_list):
            word_dict[w] = i + 4
            number_dict = {i: w for i, w in enumerate(word_dict)}  # TODO: try move this out from the loop.

        self.word_dict = word_dict
        self.number_dict = number_dict

        self.vocab_size = len(word_dict)
        self.max_len = max_len

        self.batch = []
        positive = negative = 0
        while positive != data_count / 2 or negative != data_count / 2:
            is_positive = randrange(0, 2)

            tokens_a_index, tokens_b_index = 0, 0
            while tokens_a_index == tokens_b_index:
                tokens_a_index, tokens_b_index = randrange(len(self.dna_nucleotides)), randrange(
                    len(self.dna_nucleotides))

            if is_positive:
                dna_a = self.dna_nucleotides[tokens_a_index]
                dna_b = dna_a
            else:
                dna_a = self.dna_nucleotides[tokens_a_index]
                dna_b = self.dna_nucleotides[tokens_b_index]

            rand_len = randrange(128, 256)

            dna_a = dna_a[0:len(dna_a) // 2][0:rand_len]  # max_len//2 - 3]
            dna_b = dna_b[len(dna_b) // 2:][0:rand_len]  # max_len//2 - 3]
            tokens_a = tokenizer(dna_a, word_dict, k_mer, stride=1)
            tokens_b = tokenizer(dna_b, word_dict, k_mer, stride=1)
            input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
            segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

            # MASK LM
            n_pred = min(max_mask_count, max(1, int(round(len(input_ids) * 0.15))))  # 15 % of tokens in one sentence
            cand_masked_pos = [i for i, token in enumerate(input_ids)
                               if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
            shuffle(cand_masked_pos)

            # remove N and gaps from cand_masked_pos
            cand_masked_pos_copy = cand_masked_pos.copy()
            for position in cand_masked_pos_copy:
                key = list(word_dict.keys())[list(word_dict.values()).index(position)]
                if ("N" in key) or ("-" in key):
                    cand_masked_pos.remove(position)
            # if the position remains is less than 15%, mask them all
            if len(cand_masked_pos) < n_pred:
                n_pred = len(cand_masked_pos)

            masked_tokens, masked_pos = [], []
            for pos in cand_masked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                # if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # make mask
                # elif random() < 0.5:  # 10%
                #     index = randint(0, vocab_size - 1)  # random index in vocabulary
                #     input_ids[pos] = word_dict[number_dict[index]]  # replace

            # Zero Paddings
            n_pad = max_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad)

            # Zero Padding (100% - 15%) tokens
            if max_mask_count > n_pred:
                n_pad = max_mask_count - n_pred
                masked_tokens.extend([0] * n_pad)
                masked_pos.extend([0] * n_pad)

            if is_positive and positive < data_count / 2:
                self.batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
                positive += 1
            elif not is_positive and negative < data_count / 2:
                self.batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
                negative += 1

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        ids = torch.Tensor(self.batch[idx][0])
        seg = torch.Tensor(self.batch[idx][1])
        msk_tok = torch.Tensor(self.batch[idx][2])
        msk_pos = torch.Tensor(self.batch[idx][3])
        label = torch.Tensor([self.batch[idx][4]])

        ids, seg, msk_pos = ids.type(torch.IntTensor), seg.type(torch.IntTensor), msk_pos.type(torch.int64)

        msk_tok = msk_tok.type(torch.LongTensor)
        label = label.type(torch.LongTensor)

        return ids, seg, msk_pos, msk_tok, label

class test_DNA_Data(Dataset):
    """Barcode Dataset"""

    def __init__(self, file_path, k_mer=4, max_len=512):
        pablo_dataset = PabloDNADataset(file_path)
        # for removing X,R,Y letters from data
        pablo_dataset.change_RXY2N("nucleotides")
        self.dna_nucleotides = list(pablo_dataset.df["nucleotides"].values)
        self.species = list(pablo_dataset.df["species_name"].values)
        word_list = SampleDNAData.get_all_kmers(k_mer)

        word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i, w in enumerate(word_list):
            word_dict[w] = i + 4
            number_dict = {i: w for i, w in enumerate(word_dict)}  # TODO: try move this out from the loop.


        self.word_dict = word_dict
        self.number_dict = number_dict
        self.vocab_size = len(word_dict)
        self.max_len = max_len

        self.IDS = []
        self.SEGMENTS = []
        self.SPECIES = []

        for seq, species in zip(self.dna_nucleotides, self.species):
            if len(seq) > (self.max_len - 2):
                seq = seq[:self.max_len - 2]
            tokens = tokenizer(seq, word_dict, 4, stride=1)
            input_ids = [word_dict['[CLS]']] + tokens + [word_dict['[SEP]']]
            segment_ids = [0] * (1 + len(tokens) ) + [1] * (1)
            masked_tokens, masked_pos = [], [] # No mask for testing

            # Zero Paddings
            n_pad = max_len - len(input_ids)
            if n_pad > 0:
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

            self.IDS.append(input_ids)
            self.SEGMENTS.append(segment_ids)
            self.SPECIES.append(species)

    def __len__(self):
        return len(self.dna_nucleotides)

    def __getitem__(self, idx):
        ids = torch.Tensor(self.IDS[idx])
        seg = torch.Tensor(self.SEGMENTS[idx])
        # msk_pos = torch.Tensor([])
        label = self.SPECIES[idx]

        ids, seg = ids.type(torch.IntTensor), seg.type(torch.IntTensor) # [msk_pos.type(torch.int64)]

        return  {'input':[ids, seg], 'label':label}
