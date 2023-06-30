import random
import re
from random import randrange, shuffle, random
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset

class PabloDNADataset:
    def __init__(self, file_path):
        # self.df = pd.read_csv(file_path, sep="\t", encoding="unicode_escape")
        self.df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

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
            alphabet = ["A", "C", "G", "T", "N"]
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
        word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
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