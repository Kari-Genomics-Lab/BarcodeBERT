import torch
from torch.utils.data import TensorDataset, DataLoader


def construct_dataloader(X_train, X_val, y_train, y_val, batch_size):
    X_train = torch.Tensor(X_train)
    X_val = torch.Tensor(X_val)
    y_train = torch.Tensor(y_train)
    y_val = torch.Tensor(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader


class KmerTokenizer(object):
    def __init__(self, k, vocabulary_mapper, stride=1, padding=False, max_len=66):
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
