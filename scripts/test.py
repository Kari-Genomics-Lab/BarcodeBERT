# Load the libraries
import argparse
import sys
from os.path import dirname, abspath
from sklearn.neighbors import NearestNeighbors
import torch
import umap
from torch.utils.data import DataLoader

project_dir = dirname(dirname(abspath(__file__)))
sys.path.append(project_dir)
from util.model import BERT
from util.dataset import PabloDNADataset, tokenizer, test_DNA_Data
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def get_all_kmers(k_mer_length, alphabet=None) -> list:
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


def get_dataloader(args):
    test_dataset = test_DNA_Data(args.data_path, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return test_dataloader


def initial_model(args, device):
    word_list = get_all_kmers(args.k_mer)
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4

    model = BERT(len(word_dict), args.d_model, args.max_len, 2, args.n_layers, 32, 32, args.n_heads, device)

    model.load_state_dict(torch.load(args.checkpoint_path))
    return model

def cls_test(args, model, test_dataloader, device):
    data_embeddings = []
    data_labels = []
    for i_batch, sample_batched in enumerate(tqdm(test_dataloader)):
        ids, seg = sample_batched['input']
        label_tensor = sample_batched['label']
        ids = ids.to(device)
        seg = seg.to(device)

        with torch.no_grad():
            out = model.get_embeddings(ids, seg)

            cls_embeddings = out[:,0,:]

            for i in range(cls_embeddings.shape[0]):
                data_embeddings.append(cls_embeddings[i].cpu().numpy())
                data_labels.append(label_tensor[i])
    latent = np.array(data_embeddings)[:args.n_samples]
    unique_labels = list(np.unique(data_labels[:args.n_samples]))
    y_true = np.array(list(map(lambda x: unique_labels.index(x), data_labels[:args.n_samples])))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title("Representation of the Latent Space")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    _embedding = umap.UMAP(random_state=42).fit_transform(latent)
    print(_embedding.shape)

    ax.scatter(_embedding[:, 0],
               _embedding[:, 1],
               c=y_true,
               s=1,
               cmap='Spectral')
    plt.show()

    metrics = ['manhattan', 'cosine', 'minkowski']
    scores = []
    neighbour_size = 1
    gt = y_true
    for metric in tqdm(metrics):
        nbrs = NearestNeighbors(n_neighbors=neighbour_size+1, metric=metric).fit(latent)
        _, neighbour_indices = nbrs.kneighbors(latent)
        neighbour_indices = neighbour_indices.astype(np.int32)[:,1:]

        y_neighbors = gt[neighbour_indices]

        y_mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_neighbors)
        scores.append(np.sum(y_mode == gt)/args.n_samples)

    print(scores)
    print("Best Metric: ", metrics[scores.index(max(scores))], "Accuracy: ", max(scores))
    best_metric = metrics[scores.index(max(scores))]






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="unseen.tsv",
                        help="path to the input(unseen) tsv.")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--k_mer', type=int, default=4, help="k for k-mer.")
    parser.add_argument('--checkpoint_path', type=str, default="outputs/mask_ratio_0_25/k_mer_4/model_1000.pth",
                        help="k for k-mer.")

    parser.add_argument('--max_len', type=int, default=512,
                        help="input sequence length.")  # TODO: figure out what this is. write better explanation.
    parser.add_argument('--d_model', type=int, default=768, help="number of hidden dimensions of BERT.")
    parser.add_argument('--n_heads', type=int, default=12, help="number of attention heads.")
    parser.add_argument('--n_layers', type=int, default=12, help="number of transformer blocks.")
    parser.add_argument('--n_samples', type=int, default=20000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    test_dataloader = get_dataloader(args)

    model = initial_model(args, device)

    cls_test(args, model, test_dataloader, device)
