import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def extract_clean_barcode_list(barcodes: np.ndarray) -> list[str]:
    """Extract non-aligned barcodes from original numpy array format into a simple list.

    :param barcodes: array of size (num_samples, 1) in the following form:
        array([
            [
                array([
                    <barcode_str>
                ])
            ],
            ...
        ])
    (This format is due to some weird setting in the original .mat file from INSECT)
    :return: list of barcode strings
    """
    return [str(i[0][0]) for i in barcodes]


def extract_clean_barcode_list_for_aligned(barcodes: np.ndarray) -> list[str]:
    """Extract aligned barcodes from original numpy array format into a simple list.

    :param barcodes: array of size (1, num_samples) in the following form:
        array([
            [
                array([
                    <barcode_str>
                ]),
                ...
            ]
        ])
    (This format is due to some weird setting in the original .mat file from INSECT)
    :return: list of barcode strings
    """
    return [str(i[0]) for i in barcodes.squeeze()]


def extract_dna_features(
    model_name: str,
    model: nn.Module,
    sequence_pipeline: Callable[[str], torch.Tensor],
    barcodes: list[str],
    labels: np.ndarray,
    output: str,
    extract_class_embeddings: bool = True,
):
    """Extract features from DNA barcodes and save to CSV.

    :param model_name: name of model (barcodebert, dnabert, dnabert2)
    :param model: model to be run on DNA barcode inputs to generate features
    :param sequence_pipeline: callable to preprocess the barcode input into a form ingestible by the model
    :param barcodes: list of barcode samples
    :param labels: list of corresponding class labels as integers between 1 and num_classes, inclusive.
    :param output: path to output file
    :param extract_class_embeddings: if True, compute mean class embedding per label, else save each sample embedding
    independently. Surrogate-species BZSL requires this to be True, whereas genus-species BZSL requries this to be
    False. Defaults to True
    """
    all_label = np.unique(labels)
    all_label.sort()
    dict_emb = {}

    with torch.no_grad():
        all_embeddings = []
        pbar = tqdm(enumerate(labels), total=len(labels))
        for i, label in pbar:
            pbar.set_description("Extracting features: ")
            _barcode = barcodes[i]

            # run model
            if model_name == "dnabert2":
                x = sequence_pipeline(_barcode).to(device)
                x = model(x)[0]
                x = torch.max(x[0], dim=0)[0]  # max pooling
            else:
                x = torch.tensor(sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
                x = model(x).hidden_states[-1]
                x = x.mean(1)  # Global Average Pooling excluding CLS token
            x = x.cpu().numpy()

            if extract_class_embeddings:
                if str(label) not in dict_emb.keys():
                    dict_emb[str(label)] = []
                dict_emb[str(label)].append(x)
            else:
                all_embeddings.append(x)

    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)

    # aggregate and save embeddings
    if extract_class_embeddings:
        class_embed = []
        for i in all_label:
            class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
        class_embed = np.array(class_embed, dtype=object)
        class_embed = class_embed.T.squeeze()
        np.savetxt(output, class_embed, delimiter=",")
    else:
        all_embeddings = np.array(all_embeddings).squeeze()

        np.savetxt(output, all_embeddings, delimiter=",")
    print("DNA embeddings is saved.")
