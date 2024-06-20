"""
Input/output utilities.
"""

import os
from inspect import getsourcefile

import torch
from transformers import BertConfig, BertForMaskedLM, BertForTokenClassification

from baselines.embedders import (
    BarcodeBERTEmbedder,
    DNABert2Embedder,
    DNABertSEmbedder,
    HyenaDNAEmbedder,
    NucleotideTransformerEmbedder,
)

# PACKAGE_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


# def get_project_root() -> str:
#    return os.path.dirname(PACKAGE_DIR)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def load_baseline_model(backbone_name, *args, **kwargs):

    backbones = {
        "NT": NucleotideTransformerEmbedder,
        "Hyena_DNA": HyenaDNAEmbedder,
        "DNABERT-2": DNABert2Embedder,
        "DNABERT-S": DNABertSEmbedder,
        "BarcodeBERT": BarcodeBERTEmbedder,
    }

    # Positional arguments as a list
    # Keyword arguments as a dictionary
    checkpoints = {
        "NT": (["InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"], {}),
        "Hyena_DNA": (["pretrained_models/hyenadna-tiny-1k-seqlen"], {}),
        "DNABERT-2": (["zhihan1996/DNABERT-2-117M"], {}),
        "DNABERT-S": (["zhihan1996/DNABERT-S"], {}),
        "BarcodeBERT": ([None], {}),
    }

    out_dimensions = {
        "NT": 512,
        "Hyena_DNA": 128,
        "DNABERT-2": 768,
        "DNABERT-S": 768,
        "BarcodeBERT": 768,
    }

    positional_args, keyword_args = checkpoints[backbone_name]
    embedder = backbones[backbone_name](*positional_args, **kwargs)
    embedder.hidden_size = out_dimensions[backbone_name]
    return embedder
