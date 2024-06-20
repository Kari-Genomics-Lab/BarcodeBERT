"""
embedders.py
------------
Wrapper classes for embedding sequences with pretrained DNA language models using a common interface.
The wrapper classes handle loading the models and tokenizers, and embedding the sequences. As far as possible,
models are downloaded automatically.
They also handle removal of special tokens, and optionally upsample the embeddings to the original sequence length.

Embedders can be used as follows. Please check the individual classes for more details on the arguments.

``embedder = EmbedderClass(model_name, some_additional_config_argument=6)``

``embedding = embedder(sequence, remove_special_tokens=True, upsample_embeddings=True)``

"""

# Adapted from https://github.com/frederikkemarin/BEND/blob/main/bend/models/dnabert2_padding.py
# Which was adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
# Which was adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py


import os
import urllib
from functools import partial
from itertools import product
from typing import Iterable, List

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    BigBirdModel,
    logging,
)

from barcodebert.datasets import KmerTokenizer
from barcodebert.io import load_old_pretrained_model, load_pretrained_model

from .models.dnabert2 import BertModel as DNABert2BertModel
from .models.hyena_dna import CharacterTokenizer, HyenaDNAPreTrainedModel

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseEmbedder:
    """Base class for embedders.
    All embedders should inherit from this class.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the embedder. Calls `load_model` with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments. Passed to `load_model`.
        **kwargs
            Keyword arguments. Passed to `load_model`.
        """
        self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Load the model. Should be implemented by the inheriting class."""
        raise NotImplementedError

    def embed(self, sequences: str, *args, **kwargs):
        """Embed a sequence. Should be implemented by the inheriting class.

        Parameters
        ----------
        sequences : str
            The sequences to embed.
        """
        raise NotImplementedError

    def __call__(self, sequence: str, *args, **kwargs):
        """Embed a single sequence. Calls `embed` with the given arguments.

        Parameters
        ----------
        sequence : str
            The sequence to embed.
        *args
            Positional arguments. Passed to `embed`.
        **kwargs
            Keyword arguments. Passed to `embed`.

        Returns
        -------
        np.ndarray
            The embedding of the sequence.
        """
        return self.embed([sequence], *args, disable_tqdm=True, **kwargs)[0]
        return embeddings


##
## DNABert https://doi.org/10.1093/bioinformatics/btab083
## Download from https://github.com/jerryji1993/DNABERT


class DNABertEmbedder(BaseEmbedder):
    """Embed using the DNABert model https://doi.org/10.1093/bioinformatics/btab083"""

    def load_model(self, model_path: str = "../../external-models/DNABERT/", kmer: int = 6, **kwargs):
        """Load the DNABert model.

        Parameters
        ----------
        model_path : str
            The path to the model directory. Defaults to "../../external-models/DNABERT/".
            The DNABERT models need to be downloaded manually as indicated in the DNABERT repository at https://github.com/jerryji1993/DNABERT.
        kmer : int
            The kmer size of the model. Defaults to 6.

        """

        dnabert_path = model_path
        # dnabert_path = f'{dnabert_path}/DNABERT{kmer}/'
        # check if path exists

        if not os.path.exists(dnabert_path):
            print(
                f"Path {dnabert_path} does not exists, check if the wrong path was given. If not download from https://github.com/jerryji1993/DNABERT"
            )

        config = BertConfig.from_pretrained(dnabert_path)
        self.tokenizer = BertTokenizer.from_pretrained(dnabert_path)
        self.model = BertModel.from_pretrained(dnabert_path, config=config)
        self.model.to(device)

        self.kmer = kmer

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
        remove_special_tokens: bool = True,
        upsample_embeddings: bool = False,
    ):
        """
        Embed a list of sequences.

        Parameters
        ----------
        sequences : List[str]
            The sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the special tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.

        Returns
        -------
        List[np.ndarray]
            The embeddings of the sequences.
        """
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):
                sequence = [sequence]
                kmers = self._seq2kmer_batch(sequence, self.kmer)
                model_input = self.tokenizer.batch_encode_plus(
                    kmers,
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=len(sequence[0]) + 2,
                    return_tensors="pt",
                )["input_ids"]

                if model_input.shape[1] > 512:
                    model_input = torch.split(model_input, 512, dim=1)
                    output = []
                    for chunk in model_input:
                        output.append(self.model(chunk.to(device))[0].detach().cpu())
                    output = torch.cat(output, dim=1).numpy()
                else:
                    output = self.model(model_input.to(device))[0].detach().cpu().numpy()
                embedding = output

                if upsample_embeddings:
                    embedding = self._repeat_embedding_vectors(embedding)

                embeddings.append(embedding[:, 1:-1] if remove_special_tokens else embedding)

        return embeddings

    @staticmethod
    def _seq2kmer(seq, k):
        """
        Convert original sequence to kmers

        Arguments:
        seq -- str, original sequence.
        k -- int, kmer of length k specified.

        Returns:
        kmers -- str, kmers separated by space
        """
        kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
        kmers = " ".join(kmer)
        return kmers

    def _seq2kmer_batch(self, batch, k=3, step_size=1, kmerise=True):
        return list(map(partial(self._seq2kmer, k=k), batch))

    # repeating.
    # GATTTATTAGGGGAGATTTTATATATCCCGA
    # kmer =3, input = 31 --> embedding = 29 --> repeat first and last once.
    # kmer =3, input = 32 --> embedding = 30 --> repeat first and last once.

    # kmer=4 input = 31 --> embedding = 28 --> repeat first once and last twice.
    # kmer=4 input = 32 --> embedding = 29
    # kmer=4 input = 33 --> embedding = 30

    # kmer=5 input = 31 --> embedding = 27 --> repeat first twice and last twice.
    # kmer=5 input = 32 --> embedding = 28 --> repeat first twice and last twice.

    # kmer=6 input = 31 --> embedding = 26 --> repeat first twice and last three times.
    def _repeat_embedding_vectors(self, embeddings: np.ndarray, has_special_tokens: bool = True):
        """Repeat embeddings at sequence edges to match input length"""
        if has_special_tokens:
            cls_vector = embeddings[:, [0]]
            sep_vector = embeddings[:, [-1]]
            embeddings = embeddings[:, 1:-1]

        # repeat first and last embedding
        if self.kmer == 3:
            embeddings = np.concatenate([embeddings[:, [0]], embeddings, embeddings[:, [-1]]], axis=1)
        elif self.kmer == 4:
            embeddings = np.concatenate(
                [embeddings[:, [0]], embeddings, embeddings[:, [-1]], embeddings[:, [-1]]], axis=1
            )
        elif self.kmer == 5:
            embeddings = np.concatenate(
                [embeddings[:, [0]], embeddings, embeddings[:, [0]], embeddings[:, [-1]], embeddings[:, [-1]]], axis=1
            )
        elif self.kmer == 6:
            embeddings = np.concatenate(
                [
                    embeddings[:, [0]],
                    embeddings,
                    embeddings[:, [0]],
                    embeddings[:, [-1]],
                    embeddings[:, [-1]],
                    embeddings[:, [-1]],
                ],
                axis=1,
            )

        if has_special_tokens:
            embeddings = np.concatenate([cls_vector, embeddings, sep_vector], axis=1)

        return embeddings


# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerEmbedder(BaseEmbedder):
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """

    def load_model(self, model_name, **kwargs):
        """
        Load the Nuclieotide Transformer (NT) model.

        Parameters
        ----------
        model_name : str
            The name of the model to load.
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory. We check whether the model_name
            contains 'v2' to determine whether we need to follow the V2 model API or not.
        """

        # Get pretrained model
        if "v2" in model_name:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.max_seq_len = 12282  # "model_max_length": 2048, --> 12,288
            self.max_tokens = 2048
            self.v2 = True
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_seq_len = 5994  # "model_max_length": 1000, 6-mer --> 6000
            self.max_tokens = 1000
            self.v2 = False
        self.model.to(device)

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
        remove_special_tokens: bool = True,
        upsample_embeddings: bool = False,
    ):
        """
        Embed sequences using the Nuclieotide Transformer (NT) model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
             Whether to remove the special tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.

        Returns
        -------
        List[np.ndarray]
            List of embeddings.
        """

        self.model.eval()
        cls_tokens = []
        embeddings = []

        with torch.no_grad():
            for n, s in enumerate(tqdm(sequences, disable=disable_tqdm)):
                # print('sequence', n)
                s_chunks = [
                    s[chunk : chunk + self.max_seq_len] for chunk in range(0, len(s), self.max_seq_len)
                ]  # split into chunks
                embedded_seq = []
                for n_chunk, chunk in enumerate(s_chunks):  # embed each chunk
                    tokens_ids = self.tokenizer(chunk, return_tensors="pt")["input_ids"].int().to(device)
                    if len(tokens_ids[0]) > self.max_tokens:  # too long to fit into the model
                        split = torch.split(tokens_ids, self.max_tokens, dim=-1)
                        if self.v2:
                            outs = [
                                self.model(item, output_hidden_states=True)["hidden_states"][-1].detach().cpu().numpy()
                                for item in split
                            ]
                        else:
                            outs = [self.model(item)["last_hidden_state"].detach().cpu().numpy() for item in split]
                        outs = np.concatenate(outs, axis=1)
                    else:
                        if self.v2:
                            outs = (
                                self.model(tokens_ids, output_hidden_states=True)["hidden_states"][-1]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        else:
                            outs = (
                                self.model(tokens_ids)["last_hidden_state"].detach().cpu().numpy()
                            )  # get last hidden state

                    if upsample_embeddings:
                        outs = self._repeat_embedding_vectors(self.tokenizer.convert_ids_to_tokens(tokens_ids[0]), outs)
                    embedded_seq.append(outs[:, 1:] if remove_special_tokens else outs)
                    # print('chunk', n_chunk, 'chunk length', len(chunk), 'tokens length', len(tokens_ids[0]), 'chunk embedded shape', outs.shape)
                embeddings.append(np.concatenate(embedded_seq, axis=1))

        return embeddings

    @staticmethod
    def _repeat_embedding_vectors(tokens: Iterable[str], embeddings: np.ndarray, has_special_tokens: bool = True):
        """
        Nucleotide transformer uses 6-mer embedding, but single-embedding for remaining nucleotides.
        """
        assert len(tokens) == embeddings.shape[1], "Number of tokens and embeddings must match."
        new_embeddings = []
        for idx, token in enumerate(tokens):

            if has_special_tokens and idx == 0:
                new_embeddings.append(embeddings[:, [idx]])  # (1, hidden_dim)
                continue
            token_embedding = embeddings[:, [idx]]  # (1, hidden_dim)
            new_embeddings.extend([token_embedding] * len(token))

        # list of (1,1, 768) arrays
        new_embeddings = np.concatenate(new_embeddings, axis=1)
        return new_embeddings


class HyenaDNAEmbedder(BaseEmbedder):
    """Embed using the HyenaDNA model https://arxiv.org/abs/2306.15794"""

    def load_model(self, model_path="pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen", **kwargs):
        # '''Load the model from the checkpoint path
        # 'hyenadna-tiny-1k-seqlen'
        # 'hyenadna-small-32k-seqlen'
        # 'hyenadna-medium-160k-seqlen'
        # 'hyenadna-medium-450k-seqlen'
        # 'hyenadna-large-1m-seqlen'
        # '''
        # you only need to select which model to use here, we'll do the rest!
        """
        Load the HyenaDNA model.

        Parameters
        ----------
        model_path : str, optional
            Path to the model checkpoint. Defaults to 'pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen'.
            If the path does not exist, the model will be downloaded from HuggingFace. Rather than just downloading the model,
            HyenaDNA's `from_pretrained` method relies on cloning the HuggingFace-hosted repository, and using git lfs to download the model.
            This requires git lfs to be installed on your system, and will fail if it is not.


        """
        checkpoint_path, model_name = os.path.split(model_path)
        max_lengths = {
            "hyenadna-tiny-1k-seqlen": 1024,
            "hyenadna-small-32k-seqlen": 32768,
            "hyenadna-medium-160k-seqlen": 160000,
            "hyenadna-medium-450k-seqlen": 450000,
            "hyenadna-large-1m-seqlen": 1_000_000,
        }

        self.max_length = max_lengths[model_name]  # auto selects

        # all these settings are copied directly from huggingface.py

        # data settings:
        use_padding = True
        rc_aug = False  # reverse complement augmentation
        add_eos = False  # add end of sentence token

        # we need these for the decoder head, if using
        use_head = False
        n_classes = 2  # not used for embeddings only

        # you can override with your own backbone config here if you want,
        # otherwise we'll load the HF one in None
        backbone_cfg = None

        is_git_lfs_repo = os.path.exists(".git/hooks/pre-push")
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            checkpoint_path,
            model_name,
            download=not os.path.exists(model_path),
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

        model.to(device)
        self.model = model

        # NOTE the git lfs download command will add this,
        # but we actually dont use LFS for BEND itself.
        if not is_git_lfs_repo:
            try:
                os.remove(".git/hooks/pre-push")
            except FileNotFoundError:
                pass

        # create tokenizer - NOTE this adds CLS and SEP tokens when add_special_tokens=False
        self.tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],  # add DNA characters, N is uncertain
            model_max_length=self.max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side="left",  # since HyenaDNA is causal, we pad on the left
        )

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
        remove_special_tokens: bool = True,
        upsample_embeddings: bool = False,
    ):
        """Embeds a list of sequences using the HyenaDNA model.
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.
            Only provided for compatibility with other embedders. GPN embeddings are already the same length as the input sequence.
        Returns
        -------

        embeddings : List[np.ndarray]
            List of embeddings.
        """

        # # prep model and forward
        # model.to(device)
        #             with torch.inference_mode():

        embeddings = []
        with torch.inference_mode():
            for s in tqdm(sequences, disable=disable_tqdm):
                chunks = [
                    s[chunk : chunk + self.max_length] for chunk in range(0, len(s), self.max_length)
                ]  # split into chunks
                embedded_chunks = []
                for n_chunk, chunk in enumerate(chunks):
                    #### Single embedding example ####

                    # create a sample 450k long, prepare
                    # sequence = 'ACTG' * int(self.max_length/4)
                    tok_seq = self.tokenizer(chunk)  # adds CLS and SEP tokens
                    tok_seq = tok_seq["input_ids"]  # grab ids

                    # place on device, convert to tensor
                    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
                    tok_seq = tok_seq.to(device)

                    output = self.model(tok_seq)
                    if remove_special_tokens:
                        output = output[:, 1:-1]

                    embedded_chunks.append(output.detach().cpu().numpy())

                embedding = np.concatenate(embedded_chunks, axis=1)

                embeddings.append(embedding)

        return embeddings

    # print(embeddings.shape)  # embeddings here!


class DNABert2Embedder(BaseEmbedder):
    """
    Embed using the DNABERT2 model https://arxiv.org/pdf/2306.15006.pdf
    """

    def load_model(self, model_name="zhihan1996/DNABERT-2-117M", **kwargs):
        """
        Load the DNABERT2 model.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "zhihan1996/DNABERT-2-117M".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.
        """

        # keep the source in this repo to avoid using flash attn.
        self.model = DNABert2BertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)

        # https://github.com/Zhihan1996/DNABERT_2/issues/2
        self.max_length = 10000  # nucleotides.

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
        remove_special_tokens: bool = True,
        upsample_embeddings: bool = False,
    ):
        """Embeds a list sequences using the DNABERT2 model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        """
        # '''
        # Note that this model uses byte pair encoding.
        # upsample_embedding repeats BPE token embeddings so that each nucleotide has its own embedding.
        # The [CLS] and [SEP] tokens are removed from the output if remove_special_tokens is True.
        # '''

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):

                chunks = [
                    sequence[chunk : chunk + self.max_length] for chunk in range(0, len(sequence), self.max_length)
                ]  # split into chunks

                embedded_chunks = []
                for n_chunk, chunk in enumerate(chunks):
                    # print(n_chunk)

                    input_ids = self.tokenizer(
                        chunk, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False
                    )["input_ids"]
                    # print(input_ids.shape)
                    output = self.model(input_ids.to(device))[0].detach().cpu().numpy()

                    if upsample_embeddings:
                        output = self._repeat_embedding_vectors(
                            self.tokenizer.convert_ids_to_tokens(input_ids[0]), output
                        )

                    # for intermediate chunks the special tokens need to go.
                    # if we only have 1 chunk, keep them for now.
                    if len(chunks) != 1:
                        if n_chunk == 0:
                            output = output[:, :-1]  # no SEP
                        elif n_chunk == len(chunks) - 1:
                            output = output[:, 1:]  # no CLS
                        else:
                            output = output[:, 1:-1]  # no CLS and no SEP

                    embedded_chunks.append(output)

                embedding = np.concatenate(embedded_chunks, axis=1)

                if remove_special_tokens:
                    embedding = embedding[:, 1:-1]

                embeddings.append(embedding)

        return embeddings

    # GATTTATTAGGGGAGATTTTATATATCCCGA
    # ['[CLS]', 'G', 'ATTTATT', 'AGGGG', 'AGATT', 'TTATAT', 'ATCCCG', 'A', '[SEP]']
    @staticmethod
    def _repeat_embedding_vectors(tokens: Iterable[str], embeddings: np.ndarray, has_special_tokens: bool = True):
        """
        Byte-pair encoding merges a variable number of letters into one token.
        We need to repeat each token's embedding vector for each letter in the token.
        """
        assert len(tokens) == embeddings.shape[1], "Number of tokens and embeddings must match."
        new_embeddings = []
        for idx, token in enumerate(tokens):

            if has_special_tokens and (idx == 0 or idx == len(tokens) - 1):
                new_embeddings.append(embeddings[:, [idx]])  # (1, 768)
                continue
            token_embedding = embeddings[:, [idx]]  # (1, 768)
            if token == "[UNK]":
                new_embeddings.extend([token_embedding])
            else:
                new_embeddings.extend([token_embedding] * len(token))

        # list of (1,1, 768) arrays
        new_embeddings = np.concatenate(new_embeddings, axis=1)
        return new_embeddings


class DNABertSEmbedder(BaseEmbedder):
    """
    Embed using the DNABERTS model https://arxiv.org/abs/2402.08777
    """

    def load_model(self, model_name="zhihan1996/DNABERT-S", **kwargs):
        """
        Load the DNABERTS model.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "zhihan1996/DNABERT-S".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.
        """

        # keep the source in this repo to avoid using flash attn.
        self.model = DNABert2BertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)

        # https://github.com/Zhihan1996/DNABERT_2/issues/2
        self.max_length = 10000  # nucleotides.

    def embed(
        self,
        sequences: List[str],
        disable_tqdm: bool = False,
        remove_special_tokens: bool = True,
        upsample_embeddings: bool = False,
    ):
        """Embeds a list sequences using the DNABERT2 model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        """
        # '''
        # Note that this model uses byte pair encoding.
        # upsample_embedding repeats BPE token embeddings so that each nucleotide has its own embedding.
        # The [CLS] and [SEP] tokens are removed from the output if remove_special_tokens is True.
        # '''
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):

                chunks = [
                    sequence[chunk : chunk + self.max_length] for chunk in range(0, len(sequence), self.max_length)
                ]  # split into chunks

                embedded_chunks = []
                for n_chunk, chunk in enumerate(chunks):
                    # print(n_chunk)

                    input_ids = self.tokenizer(
                        chunk, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False
                    )["input_ids"]
                    # print(input_ids.shape)
                    output = self.model(input_ids.to(device))[0].detach().cpu().numpy()

                    if upsample_embeddings:
                        output = self._repeat_embedding_vectors(
                            self.tokenizer.convert_ids_to_tokens(input_ids[0]), output
                        )

                    # for intermediate chunks the special tokens need to go.
                    # if we only have 1 chunk, keep them for now.
                    if len(chunks) != 1:
                        if n_chunk == 0:
                            output = output[:, :-1]  # no SEP
                        elif n_chunk == len(chunks) - 1:
                            output = output[:, 1:]  # no CLS
                        else:
                            output = output[:, 1:-1]  # no CLS and no SEP

                    embedded_chunks.append(output)

                embedding = np.concatenate(embedded_chunks, axis=1)

                if remove_special_tokens:
                    embedding = embedding[:, 1:-1]

                embeddings.append(embedding)

        return embeddings

    # GATTTATTAGGGGAGATTTTATATATCCCGA
    # ['[CLS]', 'G', 'ATTTATT', 'AGGGG', 'AGATT', 'TTATAT', 'ATCCCG', 'A', '[SEP]']
    @staticmethod
    def _repeat_embedding_vectors(tokens: Iterable[str], embeddings: np.ndarray, has_special_tokens: bool = True):
        """
        Byte-pair encoding merges a variable number of letters into one token.
        We need to repeat each token's embedding vector for each letter in the token.
        """
        assert len(tokens) == embeddings.shape[1], "Number of tokens and embeddings must match."
        new_embeddings = []
        for idx, token in enumerate(tokens):

            if has_special_tokens and (idx == 0 or idx == len(tokens) - 1):
                new_embeddings.append(embeddings[:, [idx]])  # (1, 768)
                continue
            token_embedding = embeddings[:, [idx]]  # (1, 768)
            if token == "[UNK]":
                new_embeddings.extend([token_embedding])
            else:
                new_embeddings.extend([token_embedding] * len(token))

        # list of (1,1, 768) arrays
        new_embeddings = np.concatenate(new_embeddings, axis=1)
        return new_embeddings


class BarcodeBERTEmbedder(BaseEmbedder):
    """
    Embed using the DNABERTS model https://arxiv.org/abs/2402.08777
    """

    def load_model(self, checkpoint_path=None, from_paper=False, k_mer=8, n_heads=4, n_layers=4, **kwargs):
        """
        Load a pretrained model it can be downloaded or it can be from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the pretrained checkpoint file.

        Returns
        -------
        model : torch.nn.Module
            The pretrained model.
        ckpt : dict
            The contents of the checkpoint file.
        """

        ckpt = None
        if checkpoint_path:
            print(f"\nLoading model from {checkpoint_path}")
            if not from_paper:
                model, ckpt = load_pretrained_model(checkpoint_path, device=device)
            else:
                model = load_old_pretrained_model(checkpoint_path, config, device=device)

        else:
            arch = f"{k_mer}_{n_heads}_{n_layers}"

            available_archs = {
                "4_12_12": "https://vault.cs.uwaterloo.ca/s/5XdqgegTC6xe2yQ/download/new_model_4.pth",
                "5_12_12": "https://vault.cs.uwaterloo.ca/s/Cb6yzBpPdHQzjzg/download/new_model_5.pth",
                "6_12_12": "https://vault.cs.uwaterloo.ca/s/GCfZdeZEDCcdSNf/download/new_model_6.pth",
            }

            if from_paper:
                print(f"Checkpoint PATH not provided, searching for model {arch} in pretrained_models/")
                # create "model_chekpoints" folder
                if not os.path.isdir("pretrained_models/"):
                    os.mkdir("pretrained_models")

                if not os.path.exists(f"pretrained_models/{arch}.pt"):
                    # download model from the server
                    urllib.request.urlretrieve(available_archs[arch], filename=f"pretrained_models/{arch}.pt")

                checkpoint_path = f"pretrained_models/{arch}.pt"
                model = load_old_pretrained_model(checkpoint_path, k_mer, device=device)

            else:
                print(f"Checkpoint PATH not provided, searching for model {arch} in checkpoint_pretraining/")
                if os.path.exists(f"model_checkpoints/{arch}/checkpoint_pretraining.pt"):
                    model, ckpt = load_pretrained_model(
                        f"model_checkpoints/{arch}/checkpoint_pretraining.pt", device=device
                    )
                else:
                    raise NotImplementedError(
                        f"A new model checkpoint for {arch}.pt was not found. The automatic download is only available for published models"
                    )

        # Override the last layer
        # model.classifier = nn.Identity()
        self.model = model
        self.model.to(device)

        # tokenizer:
        kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k_mer))
        if from_paper:
            vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
        else:
            vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])  # <UNK> is necessary in the hard case

        tokenizer = KmerTokenizer(k_mer, vocab, stride=k_mer, padding=True, max_len=660)
        self.tokenizer = tokenizer
