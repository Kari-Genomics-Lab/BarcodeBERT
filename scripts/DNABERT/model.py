from itertools import product
from typing import Optional

import torch
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoModel, AutoTokenizer, BertConfig, BertForMaskedLM

from dnabert.tokenization_dna import DNATokenizer
from pablo_bert_with_prediction_head import Bert_With_Prediction_Head

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class KmerTokenizer:
    """
    Applies tokenization based on k-mers
    """

    def __init__(self, k: int, stride: int = 1):
        """
        :param k: size of k-mers for tokens
        :param stride: value by which to shift k-mers. For instance, a shift of 1 represents completely overlapping
        k-mers, whereas a shift of k represents completely non-overlapping k-mers. Defaults to 1
        """
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence: str) -> list[str]:
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i : i + self.k]
            tokens.append(k_mer)
        return tokens


class PadSequence(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        else:
            return dna_sequence + "N" * (self.max_len - len(dna_sequence))

        # return new_sequence


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # 去除 'module.' 前缀
        new_state_dict[key] = value
    return new_state_dict


def split_input_barcode_for_dnabert(barcode: str, k: int = 6) -> str:
    """
    Splits barcode input into overlapping k-mers (stride=1), e.g.

    >>> split_input_barcode_for_dnabert("ACGAATCGA", k=6)

    "ACGAAT CGAATC GAATCG AATCGA"

    :param barcode: input barcode string
    :param k: k-mer size
    :return: barcode with splits by whitespace into k-mers
    """
    if isinstance(barcode, list):
        return [split_input_barcode_for_dnabert(bc, k) for bc in barcode]

    return " ".join([barcode[idx : idx + k] for idx in range(len(barcode) - k + 1)])


def get_dnabert_encoder(tokenizer, max_len: int, k: int = 6):
    def dnabert_encoder(barcode: str):
        preprocessed = split_input_barcode_for_dnabert(barcode, k)

        if isinstance(preprocessed, list):
            return [
                tokenizer.encode_plus(x, max_length=max_len, add_special_tokens=True, pad_to_max_length=True)[
                    "input_ids"
                ]
                for x in preprocessed
            ]
        else:
            return tokenizer.encode_plus(preprocessed, max_length=max_len, add_special_tokens=True, pad_to_max_length=True)[
                "input_ids"
            ]

    return dnabert_encoder


def load_model(args, *, k: int = 6, classification_head: bool = False, num_classes: Optional[int] = None):
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    vocab_size = len(vocab)
    max_len = 660
    pad = PadSequence(max_len)

    print("Initializing the model . . .")

    if args.model == "bioscanbert":
        tokenizer = KmerTokenizer(k, stride=k)
        sequence_pipeline = lambda x: [0, *vocab(tokenizer(pad(x)))]

        configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)

        model = BertForMaskedLM(configuration)
        state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        state_dict = remove_extra_pre_fix(state_dict)
        model.load_state_dict(state_dict)

    elif args.model == "dnabert":
        max_len = 512
        configuration = BertConfig.from_pretrained(
            pretrained_model_name_or_path=args.checkpoint, output_hidden_states=True
        )
        if getattr(args, "use_pablo_tokenizer", False):
            tokenizer = KmerTokenizer(k, stride=k)
            sequence_pipeline = lambda x: vocab(tokenizer(pad(x)))
        else:
            tokenizer = DNATokenizer.from_pretrained(args.checkpoint, do_lower_case=False)
            sequence_pipeline = get_dnabert_encoder(tokenizer, max_len, k)

        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.checkpoint, config=configuration)

    elif args.model == "dnabert2":
        checkpoint = args.checkpoint if args.checkpoint else "zhihan1996/DNABERT-2-117M"
        if getattr(args, "use_pablo_tokenizer", False):
            tokenizer = KmerTokenizer(k, stride=k)
            sequence_pipeline = lambda x: vocab(tokenizer(pad(x)))
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
            sequence_pipeline = lambda x: tokenizer(
                x,
                return_tensors="pt",
                padding="longest",
            )["input_ids"]
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    else:
        raise ValueError(f"Could not parse model name: {args.model}")

    if classification_head:
        model = Bert_With_Prediction_Head(out_feature=num_classes, bert_model=model, model_type=args.model)

    model.to(device)

    print("The model has been succesfully loaded . . .")
    return model, sequence_pipeline
