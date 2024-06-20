"""
Input/output utilities.
"""

import os
import urllib
from inspect import getsourcefile

import torch
from transformers import BertConfig, BertForMaskedLM, BertForTokenClassification

from .utils import remove_extra_pre_fix

PACKAGE_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


def get_project_root() -> str:
    return os.path.dirname(PACKAGE_DIR)


def safe_save_model(modules, checkpoint_path=None, config=None, **kwargs):
    """
    Save a model to a checkpoint file, along with any additional data.

    Parameters
    ----------
    modules : dict
        A dictionary of modules to save. The keys are the names of the modules
        and the values are the modules themselves.
    checkpoint_path : str, optional
        Path to the checkpoint file. If not provided, the path will be taken
        from the config object.
    config : :class:`argparse.Namespace`, optional
        A configuration object containing the checkpoint path.
    **kwargs
        Additional data to save to the checkpoint file.
    """
    if checkpoint_path is not None:
        pass
    elif config is not None and hasattr(config, "checkpoint_path"):
        checkpoint_path = config.checkpoint_path
    else:
        raise ValueError("No checkpoint path provided")
    print(f"\nSaving model to {checkpoint_path}")
    # Save to a temporary file first, then move the temporary file to the target
    # destination. This is to prevent clobbering the checkpoint with a partially
    # saved file, in the event that the saving process is interrupted. Saving
    # the checkpoint takes a little while and can be disrupted by preemption,
    # whereas moving the file is an atomic operation.
    tmp_a, tmp_b = os.path.split(checkpoint_path)
    tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
    data = {k: v.state_dict() for k, v in modules.items()}
    data.update(kwargs)
    if config is not None:
        data["config"] = config

    torch.save(data, tmp_fname)
    os.rename(tmp_fname, checkpoint_path)
    print(f"Saved model to  {checkpoint_path}")


def load_pretrained_model(checkpoint_path, device=None):
    """
    Load a pretrained model from a checkpoint file.

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
    print(f"\nLoading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    assert "bert_config" in ckpt  # You may be trying to load an old checkpoint

    bert_config = BertConfig(**ckpt["bert_config"])
    model = BertForTokenClassification(bert_config)
    model.load_state_dict(remove_extra_pre_fix(ckpt["model"]))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, ckpt


def load_old_pretrained_model(checkpoint_path, k_mer, device=None):
    """
    Load a pretrained model using the publised format from a checkpoint file.

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
    vocab_size = 4**k_mer + 3
    configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)
    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    model = BertForMaskedLM(configuration)
    # Load the weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    single_state_dict = {}
    for key in state_dict:
        new_key = key.replace("module.", "")
        single_state_dict[new_key] = state_dict[key]
    model.load_state_dict(single_state_dict, strict=False)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def load_inference_model(checkpoint_path, config, device=None):
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
        if not config.from_paper:
            model, ckpt = load_pretrained_model(checkpoint_path, device=device)
        else:
            model = load_old_pretrained_model(checkpoint_path, config, device=device)

    else:
        arch = f"{config.k_mer}_{config.n_heads}_{config.n_layers}"

        available_archs = {
            "4_12_12": "https://vault.cs.uwaterloo.ca/s/5XdqgegTC6xe2yQ/download/new_model_4.pth",
            "5_12_12": "https://vault.cs.uwaterloo.ca/s/Cb6yzBpPdHQzjzg/download/new_model_5.pth",
            "6_12_12": "https://vault.cs.uwaterloo.ca/s/GCfZdeZEDCcdSNf/download/new_model_6.pth",
        }

        print(f"Checkpoint PATH not provided, searching for model {arch} in model_checkpoints/")

        if config.from_paper:
            # create "model_chekpoints" folder
            if not os.path.isdir("model_checkpoints/"):
                os.mkdir("model_checkpoints")

            if not os.path.exists(f"model_checkpoints/{arch}.pt"):
                # download model from the server
                urllib.request.urlretrieve(available_archs[arch], filename=f"model_checkpoints/{arch}.pt")

            checkpoint_path = f"model_checkpoints/{arch}.pt"
            model = load_old_pretrained_model(checkpoint_path, config, device=device)

        else:
            if os.path.exists(f"model_checkpoints/{arch}.pt"):
                model, ckpt = load_pretrained_model(f"model_checkpoints/{arch}.pt", device=device)
            else:
                raise NotImplementedError(
                    f"A new model checkpoint for {arch}.pt was not found. Automatic download is only available for published models"
                )

    if not ckpt:
        # Dummy assigment for compatibility
        ckpt = {
            "model": None,
            "optimizer": None,
            "scheduler": None,
            "config": config,
            "epoch": 0,
            "total_step": 0,
            "n_samples_seen": 0,
            "bert_config": None,
        }

    return model, ckpt
