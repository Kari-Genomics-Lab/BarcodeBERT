"""
Utility functions.
"""

import os
import random
import secrets
import string
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader


def check_is_distributed():
    r"""
    Check if the current job is running in distributed mode.

    Returns
    -------
    bool
        Whether the job is running in distributed mode.
    """
    return (
        "WORLD_SIZE" in os.environ
        and "RANK" in os.environ
        and "LOCAL_RANK" in os.environ
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
    )


def setup_slurm_distributed():
    r"""
    Use SLURM environment variables to set up environment variables needed for DDP.

    Note: This is not used when using torchrun, as that sets RANK etc. for us,
    but is useful if you're using srun without torchrun (i.e. using srun within
    the sbatch file to lauching one task per GPU).
    """
    if "WORLD_SIZE" in os.environ:
        pass
    elif "SLURM_NNODES" in os.environ and "SLURM_GPUS_ON_NODE" in os.environ:
        os.environ["WORLD_SIZE"] = str(int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_GPUS_ON_NODE"]))
    elif "SLURM_NPROCS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        if int(os.environ["RANK"]) > 0 and "WORLD_SIZE" not in os.environ:
            raise EnvironmentError(
                f"SLURM_PROCID is {os.environ['SLURM_PROCID']}, implying"
                " distributed training, but WORLD_SIZE could not be determined."
            )
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "MASTER_ADDR" not in os.environ and "SLURM_NODELIST" in os.environ:
        os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split("-")[0]
    if "MASTER_PORT" not in os.environ and "SLURM_JOB_ID" in os.environ:
        os.environ["MASTER_PORT"] = str(49152 + int(os.environ["SLURM_JOB_ID"]) % 16384)


def get_num_cpu_available():
    r"""
    Get the number of available CPU cores.

    Uses :func:`os.sched_getaffinity` if available, otherwise falls back to
    :func:`os.cpu_count`.

    Returns
    -------
    ncpus : int
        The number of available CPU cores.
    """
    try:
        # This is the number of CPUs available to this process, which may
        # be smaller than the number of CPUs on the system.
        # This command is only available on Unix-like systems.
        return len(os.sched_getaffinity(0))
    except Exception:
        # Fall-back for Windows or other systems which don't support sched_getaffinity
        warnings.warn(
            "Unable to determine number of available CPUs available to this python"
            " process specifically. Falling back to the total number of CPUs on the"
            " system.",
            RuntimeWarning,
            stacklevel=2,
        )
        return os.cpu_count()


def set_rng_seeds_fixed(seed, all_gpu=True):
    r"""
    Seed pseudo-random number generators throughout python's random module, numpy.random, and pytorch.

    Parameters
    ----------
    seed : int
        The random seed to use. Should be between 0 and 4294967295 to ensure
        unique behaviour for numpy, and between 0 and 18446744073709551615 to
        ensure unique behaviour for pytorch.
    all_gpu : bool, default=True
        Whether to set the torch seed on every GPU. If ``False``, only the
        current GPU has its seed set.

    Returns
    -------
    None
    """
    # Note that random, numpy, and pytorch all use different RNG methods/
    # implementations, so you'll get different outputs from each of them even
    # if you use the same seed for them all.
    # We use modulo with the maximum values permitted for np.random and torch.
    # If your seed exceeds 4294967295, numpy will have looped around to a
    random.seed(seed)
    np.random.seed(seed % 0xFFFF_FFFF)
    torch.manual_seed(seed % 0xFFFF_FFFF_FFFF_FFFF)
    if all_gpu:
        torch.cuda.manual_seed_all(seed % 0xFFFF_FFFF_FFFF_FFFF)
    else:
        torch.cuda.manual_seed(seed % 0xFFFF_FFFF_FFFF_FFFF)


def worker_seed_fn(worker_id):
    r"""
    Seed builtin :mod:`random` and :mod:`numpy`.

    A worker initialization function for :class:`torch.utils.data.DataLoader`
    objects which seeds builtin :mod:`random` and :mod:`numpy` with the
    torch seed for the worker.

    Parameters
    ----------
    worker_id : int
        The ID of the worker.
    """
    worker_seed = torch.utils.data.get_worker_info().seed
    random.seed(worker_seed)
    np.random.seed(worker_seed % 0xFFFF_FFFF)


def determine_epoch_seed(seed, epoch):
    r"""
    Determine the seed to use for the random number generator for a given epoch.

    Parameters
    ----------
    seed : int
        The original random seed, used to generate the sequence of seeds for
        the epochs.
    epoch : int
        The epoch for which to determine the seed.

    Returns
    -------
    epoch_seed : int
        The seed to use for the random number generator for the given epoch.
    """
    if epoch == 0:
        raise ValueError("Epoch must be indexed from 1, not 0.")
    random.seed(seed)
    # Generate a seed for every epoch so far. We have to traverse the
    # series of RNG calls to reach the next value (our next seed). The final
    # value is the one for our current epoch.
    # N.B. We use random.randint instead of torch.randint because torch.randint
    # only supports int32 at most (max value of 0xFFFF_FFFF).
    for _ in range(epoch):
        epoch_seed = random.randint(0, 0xFFFF_FFFF_FFFF_FFFF)
    return epoch_seed


def generate_id(length: int = 8) -> str:
    r"""
    Generate a random base-36 string of `length` digits.

    Parameters
    ----------
    length : int, default=8
        Length of the string to generate.

    Returns
    -------
    id : str
        The randomly generated id.
    """
    # Borrowed from https://github.com/wandb/wandb/blob/0e00efd/wandb/sdk/lib/runid.py
    # under the MIT license.
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def count_parameters(model, only_trainable=True):
    r"""
    Count the number of (trainable) parameters within a model and its children.

    Parameters
    ----------
    model : torch.nn.Model
        The parametrized model.
    only_trainable : bool, optional
        Whether the count should be restricted to only trainable parameters
        (default), otherwise all parameters are included.
        Default is ``True``.

    Returns
    -------
    int
        Total number of (trainable) parameters possessed by the model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def concat_all_gather(tensor, **kwargs):
    r"""
    Gather a tensor over all processes and concatenate them into one.

    Similar to :func:`torch.distributed.all_gather`, except this function
    concatenates the result into a single tensor instead of a list of tensors.

    Parameters
    ----------
    tensor : torch.Tensor
        The distributed tensor on the current process.
    group : ProcessGroup, optional
        The process group to work on. If ``None``, the default process group
        will be used.
    async_op : bool, default=False
        Whether this op should be an async op.

    Returns
    -------
    gathered_tensor : torch.Tensor
        The contents of ``tensor`` from every distributed process, gathered
        together. None of the entries support a gradient.

    Warning
    -------
    As with :func:`torch.distributed.all_gather`, this has no gradient.
    """
    world_size = torch.distributed.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, **kwargs)
    output = torch.cat(tensors_gather, dim=0)
    return output
