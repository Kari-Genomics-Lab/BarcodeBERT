#!/usr/bin/env python

import builtins
import copy
import os
import shutil
import time
from datetime import datetime
from socket import gethostname

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_outputs import TokenClassifierOutput

from barcodebert import utils
from barcodebert.datasets import DNADataset
from barcodebert.evaluation import evaluate
from barcodebert.io import (get_project_root, load_pretrained_model,
                            safe_save_model)

BASE_BATCH_SIZE = 64


class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.base_model = base_model
        self.hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids=None, mask=None, labels=None):
        # Getting the embedding
        outputs = self.base_model(input_ids=input_ids, attention_mask=mask)
        embeddings = outputs.hidden_states[-1]
        GAP_embeddings = embeddings.mean(1)  # TODO: Swap between GAP and CLS
        # calculate losses
        logits = self.classifier(GAP_embeddings.view(-1, self.hidden_size))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


def run(config):
    r"""
    Run training job (one worker if using distributed training).

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DISTRIBUTION ============================================================
    # Setup for distributed training
    utils.setup_slurm_distributed()
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.distributed = utils.check_is_distributed()
    if config.world_size > 1 and not config.distributed:
        raise EnvironmentError(
            f"WORLD_SIZE is {config.world_size}, but not all other required"
            " environment variables for distributed training are set."
        )
    # Work out the total batch size depending on the number of GPUs we are using
    config.batch_size = config.batch_size_per_gpu * config.world_size

    if config.distributed:
        # For multiprocessing distributed training, gpu rank needs to be
        # set to the global rank among all the processes.
        config.global_rank = int(os.environ["RANK"])
        config.local_rank = int(os.environ["LOCAL_RANK"])
        print(
            f"Rank {config.global_rank} of {config.world_size} on {gethostname()}"
            f" (local GPU {config.local_rank} of {torch.cuda.device_count()})."
            f" Communicating with master at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        dist.init_process_group(backend="nccl")
    else:
        config.global_rank = 0

    # Suppress printing if this is not the master process for the node
    if config.distributed and config.global_rank != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.")

    # Check which device to use
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if config.distributed and not use_cuda:
        raise EnvironmentError("Distributed training with NCCL requires CUDA.")
    if not use_cuda:
        device = torch.device("cpu")
    elif config.local_rank is not None:
        device = f"cuda:{config.local_rank}"
    else:
        device = "cuda"

    print(f"Using device {device}", flush=True)

    # LOAD PRE-EMPTION CHECKPOINT =============================================
    checkpoint = None
    config.model_output_dir = None
    if config.checkpoint_path:
        config.model_output_dir = os.path.dirname(config.checkpoint_path)
    if not config.checkpoint_path:
        # Not trying to resume from a checkpoint
        pass
    elif not os.path.isfile(config.checkpoint_path):
        # Looks like we're trying to resume from the checkpoint that this job
        # will itself create. Let's assume this is to let the job resume upon
        # preemption, and it just hasn't been preempted yet.
        print(f"Skipping premature resumption from preemption: no checkpoint file found at '{config.checkpoint_path}'")
    else:
        print(f"Loading resumption checkpoint '{config.checkpoint_path}'", flush=True)
        # Map model parameters to be load to the specified gpu.
        checkpoint = torch.load(config.checkpoint_path, map_location=device)

    if checkpoint is None:
        # Our epochs go from 1 to n_epoch, inclusive
        start_epoch = 1
    else:
        # Continue from where we left off
        start_epoch = checkpoint["epoch"] + 1
        if config.seed is not None:
            # Make sure we don't get the same behaviour as we did on the
            # first epoch repeated on this resumed epoch.
            utils.set_rng_seeds_fixed(config.seed + start_epoch, all_gpu=False)

    # LOAD PRE-TRAINED CHECKPOINT =============================================
    # Map model parameters to be load to the specified gpu.
    pre_model, pre_checkpoint = load_pretrained_model(config.pretrained_checkpoint_path, device=device)
    # Override the classifier with an identity function as we only want the embeddings
    pre_model.classifier = nn.Identity()

    keys_to_reuse = [
        "k_mer",
        "stride",
        "max_len",
        "tokenizer",
        "use_unk_token",
        "n_layers",
        "n_heads",
    ]
    default_kwargs = vars(get_parser().parse_args(["--pretrained_checkpoint=dummy.pt"]))
    for key in keys_to_reuse:
        if not hasattr(config, key) or getattr(config, key) == getattr(pre_checkpoint["config"], key):
            pass
        elif getattr(config, key) == default_kwargs[key]:
            print(
                f"  Overriding default config value {key}={getattr(config, key)}"
                f" with {getattr(pre_checkpoint['config'], key)} from pretained checkpoint."
            )
        elif getattr(config, key) != getattr(pre_checkpoint["config"], key):
            raise ValueError(
                f"config value for {key} differs from pretrained checkpoint:"
                f" {getattr(config, key)} (ours) vs {getattr(pre_checkpoint['config'], key)} (pretrained checkpoint)"
            )
        setattr(config, key, getattr(pre_checkpoint["config"], key, None))

    config.pretrained_run_name = pre_checkpoint["config"].run_name
    config.pretrained_run_id = pre_checkpoint["config"].run_id

    # DATASET =================================================================

    if config.dataset_name != "BIOSCAN-1M":
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported.")

    if config.data_dir is None:
        config.data_dir = os.path.join(get_project_root(), "data")

    dataset_args = {
        "k_mer": config.k_mer,
        "stride": config.stride,
        "max_len": config.max_len,
        "tokenizer": config.tokenizer,
        "use_unk_token": config.use_unk_token,
    }
    dataset_train = DNADataset(
        file_path=os.path.join(config.data_dir, "supervised_train.csv"),
        randomize_offset=True,
        **dataset_args,
    )
    dataset_val = DNADataset(
        file_path=os.path.join(config.data_dir, "supervised_val.csv"),
        randomize_offset=False,
        **dataset_args,
    )
    dataset_test = DNADataset(
        file_path=os.path.join(config.data_dir, "supervised_test.csv"),
        randomize_offset=False,
        **dataset_args,
    )
    distinct_val_test = True
    eval_set = "Val" if distinct_val_test else "Test"

    # Dataloader --------------------------------------------------------------
    dl_train_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": True,
        "sampler": None,
        "shuffle": True,
        "worker_init_fn": utils.worker_seed_fn,
    }
    dl_val_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if config.cpu_workers is None:
        config.cpu_workers = utils.get_num_cpu_available()
    if use_cuda:
        cuda_kwargs = {"num_workers": config.cpu_workers, "pin_memory": True}
        dl_train_kwargs.update(cuda_kwargs)
        dl_val_kwargs.update(cuda_kwargs)
    dl_test_kwargs = copy.deepcopy(dl_val_kwargs)

    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_train_kwargs["sampler"] = DistributedSampler(
            dataset_train,
            shuffle=True,
            seed=config.seed if config.seed is not None else 0,
            drop_last=False,
        )
        dl_train_kwargs["shuffle"] = None
        dl_val_kwargs["sampler"] = DistributedSampler(dataset_val, shuffle=False, drop_last=False)
        dl_val_kwargs["shuffle"] = None
        dl_test_kwargs["sampler"] = DistributedSampler(dataset_test, shuffle=False, drop_last=False)
        dl_test_kwargs["shuffle"] = None

    dataloader_train = torch.utils.data.DataLoader(dataset_train, **dl_train_kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, **dl_val_kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **dl_test_kwargs)

    # MODEL ===================================================================

    model = ClassificationModel(pre_model, dataset_train.num_labels)

    # Mark frozen parameters
    if config.freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Configure model for distributed training --------------------------------
    print("\nModel architecture:")
    print(model, flush=True)
    print()

    if not use_cuda:
        print("Using CPU (this will be slow)", flush=True)
    elif config.distributed:
        # Convert batchnorm into SyncBN, using stats computed from all GPUs
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, the DistributedDataParallel
        # constructor should always set a single device scope, otherwise
        # DistributedDataParallel will use all available devices.
        model = model.to(device)
        torch.cuda.set_device(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank
        )
    else:
        if config.local_rank is not None:
            torch.cuda.set_device(config.local_rank)
        model = model.to(device)

    # OPTIMIZATION ============================================================
    # Optimizer ---------------------------------------------------------------
    # Set up the optimizer

    # Bigger batch sizes mean better estimates of the gradient, so we can use a
    # bigger learning rate. See https://arxiv.org/abs/1706.02677
    # Hence we scale the learning rate linearly with the total batch size.
    config.lr = config.lr_relative * config.batch_size / BASE_BATCH_SIZE

    # Select which parameters we want to optimize
    if config.freeze_encoder:
        model_deref = model.module if config.distributed else model
        params = model_deref.classifier.parameters()
    else:
        params = model.parameters()
    # Fetch the constructor of the appropriate optimizer from torch.optim
    optimizer = getattr(torch.optim, config.optimizer)(params, lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler ---------------------------------------------------------------
    # Set up the learning rate scheduler
    if config.scheduler.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [p["lr"] for p in optimizer.param_groups],
            epochs=config.epochs,
            steps_per_epoch=len(dataloader_train),
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not supported.")

    # Loss function -----------------------------------------------------------
    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    # LOGGING =================================================================
    # Setup logging and saving

    # If we're using wandb, initialize the run, or resume it if the job was preempted.
    if config.log_wandb and config.global_rank == 0:
        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
        EXCLUDED_WANDB_CONFIG_KEYS = [
            "log_wandb",
            "wandb_entity",
            "wandb_project",
            "global_rank",
            "local_rank",
            "run_name",
            "run_id",
            "model_output_dir",
        ]
        job_type = "linearprobe" if config.freeze_encoder else "finetune"
        wandb.init(
            name=wandb_run_name,
            id=config.run_id,
            resume="allow",
            group=config.pretrained_run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type=job_type,
            tags=["evaluate", job_type],
        )
        # If a run_id was not supplied at the command prompt, wandb will
        # generate a name. Let's use that as the run_name.
        if config.run_name is None:
            config.run_name = wandb.run.name
        if config.run_id is None:
            config.run_id = wandb.run.id

    # If we still don't have a run name, generate one from the current time.
    if config.run_name is None:
        config.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.run_id is None:
        config.run_id = utils.generate_id()

    # If no checkpoint path was supplied, automatically determine the path to
    # which we will save the model checkpoint.
    if not config.checkpoint_path and config.models_dir:
        config.model_output_dir = os.path.join(
            config.models_dir,
            config.dataset_name,
            f"{config.run_name}__{config.run_id}",
        )
        config.checkpoint_path = os.path.join(config.model_output_dir, "checkpoint_finetune.pt")
        if config.log_wandb and config.global_rank == 0:
            wandb.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

    if config.checkpoint_path is None:
        print("Model will not be saved.")
    else:
        print(f"Model will be saved to '{config.checkpoint_path}'")

    # RESUME ==================================================================
    # Now that everything is set up, we can load the state of the model,
    # optimizer, and scheduler from a checkpoint, if supplied.

    # Initialize step related variables as if we're starting from scratch.
    # Their values will be overridden by the checkpoint if we're resuming.
    total_step = 0
    n_samples_seen = 0

    best_stats = {"max_accuracy": 0, "best_epoch": 0}

    if checkpoint is not None:
        print(f"Loading state from checkpoint (epoch {checkpoint['epoch']})")
        total_step = checkpoint["total_step"]
        n_samples_seen = checkpoint["n_samples_seen"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_stats["max_accuracy"] = checkpoint.get("max_accuracy", 0)
        best_stats["best_epoch"] = checkpoint.get("best_epoch", 0)

    # TRAIN ===================================================================
    print()
    print("Configuration:")
    print()
    print(config, flush=True)
    print()

    # Ensure modules are on the correct device
    model = model.to(device)

    timing_stats = {}
    t_end_epoch = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        t_start_epoch = time.time()
        if config.seed is not None:
            # If the job is resumed from preemption, our RNG state is currently set the
            # same as it was at the start of the first epoch, not where it was when we
            # stopped training. This is not good as it means jobs which are resumed
            # don't do the same thing as they would be if they'd run uninterrupted
            # (making preempted jobs non-reproducible).
            # To address this, we reset the seed at the start of every epoch. Since jobs
            # can only save at the end of and resume at the start of an epoch, this
            # makes the training process reproducible. But we shouldn't use the same
            # RNG state for each epoch - instead we use the original seed to define the
            # series of seeds that we will use at the start of each epoch.
            epoch_seed = utils.determine_epoch_seed(config.seed, epoch=epoch)
            # We want each GPU to have a different seed to the others to avoid
            # correlated randomness between the workers on the same batch.
            # We offset the seed for this epoch by the GPU rank, so every GPU will get a
            # unique seed for the epoch. This means the job is only precisely
            # reproducible if it is rerun with the same number of GPUs (and the same
            # number of CPU workers for the dataloader).
            utils.set_rng_seeds_fixed(epoch_seed + config.global_rank, all_gpu=False)
            if isinstance(getattr(dataloader_train, "generator", None), torch.Generator):
                # Finesse the dataloader's RNG state, if it is not using the global state.
                dataloader_train.generator.manual_seed(epoch_seed + config.global_rank)
            if isinstance(getattr(dataloader_train.sampler, "generator", None), torch.Generator):
                # Finesse the sampler's RNG state, if it is not using the global RNG state.
                dataloader_train.sampler.generator.manual_seed(config.seed + epoch + 10000 * config.global_rank)

        if hasattr(dataloader_train.sampler, "set_epoch"):
            # Handling for DistributedSampler.
            # Set the epoch for the sampler so that it can shuffle the data
            # differently for each epoch, but synchronized across all GPUs.
            dataloader_train.sampler.set_epoch(epoch)

        # Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Note the number of samples seen before this epoch started, so we can
        # calculate the number of samples seen in this epoch.
        n_samples_seen_before = n_samples_seen
        # Run one epoch of training
        train_stats, total_step, n_samples_seen = train_one_epoch(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            dataloader=dataloader_train,
            device=device,
            epoch=epoch,
            n_epoch=config.epochs,
            total_step=total_step,
            n_samples_seen=n_samples_seen,
        )
        t_end_train = time.time()

        timing_stats["train"] = t_end_train - t_start_epoch
        n_epoch_samples = n_samples_seen - n_samples_seen_before
        train_stats["throughput"] = n_epoch_samples / timing_stats["train"]

        print(f"Training epoch {epoch}/{config.epochs} summary:")
        print(f"  Steps ..............{len(dataloader_train):8d}")
        print(f"  Samples ............{n_epoch_samples:8d}")
        if timing_stats["train"] > 172800:
            print(f"  Duration ...........{timing_stats['train']/86400:11.2f} days")
        elif timing_stats["train"] > 5400:
            print(f"  Duration ...........{timing_stats['train']/3600:11.2f} hours")
        elif timing_stats["train"] > 120:
            print(f"  Duration ...........{timing_stats['train']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['train']:11.2f} seconds")
        print(f"  Throughput .........{train_stats['throughput']:11.2f} samples/sec")
        print(f"  Loss ...............{train_stats['loss']:14.5f}")
        print(f"  Accuracy ...........{train_stats['accuracy']:11.2f} %")
        print(flush=True)

        # Validate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate on validation set
        t_start_val = time.time()

        eval_stats = evaluate(
            dataloader=dataloader_val,
            model=model,
            device=device,
            partition_name=eval_set,
            is_distributed=config.distributed,
        )
        t_end_val = time.time()
        timing_stats["val"] = t_end_val - t_start_val
        eval_stats["throughput"] = len(dataloader_val.dataset) / timing_stats["val"]

        # Check if this is the new best model
        if eval_stats["accuracy"] >= best_stats["max_accuracy"]:
            best_stats["max_accuracy"] = eval_stats["accuracy"]
            best_stats["best_epoch"] = epoch

        print(f"Evaluating epoch {epoch}/{config.epochs} summary:")
        if timing_stats["val"] > 172800:
            print(f"  Duration ...........{timing_stats['val']/86400:11.2f} days")
        elif timing_stats["val"] > 5400:
            print(f"  Duration ...........{timing_stats['val']/3600:11.2f} hours")
        elif timing_stats["val"] > 120:
            print(f"  Duration ...........{timing_stats['val']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['val']:11.2f} seconds")
        print(f"  Throughput .........{eval_stats['throughput']:11.2f} samples/sec")

        # Save model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        t_start_save = time.time()
        if config.model_output_dir and (not config.distributed or config.global_rank == 0):
            safe_save_model(
                {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
                config.checkpoint_path,
                config=config,
                epoch=epoch,
                total_step=total_step,
                n_samples_seen=n_samples_seen,
                bert_config=pre_checkpoint["bert_config"],
                **best_stats,
            )
            if config.save_best_model and best_stats["best_epoch"] == epoch:
                ckpt_path_best = os.path.join(config.model_output_dir, "best_finetune.pt")
                print(f"Copying model to {ckpt_path_best}")
                shutil.copyfile(config.checkpoint_path, ckpt_path_best)

        t_end_save = time.time()
        timing_stats["saving"] = t_end_save - t_start_save

        # Log to wandb ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Overall time won't include uploading to wandb, but there's nothing
        # we can do about that.
        timing_stats["overall"] = time.time() - t_end_epoch
        t_end_epoch = time.time()

        # Send training and eval stats for this epoch to wandb
        if config.log_wandb and config.global_rank == 0:
            wandb.log(
                {
                    "Training/stepwise/epoch": epoch,
                    "Training/stepwise/epoch_progress": epoch,
                    "Training/stepwise/n_samples_seen": n_samples_seen,
                    "Training/epochwise/epoch": epoch,
                    **{f"Training/epochwise/Train/{k}": v for k, v in train_stats.items()},
                    **{f"Training/epochwise/{eval_set}/{k}": v for k, v in eval_stats.items()},
                    **{f"Training/epochwise/duration/{k}": v for k, v in timing_stats.items()},
                },
                step=total_step,
            )
            # Record the wandb time as contributing to the next epoch
            timing_stats = {"wandb": time.time() - t_end_epoch}
        else:
            # Reset timing stats
            timing_stats = {}
        # Print with flush=True forces the output buffer to be printed immediately
        print(flush=True)

    if start_epoch > config.epochs:
        print("Training already completed!")
    else:
        print(f"Training complete! (Trained epochs {start_epoch} to {config.epochs})")
    print(
        f"Best {eval_set} accuracy was {best_stats['max_accuracy']:.2f}%,"
        f" seen at the end of epoch {best_stats['best_epoch']}"
    )

    # TEST ====================================================================
    print(f"\nEvaluating final model (epoch {config.epochs}) performance")
    # Evaluate on test set
    print("\nEvaluating final model on test set...", flush=True)
    eval_stats = evaluate(
        dataloader=dataloader_test,
        model=model,
        device=device,
        partition_name="Test",
        is_distributed=config.distributed,
    )
    # Send stats to wandb
    if config.log_wandb and config.global_rank == 0:
        wandb.log({**{f"Eval/Test/{k}": v for k, v in eval_stats.items()}}, step=total_step)

    if distinct_val_test:
        # Evaluate on validation set
        print(f"\nEvaluating final model on {eval_set} set...", flush=True)
        eval_stats = evaluate(
            dataloader=dataloader_val,
            model=model,
            device=device,
            partition_name=eval_set,
            is_distributed=config.distributed,
        )
        # Send stats to wandb
        if config.log_wandb and config.global_rank == 0:
            wandb.log(
                {**{f"Eval/{eval_set}/{k}": v for k, v in eval_stats.items()}},
                step=total_step,
            )

    # Create a copy of the train partition with evaluation transforms
    # and a dataloader using the evaluation configuration (don't drop last)
    print("\nEvaluating final model on train set under test conditions (no augmentation, dropout, etc)...", flush=True)
    dataset_train_eval = dataset_train
    dataset_train_eval.randomize_offset = False
    dl_train_eval_kwargs = copy.deepcopy(dl_test_kwargs)
    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_train_eval_kwargs["sampler"] = DistributedSampler(
            dataset_train_eval,
            shuffle=False,
            drop_last=False,
        )
        dl_train_eval_kwargs["shuffle"] = None
    dataloader_train_eval = torch.utils.data.DataLoader(dataset_train_eval, **dl_train_eval_kwargs)
    eval_stats = evaluate(
        dataloader=dataloader_train_eval,
        model=model,
        device=device,
        partition_name="Train",
        is_distributed=config.distributed,
    )
    # Send stats to wandb
    if config.log_wandb and config.global_rank == 0:
        wandb.log({**{f"Eval/Train/{k}": v for k, v in eval_stats.items()}}, step=total_step)


def train_one_epoch(
    config,
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    device="cuda",
    epoch=1,
    n_epoch=None,
    total_step=0,
    n_samples_seen=0,
):
    r"""
    Train the encoder and classifier for one epoch.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The global config object.
    model : torch.nn.Module
        The encoder/decoder network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    criterion : torch.nn.Module
        The loss function.
    dataloader : torch.utils.data.DataLoader
        A dataloader for the training set.
    device : str or torch.device, default="cuda"
        The device to use.
    epoch : int, default=1
        The current epoch number (indexed from 1).
    n_epoch : int, optional
        The total number of epochs scheduled to train for.
    total_step : int, default=0
        The total number of steps taken so far.
    n_samples_seen : int, default=0
        The total number of samples seen so far.

    Returns
    -------
    results: dict
        A dictionary containing the training performance for this epoch.
    total_step : int
        The total number of steps taken after this epoch.
    n_samples_seen : int
        The total number of samples seen after this epoch.
    """
    # Put the model in train mode
    model.train()

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    loss_epoch = 0
    acc_epoch = 0
    acc_kpt_epoch = 0
    acc_all_epoch = 0

    if config.print_interval is None:
        # Default to printing to console every time we log to wandb
        config.print_interval = config.log_interval

    t_end_batch = time.time()
    t_start_wandb = t_end_wandb = None
    for batch_idx, (sequences, y_true, att_mask) in enumerate(dataloader):
        t_start_batch = time.time()
        batch_size_this_gpu = sequences.shape[0]

        # Move training inputs and targets to the GPU
        sequences = sequences.to(device)
        y_true = y_true.to(device)
        att_mask = att_mask.to(device)

        # Forward pass --------------------------------------------------------
        t_start_forward = time.time()
        # N.B. To accurately time steps on GPU we need to use torch.cuda.Event
        ct_forward = torch.cuda.Event(enable_timing=True)
        ct_forward.record()
        # Perform the forward pass through the model
        out = model(sequences, mask=att_mask, labels=y_true)
        loss = out.loss

        # Backward pass -------------------------------------------------------
        # Reset gradients
        optimizer.zero_grad()
        # Now the backward pass
        ct_backward = torch.cuda.Event(enable_timing=True)
        ct_backward.record()
        loss.backward()

        # Update --------------------------------------------------------------
        # Use our optimizer to update the model parameters
        ct_optimizer = torch.cuda.Event(enable_timing=True)
        ct_optimizer.record()
        optimizer.step()

        # Step the scheduler each batch
        scheduler.step()

        # Increment training progress counters
        total_step += 1
        batch_size_all = batch_size_this_gpu * config.world_size
        n_samples_seen += batch_size_all

        # Logging -------------------------------------------------------------
        # Log details about training progress
        t_start_logging = time.time()
        ct_logging = torch.cuda.Event(enable_timing=True)
        ct_logging.record()

        # Update the total loss for the epoch
        loss_batch = loss.clone()
        if config.distributed:
            # Fetch results from other GPUs
            dist.reduce(loss_batch, 0, op=dist.ReduceOp.AVG)
        loss_batch = loss_batch.item()
        loss_epoch += loss_batch

        with torch.no_grad():
            y_pred = out.logits.argmax(dim=1)

        if epoch <= 1 and batch_idx == 0:
            # Debugging
            print("sequences.shape     =", sequences.shape)
            print("att_mask.shape      =", att_mask.shape)
            print("y_true.shape        =", y_true.shape)
            print("y_pred.shape        =", y_pred.shape)
            print("logits.shape        =", out.logits.shape)
            print("loss.shape          =", loss.shape)
            # Debugging intensifies
            print("sequences[0]     =", sequences[0])
            print("att_mask[0]      =", att_mask[0])
            print("y_true[0]        =", y_true[0])
            print("y_pred[0]        =", y_pred[0])
            print("logits[0]        =", out.logits[0])
            print("loss =", loss.detach().item())

        # Compute accuracy
        with torch.no_grad():
            is_correct = y_pred == y_true
            # Accuracy
            acc = is_correct.sum() / is_correct.numel()
            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(acc, 0, op=dist.ReduceOp.AVG)
            acc = 100.0 * acc.item()
            acc_epoch += acc

        # Log to console
        if batch_idx <= 2 or batch_idx % config.print_interval == 0 or batch_idx >= len(dataloader) - 1:
            print(
                f"Train Epoch:{epoch:3d}" + (f"/{n_epoch}" if n_epoch is not None else ""),
                " Step:{:6d}/{}".format(batch_idx + 1, len(dataloader)),
                " Loss:{:8.5f}".format(loss_batch),
                " Acc:{:6.2f}%".format(acc),
                " LR: {}".format(scheduler.get_last_lr()),
                flush=True,
            )

        # Log to wandb
        if config.log_wandb and config.global_rank == 0 and batch_idx % config.log_interval == 0:
            # Create a log dictionary to send to wandb
            # Epoch progress interpolates smoothly between epochs
            epoch_progress = epoch - 1 + (batch_idx + 1) / len(dataloader)
            # Throughput is the number of samples processed per second
            throughput = batch_size_all / (t_start_logging - t_end_batch)
            log_dict = {
                "Training/stepwise/epoch": epoch,
                "Training/stepwise/epoch_progress": epoch_progress,
                "Training/stepwise/n_samples_seen": n_samples_seen,
                "Training/stepwise/Train/throughput": throughput,
                "Training/stepwise/Train/loss": loss_batch,
                "Training/stepwise/Train/accuracy": acc,
            }
            # Track the learning rate of each parameter group
            for lr_idx in range(len(optimizer.param_groups)):
                if "name" in optimizer.param_groups[lr_idx]:
                    grp_name = optimizer.param_groups[lr_idx]["name"]
                elif len(optimizer.param_groups) == 1:
                    grp_name = ""
                else:
                    grp_name = f"grp{lr_idx}"
                if grp_name != "":
                    grp_name = f"-{grp_name}"
                grp_lr = optimizer.param_groups[lr_idx]["lr"]
                log_dict[f"Training/stepwise/lr{grp_name}"] = grp_lr
            # Synchronize ensures everything has finished running on each GPU
            torch.cuda.synchronize()
            # Record how long it took to do each step in the pipeline
            if t_start_wandb is not None:
                # Record how long it took to send to wandb last time
                log_dict["Training/stepwise/duration/wandb"] = t_end_wandb - t_start_wandb
            log_dict["Training/stepwise/duration/dataloader"] = t_start_batch - t_end_batch
            log_dict["Training/stepwise/duration/preamble"] = t_start_forward - t_start_batch
            log_dict["Training/stepwise/duration/forward"] = ct_forward.elapsed_time(ct_backward) / 1000
            log_dict["Training/stepwise/duration/backward"] = ct_backward.elapsed_time(ct_optimizer) / 1000
            log_dict["Training/stepwise/duration/optimizer"] = ct_optimizer.elapsed_time(ct_logging) / 1000
            log_dict["Training/stepwise/duration/overall"] = time.time() - t_end_batch
            t_start_wandb = time.time()
            log_dict["Training/stepwise/duration/logging"] = t_start_wandb - t_start_logging
            # Send to wandb
            wandb.log(log_dict, step=total_step)
            t_end_wandb = time.time()

        # Record the time when we finished this batch
        t_end_batch = time.time()

    results = {
        "loss": loss_epoch / (batch_idx + 1),
        "accuracy": acc_epoch / (batch_idx + 1),
        "accuracy_unmasked": acc_kpt_epoch / (batch_idx + 1),
        "accuracy_overall": acc_all_epoch / (batch_idx + 1),
    }
    return results, total_step, n_samples_seen


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import sys

    from barcodebert.pretraining import get_parser as get_pretraining_parser

    parser = get_pretraining_parser()

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser.prog = prog
    parser.description = "Fine-tune BarcodeBERT."

    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Input model")
    group.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to pretrained model checkpoint (required).",
    )
    group.add_argument(
        "--freeze-encoder",
        "--freeze_encoder",
        action="store_true",
    )
    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    return run(config)


if __name__ == "__main__":
    cli()
