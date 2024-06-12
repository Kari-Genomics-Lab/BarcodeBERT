"""
Evaluation routines.
"""

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F

from . import utils


def evaluate(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
    is_distributed=False,
):
    r"""
    Evaluate model performance on a dataset.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset to evaluate on.
    model : torch.nn.Module
        Model to evaluate.
    device : torch.device
        Device to run the model on.
    partition_name : str, default="Val"
        Name of the partition being evaluated.
    verbosity : int, default=1
        Verbosity level.
    is_distributed : bool, default=False
        Whether the model is distributed across multiple GPUs.

    Returns
    -------
    results : dict
        Dictionary of evaluation results.
    """
    model.eval()

    y_true_all = []
    y_pred_all = []
    xent_all = []

    for sequences, y_true, attn_mask in dataloader:
        sequences = sequences.to(device)
        y_true = y_true.to(device)
        attn_mask = attn_mask.to(device)
        with torch.no_grad():
            logits = model(sequences, mask=attn_mask, labels=y_true).logits
            xent = F.cross_entropy(logits, y_true, reduction="none")
            y_pred = torch.argmax(logits, dim=-1)

        if is_distributed:
            # Fetch results from other GPUs
            xent = utils.concat_all_gather(xent)
            y_true = utils.concat_all_gather(y_true)
            y_pred = utils.concat_all_gather(y_pred)

        xent_all.append(xent.cpu().numpy())
        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())

    # Concatenate the targets and predictions from each batch
    xent = np.concatenate(xent_all)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    # If the dataset size was not evenly divisible by the world size,
    # DistributedSampler will pad the end of the list of samples
    # with some repetitions. We need to trim these off.
    n_samples = len(dataloader.dataset)
    xent = xent[:n_samples]
    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]
    # Create results dictionary
    results = {}
    results["count"] = len(y_true)
    results["cross-entropy"] = np.mean(xent)
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_true, y_pred)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    # Could expand to other metrics too

    if verbosity >= 1:
        print(f"\n{partition_name} evaluation results:")
        for k, v in results.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif "entropy" in k:
                print(f"  {k + ' ':.<24s} {v:9.5f} nat")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")

    return results
