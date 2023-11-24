import os
from typing import Any

import numpy as np
import scipy.io as sio


def load_data(
    data_path: str, data_filename: str = "data.mat", splits_filename: str = "splits.mat"
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = sio.loadmat(os.path.join(data_path, data_filename))
    splits = sio.loadmat(os.path.join(data_path, splits_filename))

    return data, splits


def get_data_splits(embeddings_dna, embeddings_img, labels, splits, is_tuning, model):
    if model == "OSBC_IMG":
        features = embeddings_img
    elif model in {"OSBC_DIC", "OSBC_DIL"}:
        features = np.concatenate((embeddings_dna, embeddings_img), axis=1)
    else:
        features = embeddings_dna

    # note: we subtract 1 here because all of the indices are 1-indexed (because MATLAB)
    if is_tuning:
        train_idx = splits["train_loc"].flatten() - 1
        test_seen_idx = splits["val_seen_loc"].flatten() - 1
        test_unseen_idx = splits["val_unseen_loc"].flatten() - 1
    else:
        train_idx = splits["trainval_loc"].flatten() - 1
        test_seen_idx = splits["test_seen_loc"].flatten() - 1
        test_unseen_idx = splits["test_unseen_loc"].flatten() - 1

    labels -= 1  # 0-index class labels
    x_train = features[train_idx, :]
    x_train_img = embeddings_img[train_idx, :]
    y_train = labels[train_idx].flatten()
    x_test_seen = features[test_seen_idx, :]
    y_test_seen = labels[test_seen_idx].flatten()
    x_test_unseen = features[test_unseen_idx, :]
    y_test_unseen = labels[test_unseen_idx].flatten()

    return x_train, y_train, x_test_unseen, y_test_unseen, x_test_seen, y_test_seen, x_train_img
