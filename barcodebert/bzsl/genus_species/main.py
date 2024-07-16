import argparse
import time
import traceback
from typing import Optional

import numpy as np
import torch

from barcodebert.bzsl.genus_species.bayesian_classifier import (
    BayesianClassifier,
    apply_pca,
    calculate_priors,
)
from barcodebert.bzsl.genus_species.dataset import get_data_splits, load_data


def normalize(embeddings: np.ndarray):
    return (embeddings - np.mean(embeddings, axis=1, keepdims=True)) / np.std(embeddings, axis=1, ddof=1, keepdims=True)


def ridge_regression(embeddings_dna: np.ndarray, embeddings_img: np.ndarray, rho: int):
    image_dim = embeddings_img.shape[1]

    DXT = np.matmul(embeddings_dna.T, embeddings_img)  # dim: [dna_embedding_dim, img_embedding_dim]
    denom = np.matmul(embeddings_img.T, embeddings_img) + rho * np.identity(image_dim)
    # NOTE: this does not seem to be giving a value equivalent to mrdivide in matlab. If you fix this, then you can
    # reproduce the original results to at least 4 decimal points.
    params = np.matmul(DXT, np.linalg.pinv(denom))
    return params


def load_tuned_params(
    model: str,
    k_0: Optional[float] = None,
    k_1: Optional[float] = None,
    m: Optional[int] = None,
    s: Optional[float] = None,
    pca_dim: int = 500,
) -> tuple[float, float, int, float, int]:
    if model == "OSBC_IMG":
        params = [0.1, 10, 5 * pca_dim, 1, pca_dim]
    elif model == "OSBC_DNA":
        params = [0.1, 10, 25 * pca_dim, 0.5, pca_dim]
    elif model == "OSBC_DIL":
        params = [0.1, 10, 5 * pca_dim, 0.5, pca_dim]
    elif model == "OSBC_DIT":
        params = [0.1, 10, 25 * pca_dim, 0.5, pca_dim]
    else:
        params = [None, None, None, None, pca_dim]

    if k_0 is not None:
        params[0] = k_0
    if k_1 is not None:
        params[1] = k_1
    if m is not None:
        params[2] = m
    if s is not None:
        params[3] = s

    assert all(p is not None for p in params), f"expected all params to be specified, but found {params}"

    return tuple(params)


def tune_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test_unseen: np.ndarray,
    y_test_unseen: np.ndarray,
    x_test_seen: np.ndarray,
    y_test_seen: np.ndarray,
    genera: np.ndarray,
    model: str,
    pca_dim: Optional[int] = None,
):
    if model == "OSBC_DIC":
        # in order to add support, we would either need to tune parameters separately for each model or tune the same
        # set of parameters for both models. The former could be achieved by tuning OSBC_DNA and OSBC_IMG, but the
        # latter would double the tuning time and so would be computationally burdensome.
        raise NotImplementedError("Hyperparameter tuning is not yet supported for DIC.")

    if pca_dim is None:
        pca_dim = x_train.shape[1]

    # tuning range
    k0_range = [0.1, 10]
    k1_range = [1, 10]
    s_range = [0.5, 1, 5]
    m_range = [scalar * pca_dim for scalar in [2, 5, 10, 25, 100]]

    best_harmonic_mean = 0
    best_k_0 = None
    best_k_1 = None
    best_m = None
    best_s = None

    # apply pca
    x_train, x_test_seen, x_test_unseen = apply_pca(pca_dim, x_train, x_test_seen, x_test_unseen)

    # precalculation of class means and scatter matrices for unconstrained model
    mu_0, scatter = calculate_priors(x_train, y_train)

    print("Starting tuning...")
    for k_0 in k0_range:
        for k_1 in k1_range:
            for m in m_range:
                for s in s_range:
                    try:
                        bcls = BayesianClassifier(k_0, k_1, m, s, mu_0=mu_0, scatter=scatter)
                        seen_acc, unseen_acc, harmonic_mean = bcls.classify(
                            x_train,
                            y_train,
                            x_test_unseen,
                            y_test_unseen,
                            x_test_seen,
                            y_test_seen,
                            genera,
                            tuning=True,
                        )

                        print(f"Results from {k_0=:.2f}, {k_1=:.2f}, {m=}, {s=:.1f}:")
                        print(
                            f"Model {model} results on dataset: {seen_acc=:.2f}, {unseen_acc=:.2f}, {harmonic_mean=:.2f}"
                        )

                        if harmonic_mean > best_harmonic_mean:
                            print(f"  New best result: {harmonic_mean=}")
                            best_harmonic_mean = harmonic_mean
                            best_k_0 = k_0
                            best_k_1 = k_1
                            best_m = m
                            best_s = s
                    except torch.linalg.LinAlgError:
                        traceback.print_exc()

    print("-----------")
    print(f"Best parameters: k_0={best_k_0}, k_1={best_k_1}, m={best_m}, s={best_s}")

    return best_k_0, best_k_1, best_m, best_s, pca_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="path to folder containing data and splits files")
    parser.add_argument(
        "--model",
        default="OSBC_DNA",
        choices=["OSBC_DNA", "OSBC_IMG", "OSBC_DIC", "OSBC_DIL", "OSBC_DIT"],
        help="name of model to use for combining ",
    )
    parser.add_argument("--tuning", default=False, action="store_true")
    parser.add_argument("--embeddings", default=None, type=str)
    parser.add_argument("--rho", type=int, default=1, help="rho for transductive approach")
    parser.add_argument("--pca", type=int, default=500, help="dimension of PCA for image embedding")
    parser.add_argument(
        "--k0",
        dest="k_0",
        default=None,
        type=float,
        help="scaling constant for dispersion of centers of metaclasses around mu_0",
    )
    parser.add_argument(
        "--k1",
        dest="k_1",
        default=None,
        type=float,
        help="scaling constant for dispersion of actual class means around corresponding metaclass means",
    )
    parser.add_argument(
        "-m",
        dest="m",
        default=None,
        type=int,
        help="defines dimension of Wishart distribution for sampling covariance matrices of metaclasses",
    )
    parser.add_argument("-s", dest="s", default=None, type=float, help="scalar for mean of class covariances")
    parser.add_argument(
        "--output", default=None, type=str, dest="output", help="path to save final results after tuning"
    )
    args = parser.parse_args()

    # parse data
    print(f"Loading data from {args.datapath}")
    data, splits = load_data(args.datapath)
    if args.embeddings:
        print(f"Loading embeddings from {args.embeddings}")
        embeddings_dna = np.genfromtxt(args.embeddings, delimiter=",")
    else:  # use default embeddings from paper
        embeddings_dna = data["embeddings_dna"]
    embeddings_img = data["embeddings_img"]
    genera = data["G"].flatten() - 1
    labels = data["labels"]
    model = args.model

    if model == "OSBC_DIT":  # transductive
        print("Learning map for ridge regression...")

        st = [splits["trainval_loc"], splits["test_unseen_loc"], splits["test_seen_loc"]]

        # normalize vectors
        embeddings_dna = normalize(embeddings_dna)
        embeddings_img = normalize(embeddings_img)

        # ridge regression
        ridge_mapping = ridge_regression(embeddings_dna, embeddings_img, args.rho)

    if args.tuning:
        print("Starting hyperparameter tuning for k_0, k_1, m, and s...")
        x_train, y_train, x_test_unseen, y_test_unseen, x_test_seen, y_test_seen, x_train_img = get_data_splits(
            embeddings_dna, embeddings_img, labels, splits, args.tuning, model
        )
        if model == "OSBC_DIT":  # transductive
            x_tr_g = np.matmul(ridge_mapping, x_train_img.T)
            x_train = np.concatenate((x_train, x_tr_g.T), axis=0)
            y_train = np.concatenate((y_train, y_train), axis=0)

        k_0, k_1, m, s, pca_dim = tune_hyperparameters(
            x_train, y_train, x_test_unseen, y_test_unseen, x_test_seen, y_test_seen, genera, model, args.pca
        )
    else:
        k_0, k_1, m, s, pca_dim = load_tuned_params(
            model, k_0=args.k_0, k_1=args.k_1, m=args.m, s=args.s, pca_dim=args.pca
        )

    print("Running inference for selected hyperparameters...")
    x_train, y_train, x_test_unseen, y_test_unseen, x_test_seen, y_test_seen, x_train_img = get_data_splits(
        embeddings_dna, embeddings_img, labels, splits, False, model
    )

    # data augmentation from transductive method
    if model == "OSBC_DIT":  # transductive
        x_tr_g = np.matmul(ridge_mapping, x_train_img.T)
        x_train = np.concatenate((x_train, x_tr_g.T), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)

    # model training and inference
    bcls = BayesianClassifier(model, k_0, k_1, m, s)
    start_time = time.time()
    seen_acc, unseen_acc, harmonic_mean = bcls.classify(
        x_train, y_train, x_test_unseen, y_test_unseen, x_test_seen, y_test_seen, genera, pca=pca_dim
    )
    end_time = time.time()

    print(f"Results from {k_0=:.2f}, {k_1=:.2f}, {m=}, {s=:.1f}:")
    print(f"Model {model} results on dataset: {seen_acc=:.4f}, {unseen_acc=:.4f}, {harmonic_mean=:.4f}")
    print(f"Inference runtime: {end_time - start_time:.2f} seconds")
