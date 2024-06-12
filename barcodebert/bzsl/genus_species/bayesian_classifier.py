from typing import Optional

import numpy as np
import torch
from scipy.linalg import cholesky
from scipy.special import gammaln
from scipy.stats import mode
from sklearn.decomposition import PCA


def apply_pca(pca_dim, *x_data):
    # This method seems to most consistently reproduce similar results to matlab, rather than implementing PCA from
    # scratch, due to some subtleties as to how matlab computes eigenvectors/eigenvalues compared to numpy
    pca = PCA(n_components=pca_dim)
    pca.fit(x_data[0])
    return [pca.transform(x) for x in x_data]


def calculate_priors(data, labels):
    _, embedding_dim = data.shape
    unique_labels = np.unique(labels)
    nc = len(unique_labels)

    scatters = np.zeros((embedding_dim, embedding_dim, nc))
    class_means = np.zeros((nc, embedding_dim))

    for j in range(len(unique_labels)):
        class_data = data[labels == unique_labels[j]]
        scatters[:, :, j] = np.cov(class_data, rowvar=False)
        class_means[j, :] = np.mean(class_data, axis=0)

    scatter = np.mean(scatters, axis=2)
    mu_0 = np.mean(class_means, axis=0)

    return mu_0, scatter


def normalize_range(arr, axis=None):
    max_value = np.max(arr, axis=axis, keepdims=True)
    min_value = np.max(arr, axis=axis, keepdims=True)

    return (arr - min_value) / max_value


class BayesianClassifier:
    def __init__(
        self,
        model: str,
        k_0: float,
        k_1: float,
        m: int,
        s: float,
        mu_0: Optional[float] = None,
        scatter: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.k_0 = k_0
        self.k_1 = k_1
        self.m = m
        self.s = s
        self.mu_0 = mu_0
        self.scatter = scatter

    def __call__(
        self,
        x_train,
        y_train,
        x_test_unseen,
        y_test_unseen,
        x_test_seen,
        y_test_seen,
        genera,
        *,
        dna_embedding_size: Optional[int] = None,
        pca: Optional[int] = None,
        tuning: bool = False,
        num_iter: int = 1,
    ):
        if self.model == "OSBC_DIL":
            assert dna_embedding_size is not None
            dna_data = {
                "x_train": x_train[:, :dna_embedding_size],
                "y_train": y_train[:dna_embedding_size],
                "x_test_unseen": x_test_unseen[:, :dna_embedding_size],
                "y_test_unseen": y_test_unseen[:dna_embedding_size],
                "x_test_seen": x_test_seen[:, :dna_embedding_size],
                "y_test_seen": y_test_seen[:dna_embedding_size],
            }
            image_data = {
                "x_train": x_train[:, :dna_embedding_size],
                "y_train": y_train[:dna_embedding_size],
                "x_test_unseen": x_test_unseen[:, :dna_embedding_size],
                "y_test_unseen": y_test_unseen[:dna_embedding_size],
                "x_test_seen": x_test_seen[:, :dna_embedding_size],
                "y_test_seen": y_test_seen[:dna_embedding_size],
            }

            prob_seen_dna, prob_unseen_dna, class_id = self.classify(
                **dna_data, genera=genera, pca=pca, tuning=tuning, num_iter=num_iter, get_metrics=False
            )
            prob_seen_image, prob_unseen_image, class_id = self.classify(
                **image_data, genera=genera, pca=pca, tuning=tuning, num_iter=num_iter, get_metrics=False
            )

            # normalized summation of likelihoods
            prob_unseen_dna = normalize_range(prob_unseen_dna, axis=1)
            prob_seen_dna = normalize_range(prob_seen_dna, axis=1)
            prob_unseen_image = normalize_range(prob_unseen_image, axis=1)
            prob_seen_image = normalize_range(prob_seen_image, axis=1)

            prob_unseen = prob_unseen_dna + prob_unseen_image
            prob_seen = prob_seen_dna + prob_seen_image
            unseen_indices = np.argsort(prob_unseen, axis=1)[::-1]
            seen_indices = np.argsort(prob_seen, axis=1)[::-1]
            y_pred_unseen = class_id[unseen_indices]
            y_pred_seen = class_id[seen_indices]

            _, unseen_acc = self.evaluate(y_test_unseen, y_pred_unseen, genera, is_unseen=True)

            _, seen_acc = self.evaluate(y_test_seen, y_pred_seen, genera, is_unseen=False)

            harmonic_mean = 2 * unseen_acc * seen_acc / (unseen_acc + seen_acc) if unseen_acc + seen_acc > 0 else 0

            return seen_acc, unseen_acc, harmonic_mean

        else:
            return self.classify(
                x_train,
                y_train,
                x_test_unseen,
                y_test_unseen,
                x_test_seen,
                y_test_seen,
                genera,
                pca=pca,
                tuning=tuning,
                num_iter=num_iter,
                get_metrics=True,
            )

    def classify(
        self,
        x_train,
        y_train,
        x_test_unseen,
        y_test_unseen,
        x_test_seen,
        y_test_seen,
        genera,
        *,
        pca=None,
        tuning=False,
        num_iter=1,
        get_metrics: bool = True,
    ):
        embedding_dim = x_train.shape[1]
        if pca is not None:
            x_train, x_test_seen, x_test_unseen = apply_pca(pca, x_train, x_test_seen, x_test_unseen)
            embedding_dim = pca

        # the original implementation had an option to permute the embedding features into a different order when not
        # tuning, but it was disabled for some reason in a mysterious way (almost looks like a bug)
        embedding_order = [list(range(embedding_dim)) for _ in range(num_iter)]

        y_pred_unseen = np.zeros((x_test_unseen.shape[0], num_iter))
        y_pred_seen = np.zeros((x_test_seen.shape[0], num_iter))

        assert num_iter >= 1
        for iter in range(num_iter):
            x_train_iter = x_train[:, embedding_order[iter]]
            y_train_iter = y_train
            x_test_unseen_iter = x_test_unseen[:, embedding_order[iter]]
            x_test_seen_iter = x_test_seen[:, embedding_order[iter]]

            if tuning:
                assert self.mu_0 is not None
                assert self.scatter is not None
            else:
                self.mu_0, self.scatter = calculate_priors(x_train_iter, y_train_iter)
            psi = (self.m - embedding_dim - 1) * self.scatter / self.s

            sig_s, mu_s, v_s, class_id, sigmas = self.ppd_derivation(x_train_iter, y_train_iter, genera, psi)

            # inference
            y_pred_unseen[:, iter], prob_unseen = self.predict(x_test_unseen_iter, sig_s, mu_s, v_s, class_id)
            y_pred_seen[:, iter], prob_seen = self.predict(x_test_seen_iter, sig_s, mu_s, v_s, class_id)

        # performance calculation
        if get_metrics:
            y_pred_unseen_final = mode(y_pred_unseen, axis=1).mode
            _, unseen_acc = self.evaluate(y_test_unseen, y_pred_unseen_final, genera, is_unseen=True)

            y_pred_seen_final = mode(y_pred_seen, axis=1).mode
            _, seen_acc = self.evaluate(y_test_seen, y_pred_seen_final, genera, is_unseen=False)

            harmonic_mean = 2 * unseen_acc * seen_acc / (unseen_acc + seen_acc) if unseen_acc + seen_acc > 0 else 0

            return seen_acc, unseen_acc, harmonic_mean
        else:
            return prob_seen, prob_unseen, class_id

    def ppd_derivation(self, x_train: np.ndarray, y_train: np.ndarray, genera: np.ndarray, psi: np.ndarray):
        """
        Calculate the PPD (Posterior Predictive Distribution) for each seen and surrogate classes.

        :param x_train: Training data features.
        :param y_train: Training data labels.
        :param genera: Genus labels.
        :param psi: Initial covariance matrix.

        Returns:
        sig_s (ndarray): Class predictive covariances.
        mu_s (ndarray): Class predictive means.
        v_s (ndarray): Class predictive DoF.
        class_id (ndarray): Class IDs.
        sigmas (ndarray): List of covariance matrices.
        """
        assert self.mu_0 is not None

        seen_species = np.unique(y_train)
        seen_genera = genera[seen_species]
        unique_genera = np.unique(genera)
        num_classes = len(seen_species) + len(unique_genera)
        num_samples, embedding_dim = x_train.shape

        # initialize output params: derive for each class predictive cov, mean, and dof
        sig_s = np.zeros((embedding_dim, embedding_dim, num_classes))
        sigmas = np.zeros((embedding_dim, embedding_dim, num_classes))
        mu_s = np.zeros((num_classes, embedding_dim))
        v_s = np.zeros(num_classes, dtype=int)
        class_id = np.zeros(num_classes, dtype=int)

        class_index = 0

        # go through seen classes
        k_inv_sum = self.k_0 * self.k_1 / (self.k_0 + self.k_1)
        for species_idx in range(len(seen_species)):
            species_mask = y_train == seen_species[species_idx]
            x_species = x_train[species_mask, :]

            # current selected component stats: # points, mean, and scatter
            num_samples_per_species = np.sum(species_mask)
            s_species = (num_samples_per_species - 1) * np.cov(x_species, rowvar=False)
            mu_species = np.mean(x_species, axis=0)

            v_s[class_index] = num_samples_per_species + self.m - embedding_dim + 1
            mu_s[class_index, :] = (num_samples_per_species * mu_species + k_inv_sum * self.mu_0) / (
                num_samples_per_species + k_inv_sum
            )
            # eq 34 - wishart terms
            s_mu = ((num_samples_per_species * k_inv_sum) / (k_inv_sum + num_samples_per_species)) * np.dot(
                mu_species - self.mu_0, mu_species - self.mu_0
            )
            sig_s[:, :, class_index] = (psi + s_species + s_mu) / (
                (num_samples_per_species + k_inv_sum) * v_s[class_index] / (num_samples_per_species + k_inv_sum + 1)
            )
            class_id[class_index] = seen_species[species_idx]
            class_index += 1

        # surrogate class PPD
        # go through genera
        for genus_idx in range(len(unique_genera)):
            genus_mask = np.zeros(num_samples, dtype=bool)
            genus = unique_genera[genus_idx]
            species_in_genus = seen_species[seen_genera == genus]
            num_species = len(species_in_genus)

            if num_species >= 1:  # ignore if there are no species within genus
                for species_idx in range(num_species):
                    genus_mask[y_train == species_in_genus[species_idx]] = True
                y_genus = y_train[genus_mask]
                x_genus = x_train[genus_mask, :]

                # initialize component sufficient statistics
                x_kl = np.zeros((num_species, embedding_dim))  # component means
                s_kl = np.zeros((embedding_dim, embedding_dim, num_species))  # component scatter matrices
                kap = np.zeros(num_species)  # model specific
                nkl = np.zeros(num_species)  # data points in components

                # calculate sufficient statistics for each component in meta cluster
                for j in range(num_species):
                    species_mask = y_genus == species_in_genus[j]
                    nkl[j] = np.sum(species_mask)
                    kap[j] = nkl[j] * self.k_1 / (nkl[j] + self.k_1)
                    x_ij = x_genus[species_mask, :]
                    x_kl[j, :] = np.mean(x_ij, axis=0)
                    s_kl[:, :, j] = (nkl[j] - 1) * np.cov(x_ij, rowvar=False)

                # model specific parameters
                sum_kap = np.sum(kap)
                kaps = (sum_kap + self.k_0) * self.k_1 / (sum_kap + self.k_0 + self.k_1)
                sum_skl = np.sum(s_kl, axis=2)

                v_s[class_index] = np.sum(nkl) - num_species + self.m - embedding_dim + 1
                sigmas[:, :, class_index] = psi + sum_skl
                sig_s[:, :, class_index] = (psi + sum_skl) / ((kaps * v_s[class_index]) / (kaps + 1))
                mu_s[class_index, :] = (np.sum(x_kl * (kap[:, np.newaxis]), axis=0) + self.k_0 * self.mu_0) / (
                    sum_kap + self.k_0
                )
                class_id[class_index] = unique_genera[genus_idx]
                class_index += 1
            else:
                print(f"Missing genus: {genus_idx=}, {genus=}")

        return sig_s, mu_s, v_s, class_id, sigmas

    def _predict_numpy(self, x_test, sig_s, mu_s, v_s, class_id):
        num_classes, embedding_dim = mu_s.shape
        pi_const = (embedding_dim / 2) * np.log(np.pi)
        gl_pc = gammaln(np.arange(0.5, max(v_s) + embedding_dim + 0.5, 0.5))
        num_samples = x_test.shape[0]

        prob_mat = np.zeros((num_samples, num_classes))

        # calculating log student-t likelihood for numerical stability
        for j in range(num_classes):
            v = x_test - mu_s[j, :]  # center the data
            chsig = cholesky(sig_s[:, :, j], lower=False)
            tpar = (
                gl_pc[v_s[j] + embedding_dim]
                - (gl_pc[v_s[j]] + embedding_dim / 2 * np.log(v_s[j]) + pi_const)
                - np.sum(np.log(np.diag(chsig)))
            )

            # slower than solve_triangular but more accurate to mrdivide; results in 312s for classify()
            temp = np.matmul(v, np.linalg.pinv(chsig))

            norm2 = np.einsum("ij,ij->i", temp, temp)  # faster than np.sum(temp**2)

            prob_mat[:, j] = tpar - 0.5 * (v_s[j] + embedding_dim) * np.log(1 + 1 / v_s[j] * norm2)

        bb = np.argmax(prob_mat, axis=1)
        return class_id[bb], prob_mat

    def _predict_pytorch(self, x_test, sig_s, mu_s, v_s, class_id):
        num_classes, embedding_dim = mu_s.shape
        pi_const = (embedding_dim / 2) * np.log(np.pi)
        gl_pc = gammaln(np.arange(0.5, max(v_s) + embedding_dim + 0.5, 0.5))
        num_samples = x_test.shape[0]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        prob_mat = torch.zeros((num_samples, num_classes), device=device)

        # calculating log student-t likelihood for numerical stability
        class_id = torch.tensor(class_id, device=device)
        x_test = torch.tensor(x_test, device=device)
        mu_s = torch.tensor(mu_s, device=device)
        gl_pc = torch.tensor(gl_pc, device=device)
        v_s = torch.tensor(v_s, device=device)
        sig_s = torch.tensor(sig_s, device=device)
        for j in range(num_classes):
            v = x_test - mu_s[j, :]  # center the data
            chsig = torch.linalg.cholesky(sig_s[:, :, j]).mH
            tpar = (
                gl_pc[v_s[j] + embedding_dim]
                - (gl_pc[v_s[j]] + embedding_dim / 2 * torch.log(v_s[j]) + pi_const)
                - torch.sum(torch.log(torch.diag(chsig)))
            )

            temp = torch.linalg.lstsq(chsig.T, v.T)[0].T
            norm2 = torch.sum(temp * temp, dim=1)

            prob_mat[:, j] = tpar - 0.5 * (v_s[j] + embedding_dim) * torch.log(1 + 1 / v_s[j] * norm2)

        bb = torch.argmax(prob_mat, dim=1)
        return class_id[bb].cpu().numpy(), prob_mat.cpu().numpy()

    def predict(self, x_test, sig_s, mu_s, v_s, class_id):
        return self._predict_pytorch(x_test, sig_s, mu_s, v_s, class_id)

    def evaluate(self, y_true, y_pred, genera, is_unseen):
        if is_unseen:
            y_true = genera[y_true]

        y_true_classes = np.unique(y_true)
        num_classes = len(y_true_classes)
        per_class_acc = np.zeros(num_classes)

        for class_idx in range(num_classes):
            sample_indices = y_true == y_true_classes[class_idx]
            per_class_acc[class_idx] = np.sum(y_true[sample_indices] == y_pred[sample_indices]) / np.sum(sample_indices)

        acc = np.mean(per_class_acc)  # macro accuracy

        return per_class_acc, acc
