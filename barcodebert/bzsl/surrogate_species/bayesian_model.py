import json
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import gammaln

from barcodebert.bzsl.surrogate_species.utils import (
    DataLoader,
    apply_pca,
    perf_calc_acc,
)


class Model:
    """Bayesian model for species classification"""

    def __init__(self, opt):
        super().__init__()

        self.datapath = opt.datapath
        self.dataset = opt.dataset
        self.side_info = opt.side_info
        self.pca_dim = opt.pca_dim
        self.tuning = opt.tuning
        self.alignment = opt.alignment
        self.embeddings = opt.embeddings
        self.output = opt.output
        self.use_genus = opt.genus

        if opt.m and opt.m % self.pca_dim != 0:
            raise ValueError(f"m should be a multiple of the PCA dimension ({self.pca_dim}), but got {self.m} instead.")

        self.k_0 = opt.k_0
        self.k_1 = opt.k_1
        self.m = opt.m
        self.s = opt.s
        self.K = opt.K

    # Claculating class mean and covariance priors
    def calculate_priors(self, xtrain, ytrain, model_v="unconstrained"):
        dim = xtrain.shape[1]
        uy = np.unique(ytrain)
        nc = len(uy)
        if model_v == "constrained":
            class_means = np.zeros((nc, dim))
            for j in range(nc):
                idd = np.in1d(ytrain, uy[j])
                class_means[j] = np.mean(xtrain[idd], axis=0).T
            mu_0 = np.mean(class_means, axis=0)
            Sigma_0 = 0
        else:
            Sigma_0, mu_0 = 0, 0
            for j in range(nc):
                idd = np.in1d(ytrain, uy[j])
                Sigma_0 += np.cov(xtrain[idd].T)
                mu_0 += np.mean(xtrain[idd], axis=0)
            Sigma_0 /= nc
            mu_0 /= nc

        return mu_0, Sigma_0

    # Check for tie, if yes, change the last similar class with the next one untill tie broken
    def check_for_tie(self, curr_unseen_class, usclass, seenclasses, curr_classes, s_in, K):
        flag = True
        ect = 0
        while flag:
            flag = False
            for arr in usclass.values():
                if set(curr_classes) == set(arr):
                    flag = True
                    curr_classes[-1] = seenclasses[s_in[K + ect]]
                    ect += 1
                    break
        usclass[curr_unseen_class] = curr_classes

        return curr_classes, usclass

    # Calculating Posterior Predictive Distribution parameters
    def calculate_ppd_params(self, xtrain, ytrain, att_seen, att_unseen, unseenclasses, K, Psi, mu0, m, k0, k1):
        seenclasses = np.unique(ytrain)
        nc = len(seenclasses) + len(unseenclasses)
        n, d = xtrain.shape

        Sig_s = np.zeros((d, d, nc))
        Sigmas = np.zeros((d, d, nc))
        mu_s = np.zeros((nc, d))
        v_s = np.zeros((nc, 1), dtype=np.int32)
        class_id = np.zeros((nc, 1))
        usclass_list = {}

        # Start with the unseen classes
        uy = unseenclasses
        ncl = len(uy)
        cnt = 0
        # Main for loop for  unseen classes params estimation
        for i in range(ncl):
            # Calculating Euclidean distance between the selected unseen class
            # attributes and all seen classes
            tmp = att_unseen[i, np.newaxis]
            D = cdist(att_seen, tmp)
            s_in = np.argsort(D.ravel())

            # Choose the K nearest neighbor to form surrogate class local prior
            classes = seenclasses[s_in[:K]]
            classes, usclass_list = self.check_for_tie(uy[i], usclass_list, seenclasses, classes, s_in, K)

            # Extract corresponding data
            idx = np.in1d(ytrain, classes)
            Yi = ytrain[idx]
            Xi = xtrain[idx]
            uyi = np.unique(Yi)

            # Initialize component sufficient statistics
            ncpi = len(uyi)
            xkl = np.zeros((ncpi, d))  # Component means
            Skl = np.zeros((d, d, ncpi))  # Component scatter matrices
            kap = np.zeros((ncpi, 1))  # model specific
            nkl = np.zeros((ncpi, 1))  # number of data points in the components

            # Calculate  sufficient statistics for each component in meta cluster
            for j in range(ncpi):
                idx = np.in1d(Yi, uyi[j])
                nkl[j] = np.sum(idx)
                kap[j] = nkl[j] * k1 / (nkl[j] + k1)
                Xij = Xi[idx]
                xkl[j] = np.mean(Xij, axis=0)
                Skl[:, :, j] = (nkl[j] - 1) * np.cov(Xij.T)

                # Model specific parameters
            sumkap = np.sum(kap)
            kaps = (sumkap + k0) * k1 / (sumkap + k0 + k1)
            sumSkl = np.sum(Skl, axis=2)
            muk = (np.sum(np.multiply(xkl, kap * np.ones((1, d))), axis=0) + k0 * mu0) / (sumkap + k0)
            # Unseen classes' predictive cov, mean and dof
            v_s[cnt] = np.sum(nkl) - ncpi + m - d + 1
            class_id[cnt] = uy[i]
            Sigmas[:, :, cnt] = Psi + sumSkl
            Sig_s[:, :, cnt] = (Psi + sumSkl) / ((kaps * v_s[cnt]) / (kaps + 1))
            mu_s[cnt] = muk
            cnt += 1

            # The second part: same procedure for Seen classes
        uy = seenclasses
        ncl = len(uy)

        for i in range(ncl):
            idx = np.in1d(ytrain, uy[i])
            Xi = xtrain[idx]

            # The current selected component stats: # points, mean and scatter
            cur_n = np.sum(idx)
            cur_S = (cur_n - 1) * np.cov(Xi.T)
            cur_mu = np.mean(Xi, axis=0, keepdims=True)

            # Selected seen class attribute distance to all other seen class attr
            tmp = att_seen[i, np.newaxis]
            D = cdist(att_seen, tmp)
            s_in = np.argsort(D.ravel())

            # neighborhood radius
            classes = seenclasses[s_in[1 : K + 1]]

            # !!! As shown in the PPD derivation of Supplementary material model
            # supports forming surrogate classes for seen classes as well but we
            # did not utilized local priors for seen classes in this work. We
            # just used data likelihood and global prior for seen class formation
            # as mentioned in the main text !!! Thus nci is set to 0 instead of len(classes)
            nci = 0
            if nci > 0:
                idx = np.in1d(ytrain, classes)
                Yi = ytrain[idx]
                Xi = xtrain[idx]
                uyi = classes

                # data and initialization
                ncpi = len(uyi)
                xkl = np.zeros((ncpi, d))
                Skl = np.zeros((d, d, ncpi))
                kap = np.zeros((ncpi, 1))
                nkl = np.zeros((ncpi, 1))

                # sufficient stats calculation
                for j in range(ncpi):
                    idx = np.in1d(Yi, uyi[j])
                    nkl[j] = np.sum(idx)
                    kap[j] = nkl[j] * k1 / (nkl[j] + k1)
                    Xij = Xi[idx]  # Data points in component j and meta cluster i
                    xkl[j] = np.mean(Xij, axis=0, keepsdim=True)
                    Skl[:, :, j] = (nkl[j] - 1) * np.cov(Xij.T)

                sumkap = np.sum(kap)
                kaps = (sumkap + k0) * k1 / (sumkap + k0 + k1)
                sumSkl = np.sum(Skl, axis=2)
                muk = (np.sum(np.multiply(xkl, kap * np.ones((1, d))), axis=0) + k0 * mu0) / (sumkap + k0)
                vsc = np.sum(nkl) - ncpi + m - d + 1

                v_s[cnt] = vsc + cur_n
                Smu = ((cur_n * kaps) / (kaps + cur_n)) * np.dot(cur_mu - muk, (cur_mu - muk).T)
                Sigmas[:, :, cnt] = Psi + sumSkl + cur_S + Smu  # Just need for exp of rebuttal, then delete
                Sig_s[:, :, cnt] = (Psi + sumSkl + cur_S + Smu) / (((cur_n + kaps) * v_s[cnt]) / (cur_n + kaps + 1))
                mu_s[cnt] = (cur_n * cur_mu + kaps * muk) / (cur_n + kaps)
                class_id[cnt] = uy[i]
                cnt += 1

            # The case where only data likelihood and global priors are used and local priors are ignored. This
            # is the case we used for seen classes as mentioned in the paper
            else:
                v_s[cnt] = cur_n + m - d + 1
                mu_s[cnt] = (cur_n * cur_mu + (k0 * k1 / (k0 + k1)) * mu0) / (cur_n + (k0 * k1 / (k0 + k1)))
                Smu = ((cur_n * (k0 * k1 / (k0 + k1))) / ((k0 * k1 / (k0 + k1)) + cur_n)) * np.dot(
                    cur_mu - mu0, (cur_mu - mu0).T
                )
                Sig_s[:, :, cnt] = (Psi + cur_S + Smu) / (
                    ((cur_n + (k0 * k1 / (k0 + k1))) * v_s[cnt]) / (cur_n + (k0 * k1 / (k0 + k1)) + 1)
                )
                class_id[cnt] = uy[i]
                cnt += 1

        return Sig_s, mu_s, v_s, class_id, Sigmas

        # PPD calculation (Log-Likelihood of Student-t)

    def bayesian_cls_evaluate(self, X, Sig_s, mu_s, v_s, class_id):
        # Initialization
        ncl, d = mu_s.shape
        piconst = (d / 2) * np.log(np.pi)
        gl_pc = gammaln(np.arange(0.5, np.max(v_s) + d + 0.5, 0.5))
        n = X.shape[0]
        lkh = np.zeros((n, ncl))

        # Calculating log student-t likelihood
        for j in range(ncl):
            v = X - mu_s[j]  # Center the data
            k = 0
            Imat = np.eye(Sig_s[:, :, j].shape[0])
            while True:
                try:
                    chsig = np.linalg.cholesky(Sig_s[:, :, j])
                    break
                except np.linalg.LinAlgError:
                    # Find the nearest positive definite matrix for M.
                    # Referenced from: https://github.com/Cysu/open-reid/commit/61f9c4a4da95d0afc3634180eee3b65e38c54a14
                    k += 1
                    w, v_v = np.linalg.eig(Sig_s[:, :, j])
                    min_eig = v_v.min().astype(np.float)
                    Sig_s[:, :, j] += (-min_eig * k * k + np.spacing(min_eig)) * Imat

            # chsig = np.linalg.cholesky(Sig_s[:, :, j])  # Cholesky decomposition
            tpar = (
                gl_pc[v_s[j] + d - 1]
                - (gl_pc[v_s[j] - 1] + (d / 2) * np.log(v_s[j]) + piconst)
                - np.sum(np.log(chsig.diagonal()))
            )  # Stu-t lik part 1
            temp = solve_triangular(chsig, v.T, overwrite_b=True, check_finite=False, lower=True).T  # mrdivide(v,chsig)
            norm2 = np.einsum("ij,ij->i", temp, temp)  # faster than np.sum(temp**2)

            lkh[:, j] = tpar - 0.5 * (v_s[j] + d) * np.log(1 + (1 / v_s[j]) * norm2)

        bb = np.argmax(lkh, axis=1)
        ypred = class_id[bb]  # To ensure labels are correctly assigned back to original ones

        return ypred, lkh

    def bayesian_cls_train(
        self,
        x_tr,
        y_tr,
        unseenclasses,
        att,
        k_0=0.1,
        k_1=10,
        m=5 * 500,
        mu_0=0,
        s=1,
        Sigma_0=0,
        K=2,
        pca_dim=0,
        tuning=False,
    ):
        s_classes = np.unique(y_tr)
        us_classes = unseenclasses
        # Attributes of seen and unseen classes
        att_unseen = att[:, us_classes].T
        att_seen = att[:, s_classes].T
        d0 = x_tr.shape[1]

        if tuning:
            Psi = (m - d0 - 1) * Sigma_0 / s
        else:
            [mu_0, Sigma_0] = self.calculate_priors(x_tr, y_tr)
            Psi = (m - d0 - 1) * Sigma_0 / s

        # Class predictive cov, mean and DoF from unconstrained model
        print("PPD derivation is Done!!")
        return self.calculate_ppd_params(x_tr, y_tr, att_seen, att_unseen, us_classes, K, Psi, mu_0, m, k_0, k_1)

    def hyperparameter_tuning(self, constrained=False):
        # Default # features for PCA id Unconstrained model selected

        dataloader = DataLoader(
            self.datapath, self.dataset, self.side_info, self.tuning, self.alignment, self.embeddings, self.use_genus
        )

        # load attribute

        att, _, _, _, _, _ = dataloader.load_tuned_params()

        xtrain, ytrain, xtest_seen, ytest_seen, xtest_unseen, ytest_unseen = dataloader.data_split()
        dim = 500

        if self.pca_dim is not None:
            dim = self.pca_dim

        k0_range = [0.1, 1]
        k1_range = [10, 25]
        # a0_range = [1, 10, 100]
        s_range = [1, 5, 10]
        K_range = [1, 2, 3]

        bestH = 0
        best_k0 = None
        best_k1 = None
        best_m = None
        best_s = None
        best_K = None

        if not constrained:
            print("Applying PCA to reduce the dimension...")
            xtrain, xtest_seen, xtest_unseen = apply_pca(xtrain, xtest_seen, xtest_unseen, self.pca_dim)
            [mu_0, Sigma_0] = self.calculate_priors(xtrain, ytrain, model_v="unconstrained")
            m_range = [5 * dim, 25 * dim, 100 * dim, 500 * dim]
            print("Tuning is getting started...")
            try:
                for kk in K_range:
                    for k_0 in k0_range:
                        for k_1 in k1_range:
                            for m in m_range:
                                for ss in s_range:
                                    time_s = time.time()
                                    Sig_s, mu_s, v_s, class_id, _ = self.bayesian_cls_train(
                                        xtrain,
                                        ytrain,
                                        dataloader.unseenclasses,
                                        att,
                                        k_0=k_0,
                                        k_1=k_1,
                                        m=m,
                                        s=ss,
                                        K=kk,
                                        mu_0=mu_0,
                                        Sigma_0=Sigma_0,
                                        pca_dim=self.pca_dim,
                                        tuning=True,
                                    )

                                    # Prediction phase
                                    ypred_unseen, _ = self.bayesian_cls_evaluate(
                                        xtest_unseen, Sig_s, mu_s, v_s, class_id
                                    )
                                    ypred_seen, _ = self.bayesian_cls_evaluate(xtest_seen, Sig_s, mu_s, v_s, class_id)

                                    _, _, gzsl_seen_acc, gzsl_unseen_acc, H = perf_calc_acc(
                                        ytest_seen, ytest_unseen, ypred_seen, ypred_unseen, dataloader.label_to_genus
                                    )
                                    print(
                                        "\nCurrent parameters k0=%.2f, k1=%.2f, m=%d, s=%.1f, K=%d on dataset:"
                                        % (k_0, k_1, m, ss, kk)
                                    )
                                    print()
                                    if H > bestH:
                                        bestH = H
                                        best_k0 = k_0
                                        best_k1 = k_1
                                        best_m = m
                                        best_s = ss
                                        best_K = kk
                                        print(
                                            "\nResults from k0=%.2f, k1=%.2f, m=%d, s=%.1f, K=%d on dataset:"
                                            % (k_0, k_1, m, ss, kk)
                                        )
                                        print(
                                            "BSeen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%%\n"
                                            % (gzsl_seen_acc * 100, gzsl_unseen_acc * 100, H * 100)
                                        )
                                    time_e = time.time()
                                    print("total cost: " + str(time_e - time_s))
            except KeyboardInterrupt:
                print("Grid search interrupted. Returning best results found up to this point.")
        else:
            # TODO: hyper-parameter tuning for constrained model, ignored for now.
            pass

        return att, best_k0, best_k1, best_m, best_s, best_K

    def train_and_eval(self):
        """
        Tuning process and optimal parameters
        """

        if self.tuning:
            att, k_0, k_1, m, s, K = self.hyperparameter_tuning(constrained=False)
            dataloader = DataLoader(
                self.datapath,
                self.dataset,
                self.side_info,
                False,
                alignment=self.alignment,
                embeddings=self.embeddings,
                use_genus=self.use_genus,
            )
        else:
            dataloader = DataLoader(
                self.datapath,
                self.dataset,
                self.side_info,
                False,
                alignment=self.alignment,
                embeddings=self.embeddings,
                use_genus=self.use_genus,
            )
            att, k_0, k_1, m, s, K = dataloader.load_tuned_params()
            if self.k_0 is not None:
                k_0 = self.k_0
            if self.k_1 is not None:
                k_1 = self.k_1
            if self.m is not None:
                m = self.m
            if self.s is not None:
                s = self.s
            if self.K is not None:
                K = self.K

        """
        To reproduce the results from paper please use the following function to laod the
        parameters obtained by CV
        """

        xtrain, ytrain, xtest_seen, ytest_seen, xtest_unseen, ytest_unseen = dataloader.data_split()
        if self.pca_dim:
            xtrain, xtest_seen, xtest_unseen = apply_pca(xtrain, xtest_seen, xtest_unseen, self.pca_dim)
        time_s = time.time()
        # PPD parameter estimation
        Sig_s, mu_s, v_s, class_id, _ = self.bayesian_cls_train(
            xtrain,
            ytrain,
            dataloader.unseenclasses,
            att,
            k_0=k_0,
            k_1=k_1,
            m=m,
            s=s,
            K=K,
            pca_dim=self.pca_dim,
            tuning=False,
        )

        # Prediction phase
        ypred_unseen, prob_mat_unseen = self.bayesian_cls_evaluate(xtest_unseen, Sig_s, mu_s, v_s, class_id)
        ypred_seen, prob_mat_seen = self.bayesian_cls_evaluate(xtest_seen, Sig_s, mu_s, v_s, class_id)

        acc_per_cls_s, acc_per_cls_us, gzsl_seen_acc, gzsl_unseen_acc, H = perf_calc_acc(
            ytest_seen, ytest_unseen, ypred_seen, ypred_unseen, dataloader.label_to_genus
        )

        print("Results from k0=%.2f, k1=%.2f, m=%d, s=%.1f, K=%d on dataset:" % (k_0, k_1, m, s, K))
        print(
            "BSeen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%%"
            % (gzsl_seen_acc * 100, gzsl_unseen_acc * 100, H * 100)
        )

        if self.output:
            with open(self.output, "w") as f:
                json.dump(
                    {
                        "parameters": {"k0": k_0, "k1": k_1, "m": m, "s": s, "K": K},
                        "results": {"seen_acc": gzsl_seen_acc, "unseen_acc": gzsl_unseen_acc, "harmonic_mean": H},
                    },
                    f,
                    indent=2,
                )

        time_e = time.time()
        print("time cost: " + str(time_e - time_s))
