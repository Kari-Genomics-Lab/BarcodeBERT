import os
import time

import numpy as np
from scipy.special import gammaln, softmax
from scipy.linalg import solve_triangular, solve, eigh, eig
from utils import load_data, perf_calc_acc, apply_pca, get_label_in_int_to_dna_feature, extract_image_feature_from_hdf5, \
    get_dna_feature_in_numpy_array
from scipy.spatial.distance import cdist
from tqdm import tqdm


class Model(object):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.image_feature_dir = opt.image_feature_dir
        self.dna_feature_dir = opt.dna_feature_dir
        self.label_map_dir = opt.label_map_dir
        self.taxonomy_level = opt.taxonomy_level
        self.source_of_dna_barcode = opt.source_of_dna_barcode
        self.pca_dim = opt.pca_dim
        # self.tuning = opt.tuning
        self.using_cropped_image_feature = opt.using_cropped_image_feature
        self.dataset = "BioScan-1M"  # Hard code for now.
        self.unseen_classes = None
        self.x_train_seen = None
        self.y_train_seen = None
        self.x_test_seen = None
        self.y_test_seen = None
        self.x_test_unseen_easy = None
        self.y_test_unseen_easy = None
        self.x_test_unseen_hard = None
        self.y_test_unseen_hard = None
        self.label_in_int_to_dna_feature = None
        self.label_in_int_to_label_in_str = None
        self.dna_feature_in_numpy = None

    def load_data(self):

        if self.using_cropped_image_feature:
            image_type = 'cropped'
        else:
            image_type = 'original'

        path_to_image_feature_hdf5 = os.path.join(self.image_feature_dir,
                                                  self.taxonomy_level + "_image_feature_" + image_type + ".hdf5")

        self.x_train_seen, self.y_train_seen, self.x_test_seen, self.y_test_seen, self.x_test_unseen_easy, self.y_test_unseen_easy, self.x_test_unseen_hard, self.y_test_unseen_hard = extract_image_feature_from_hdf5(
            path_to_image_feature_hdf5)

        path_to_label_map_json = os.path.join(self.label_map_dir, self.taxonomy_level + "_level_label_map.json")

        path_to_dna_feature_hdf5 = os.path.join(self.dna_feature_dir, self.source_of_dna_barcode,
                                                self.taxonomy_level + "_dna_feature.hdf5")

        self.label_in_int_to_dna_feature, self.label_in_int_to_label_in_str, = get_label_in_int_to_dna_feature(
            path_to_dna_feature_hdf5, path_to_label_map_json,
            self.taxonomy_level)

        self.unseen_classes = np.unique(np.concatenate((self.y_test_unseen_easy, self.y_test_unseen_hard), axis=0))

        self.dna_feature_in_numpy = get_dna_feature_in_numpy_array(self.label_in_int_to_dna_feature).T

    ### Claculating class mean and covariance priors ###
    def calculate_priors(self, xtrain, ytrain, model_v='unconstrained'):
        dim = xtrain.shape[1]
        uy = np.unique(ytrain)
        nc = len(uy)
        if model_v == 'constrained':
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

    def bayesian_cls_train(self, x_tr, y_tr, unseenclasses, att, k_0=0.1, k_1=10, m=5 * 500, mu_0=0, s=1, Sigma_0=0,
                           K=2, pca_dim=0, tuning=False):

        s_classes = np.unique(y_tr)
        us_classes = unseenclasses
        # Attributes of seen and unseen classes
        # print(att.shape)
        # print(us_classes)
        # exit()
        att_unseen = att[:, us_classes].T
        att_seen = att[:, s_classes].T
        d0 = x_tr.shape[1]

        if tuning:
            Psi = (m - d0 - 1) * Sigma_0 / s
        else:
            [mu_0, Sigma_0] = self.calculate_priors(x_tr, y_tr)
            Psi = (m - d0 - 1) * Sigma_0 / s

        # Class predictive cov, mean and DoF from unconstrained model
        print('PPD derivation is Done!!')
        return self.calculate_ppd_params(x_tr, y_tr, att_seen, att_unseen, us_classes, K, Psi, mu_0, m, k_0, k_1)

    def check_for_tie(self, curr_unseen_class, usclass_list, seenclasses, curr_classes, s_in, K):
        ll = len(usclass_list)
        flag = True
        ect = 0
        while flag:
            flag = False
            for key, arr in usclass_list.items():
                if set(curr_classes) == set(arr):
                    flag = True
                    curr_classes[-1] = seenclasses[s_in[K + ect]]
                    ect += 1
                    break
        usclass_list[curr_unseen_class] = curr_classes

        return curr_classes, usclass_list

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
        pbar = tqdm(list(range(ncl)))
        for i in pbar:
            pbar.set_description("calculate ppd params for unseen classes: ")

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

        pbar = tqdm(list(range(ncl)))
        for i in pbar:
            pbar.set_description("calculate ppd params for seen classes: ")
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
            classes = seenclasses[s_in[1:K + 1]]

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
                Smu = ((cur_n * (k0 * k1 / (k0 + k1))) / ((k0 * k1 / (k0 + k1)) + cur_n)) * np.dot(cur_mu - mu0,
                                                                                                   (cur_mu - mu0).T)
                Sig_s[:, :, cnt] = (Psi + cur_S + Smu) / (
                        ((cur_n + (k0 * k1 / (k0 + k1))) * v_s[cnt]) / (cur_n + (k0 * k1 / (k0 + k1)) + 1))
                class_id[cnt] = uy[i]
                cnt += 1

        return Sig_s, mu_s, v_s, class_id, Sigmas

        ### PPD calculation (Log-Likelihood of Student-t) ###

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
            I = np.eye(Sig_s[:, :, j].shape[0])
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
                    Sig_s[:, :, j] += (-min_eig * k * k + np.spacing(min_eig)) * I

            # chsig = np.linalg.cholesky(Sig_s[:, :, j])  # Cholesky decomposition
            tpar = gl_pc[v_s[j] + d - 1] - (gl_pc[v_s[j] - 1] + (d / 2) * np.log(v_s[j]) + piconst) - np.sum(
                np.log(chsig.diagonal()))  # Stu-t lik part 1
            temp = solve_triangular(chsig, v.T, overwrite_b=True, check_finite=False, lower=True).T  # mrdivide(v,chsig)
            norm2 = np.einsum('ij,ij->i', temp, temp)  # faster than np.sum(temp**2)

            lkh[:, j] = tpar - 0.5 * (v_s[j] + d) * np.log(1 + (1 / v_s[j]) * norm2)

        bb = np.argmax(lkh, axis=1)
        ypred = class_id[bb]  # To ensure labels are correctly assigned back to original ones

        return ypred, lkh

    def hyperparameter_tuning(self, constrained=False):
        dim = 500

        if self.pca_dim is not None:
            dim = self.pca_dim

        k0_range = [0.1, 1]
        k1_range = [10, 25]
        s_range = [1, 5, 10]
        K_range = [1, 2, 3]



        bestH = 0
        # best_acc_s = 0
        # best_acc_us = 0
        best_k0 = None
        best_k1 = None
        best_m = None
        best_s = None
        best_K = None

        if not constrained:
            print('Applying PCA to reduce the dimension...')
            x_train_seen, x_test_seen, x_test_unseen_easy, x_test_unseen_hard = apply_pca(self.x_train_seen,
                                                                                          self.x_test_seen,
                                                                                          self.x_test_unseen_easy,
                                                                                          self.x_test_unseen_hard,
                                                                                          self.pca_dim)
            y_train_seen = self.y_train_seen

            [mu_0, Sigma_0] = self.calculate_priors(x_train_seen, y_train_seen, model_v='unconstrained')
            m_range = [5 * dim, 50 * dim, 500 * dim]
            print('Tuning is getting started...')
            for kk in K_range:
                for k_0 in k0_range:
                    for k_1 in k1_range:
                        for m in m_range:
                            for ss in s_range:
                                time_s = time.time()
                                Sig_s, mu_s, v_s, class_id, _ = self.bayesian_cls_train(x_train_seen, y_train_seen,
                                                                                        self.unseen_classes,
                                                                                        self.dna_feature_in_numpy,
                                                                                        k_0=k_0,
                                                                                        k_1=k_1, m=m, s=ss, K=kk,
                                                                                        mu_0=mu_0,
                                                                                        Sigma_0=Sigma_0,
                                                                                        pca_dim=self.pca_dim,
                                                                                        tuning=True)

                                print("Training is done.")

                                ### Prediction phase ###
                                y_pred_seen, prob_mat_seen = self.bayesian_cls_evaluate(x_test_seen, Sig_s, mu_s,
                                                                                        v_s,
                                                                                        class_id)
                                print("Prediction for seen easy is done.")

                                y_pred_unseen_easy, prob_mat_unseen_easy = self.bayesian_cls_evaluate(
                                    x_test_unseen_easy, Sig_s, mu_s,
                                    v_s, class_id)
                                print("Prediction for unseen easy is done.")

                                y_pred_unseen_hard, prob_mat_unseen_hard = self.bayesian_cls_evaluate(
                                    x_test_unseen_hard, Sig_s, mu_s,
                                    v_s, class_id)
                                print("Prediction for unseen hard is done.")


                                acc_per_cls_s, acc_per_cls_us_easy, acc_per_cls_us_hard, gzsl_seen_acc, gzsl_unseen_easy_acc, gzsl_unseen_hard_acc, H = perf_calc_acc(
                                    self.y_test_seen, self.y_test_unseen_easy, self.y_test_unseen_hard,
                                    y_pred_seen, y_pred_unseen_easy, y_pred_unseen_hard)
                                print('\nCurrent parameters k0=%.2f, k1=%.2f, m=%d, s=%.1f, K=%d on %s dataset:' % (
                                    k_0, k_1, m, ss, kk, self.dataset))
                                print()
                                if H > bestH:
                                    bestH = H
                                    best_k0 = k_0
                                    best_k1 = k_1
                                    best_m = m
                                    best_s = ss
                                    best_K = kk
                                print('\nResults from k0=%.2f, k1=%.2f, m=%d, s=%.1f, K=%d on %s dataset:' % (
                                        k_0, k_1, m, ss, kk, self.dataset))
                                print('BSeen acc: %.2f%%, Unseen easy acc: %.2f%%, Unseen hard acc: %.2f%%, Harmonic mean: %.2f%%\n' % (
                                        gzsl_seen_acc * 100, gzsl_unseen_easy_acc * 100, gzsl_unseen_hard_acc * 100, H * 100))
                                time_e = time.time()
                                print('total cost: ' + str(time_e - time_s))
        else:
            # TODO: hyper-parameter tuning for constrained model, ignored for now.
            pass

        return self.dna_feature_in_numpy, best_k0, best_k1, best_m, best_s, best_K

    def train_and_eval(self):
        att, k_0, k_1, m, s, K = self.hyperparameter_tuning(constrained=False)


        x_train_seen, x_test_seen, x_test_unseen_easy, x_test_unseen_hard = apply_pca(self.x_train_seen,
                                                                                          self.x_test_seen,
                                                                                          self.x_test_unseen_easy,
                                                                                          self.x_test_unseen_hard,
                                                                                          self.pca_dim)
        y_train_seen = self.y_train_seen
        time_s = time.time()
        ### PPD parameter estimation ###
        Sig_s, mu_s, v_s, class_id, _ = self.bayesian_cls_train(x_train_seen, y_train_seen, self.unseen_classes, att, k_0=k_0,
                                                                k_1=k_1, m=m,
                                                                s=s, K=K, pca_dim=self.pca_dim, tuning=False)

        ### Prediction phase ###
        y_pred_seen, prob_mat_seen = self.bayesian_cls_evaluate(x_test_seen, Sig_s, mu_s,
                                                                v_s,
                                                                class_id)
        print("Prediction for seen easy is done.")

        y_pred_unseen_easy, prob_mat_unseen_easy = self.bayesian_cls_evaluate(
            x_test_unseen_easy, Sig_s, mu_s,
            v_s, class_id)
        print("Prediction for unseen easy is done.")

        y_pred_unseen_hard, prob_mat_unseen_hard = self.bayesian_cls_evaluate(
            x_test_unseen_hard, Sig_s, mu_s,
            v_s, class_id)
        print("Prediction for unseen hard is done.")

        acc_per_cls_s, acc_per_cls_us_easy, acc_per_cls_us_hard, gzsl_seen_acc, gzsl_unseen_easy_acc, gzsl_unseen_hard_acc, H = perf_calc_acc(
            self.y_test_seen, self.y_test_unseen_easy, self.y_test_unseen_hard,
            y_pred_seen, y_pred_unseen_easy, y_pred_unseen_hard)

        print('Results from k0=%.2f, k1=%.2f, m=%d, s=%.1f, K=%d on %s dataset:' % (k_0, k_1, m, s, K, self.dataset))
        print('Best result: BSeen acc: %.2f%%, Unseen easy acc: %.2f%%, Unseen hard acc: %.2f%%, Harmonic mean: %.2f%%\n' % (
            gzsl_seen_acc * 100, gzsl_unseen_easy_acc * 100, gzsl_unseen_hard_acc * 100, H * 100))
        time_e = time.time()
        # print('time cost: ' + str(time_e - time_s))
