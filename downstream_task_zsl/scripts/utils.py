import json
import math
import random

import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

"""Data loading part"""


def hdf5_to_dict(group):
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = hdf5_to_dict(item)  # 递归处理子组
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # 将数据集的值存储到字典中
    return result


def reverse_dict(dictionary):
    reversed_dict = {value: key for key, value in dictionary.items()}
    return reversed_dict


def get_image_feature_from_image_names(np_array_of_image_name, path_to_hdf5, using_cropped_image_feature):
    image_feature = []
    with h5py.File(path_to_hdf5, 'r') as f:
        group_name = 'original_256_image_feature'
        if using_cropped_image_feature:
            group_name = 'cropped_256_image_feature'
        keys = list(f[group_name].keys())
        for name in tqdm(np_array_of_image_name):
            image_feature.append(f[group_name][name])
    f.close()
    image_feature = np.array(image_feature)

    return image_feature


def extract_image_feature_from_hdf5(path_to_image_feature_hdf5):
    with h5py.File(path_to_image_feature_hdf5, 'r') as f:
        all_splits = list(f.keys())
        x_train_seen = f['train_seen']['image_features'][:]
        y_train_seen = f['train_seen']['label_in_idx'][:]
        x_test_seen = f['test_seen']['image_features'][:]
        y_test_seen = f['test_seen']['label_in_idx'][:]
        x_test_unseen_easy = f['test_unseen_easy']['image_features'][:]
        y_test_unseen_easy = f['test_unseen_easy']['label_in_idx'][:]
        x_test_unseen_hard = f['test_unseen_hard']['image_features'][:]
        y_test_unseen_hard = f['test_unseen_hard']['label_in_idx'][:]
    f.close()
    return x_train_seen, y_train_seen, x_test_seen, y_test_seen, x_test_unseen_easy, y_test_unseen_easy, x_test_unseen_hard, y_test_unseen_hard


def get_label_in_int_to_dna_feature(path_to_dna_feature_hdf5, path_to_label_map_json, taxonomy_level):
    with open(path_to_label_map_json, "r") as json_file:
        label_in_str_to_label_in_int = json.load(json_file)
    json_file.close()

    label_in_int_to_label_in_str = reverse_dict(label_in_str_to_label_in_int)

    with h5py.File(path_to_dna_feature_hdf5, 'r') as f:
        group = f[taxonomy_level + '_class_level_dna_feature']
        label_in_str_to_dna_feature = hdf5_to_dict(group)
    f.close()
    label_in_int_to_dna_feature = {}
    for label_in_str in label_in_str_to_dna_feature.keys():
        label_in_int = label_in_str_to_label_in_int[label_in_str]
        label_in_int_to_dna_feature[label_in_int] = label_in_str_to_dna_feature[label_in_str]
    return label_in_int_to_dna_feature, label_in_int_to_label_in_str


def load_data(image_feature_dir, dna_feature_dir, label_map_dir, taxonomy_level, using_cropped_image_feature,
              source_of_dna_barcode):
    if using_cropped_image_feature:
        image_type = 'cropped'
    else:
        image_type = 'original'

    path_to_image_feature_hdf5 = os.path.join(image_feature_dir,
                                              taxonomy_level + "_image_feature_" + image_type + ".hdf5")

    x_train_seen, y_train_seen, x_test_seen, y_test_seen, x_test_unseen_easy, y_test_unseen_easy, x_test_unseen_hard, y_test_unseen_hard = extract_image_feature_from_hdf5(
        path_to_image_feature_hdf5)

    path_to_label_map_json = os.path.join(label_map_dir, taxonomy_level + "_level_label_map.json")

    path_to_dna_feature_hdf5 = os.path.join(dna_feature_dir, source_of_dna_barcode,
                                            taxonomy_level + "_dna_feature.hdf5")

    label_in_int_to_dna_feature, label_in_int_to_label_in_str = get_label_in_int_to_dna_feature(
        path_to_dna_feature_hdf5, path_to_label_map_json,
        taxonomy_level)


### Seen, Unseen class and Harmonic mean claculation ###
def perf_calc_acc(y_ts_s, y_ts_us_easy, y_ts_us_hard, y_pred_s, y_pred_us_easy, y_pred_us_hard):
    seen_cls = np.unique(y_ts_s)
    unseen_easy_cls = np.unique(y_ts_us_easy)
    unseen_hard_cls = np.unique(y_ts_us_hard)

    # Performance calculation
    acc_per_cls_s = np.zeros((len(seen_cls), 1))
    acc_per_cls_us_easy = np.zeros((len(unseen_easy_cls), 1))
    acc_per_cls_us_hard = np.zeros((len(unseen_hard_cls), 1))

    for i in range(len(seen_cls)):
        lb = seen_cls[i]
        idx = y_ts_s == lb
        acc_per_cls_s[i] = np.sum(y_pred_s[idx.ravel()] == lb) / np.sum(idx)

    for i in range(len(unseen_easy_cls)):
        lb = unseen_easy_cls[i]
        idx = y_ts_us_easy == lb
        acc_per_cls_us_easy[i] = np.sum(y_pred_us_easy[idx.ravel()] == lb) / np.sum(idx)

    for i in range(len(unseen_hard_cls)):
        lb = unseen_hard_cls[i]
        idx = y_ts_us_hard == lb
        acc_per_cls_us_hard[i] = np.sum(y_pred_us_hard[idx.ravel()] == lb) / np.sum(idx)


    ave_s = np.mean(acc_per_cls_s)
    ave_us_easy = np.mean(acc_per_cls_us_easy)
    ave_us_hard = np.mean(acc_per_cls_us_hard)
    H = 3 * ave_s * ave_us_easy * ave_us_hard / (ave_s + ave_us_easy + ave_us_hard)

    return acc_per_cls_s, acc_per_cls_us_easy, acc_per_cls_us_hard, ave_s, ave_us_easy, ave_us_hard, H


def apply_pca(x_train_seen, x_test_seen, x_test_unseen_easy, x_test_unseen_hard, pca_dim):
    # Dimentionality reduction using PCA
    _, eig_vec = eigh(np.cov(x_train_seen.T))
    x_train_seen = np.dot(x_train_seen, eig_vec[:, -pca_dim:])
    x_test_seen = np.dot(x_test_seen, eig_vec[:, -pca_dim:])
    x_test_unseen_easy = np.dot(x_test_unseen_easy, eig_vec[:, -pca_dim:])
    x_test_unseen_hard = np.dot(x_test_unseen_hard, eig_vec[:, -pca_dim:])
    return x_train_seen, x_test_seen, x_test_unseen_easy, x_test_unseen_hard


def get_dna_feature_in_numpy_array(label_in_int_to_dna_feature):
    list_of_label_in_int = sorted(list(label_in_int_to_dna_feature.keys()))
    list_of_dna_feature = []
    for int_label in list_of_label_in_int:
        list_of_dna_feature.append(label_in_int_to_dna_feature[int_label])
    dna_feature_in_numpy = np.vstack(list_of_dna_feature)

    return dna_feature_in_numpy


"""
def crossvalind_holdout(labels, idxs, ratio=0.2):
    # Shuffle the index
    random.seed(42)
    random.shuffle(idxs)
    Y = labels[idxs]
    unique_Y, count = np.unique(Y, return_counts=True)
    count = dict(zip(unique_Y, count))
    train = []
    test = []
    for cls in unique_Y:
        threshold = math.ceil(ratio * count[cls])
        current_count = 0
        for idx, i in enumerate(Y):
            if i == cls:
                if current_count < threshold:
                    test.append(idxs[idx])
                else:
                    train.append(idxs[idx])
                current_count += 1
    return train, test

def split_loc_by_ratio(loc_index, ratio=0.2):
    n = int(len(loc_index) * ratio)
    np.random.shuffle(loc_index)
    test_loc = loc_index[0:n]
    train_loc = loc_index[n:]
    return np.sort(train_loc), np.sort(test_loc)
"""
