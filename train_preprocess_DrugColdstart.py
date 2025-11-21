#!/usr/bin/env python
# coding: utf-8
import os
import pathlib
import pickle
import random
import numpy as np
import scipy.sparse as sp
import scipy.io
import pandas as pd
import json

seed = 123
np.random.seed(seed)
random.seed(seed)


def rows_cols_mask(mat, row_mask, col_mask=None):
    """返回一个被掩码后的 copy；False 的行/列将被清零"""
    if col_mask is None:
        col_mask = row_mask
    out = mat.copy()
    if out.ndim != 2:
        raise ValueError("rows_cols_mask expects 2D matrix")
    out[~row_mask, :] = 0
    out[:, ~col_mask] = 0
    return out

def split_node_sets(dp_idx):
    """从 (2, N) 的正样本对里，抽取该 split 出现过的药物集合 & 蛋白质集合"""
    if dp_idx is None or dp_idx.size == 0:
        return set(), set()
    d_set = set(dp_idx[0].tolist())
    p_set = set(dp_idx[1].tolist())
    return d_set, p_set


def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)

def metapath_xx(x_x_list, num, sample=None):
    x_x = []
    for x, x_list in x_x_list.items():
        if sample is not None:
            candidate_list = np.random.choice(len(x_list), min(len(x_list), sample), replace=False)
            x_list = x_list[candidate_list]
        x_x.extend([(x, x1) for x1 in x_list])
    x_x = np.array(x_x)
    x_x = x_x + num
    sorted_index = sorted(list(range(len(x_x))), key=lambda i: x_x[i].tolist())
    x_x = x_x[sorted_index]
    return x_x

def metapath_yxy(x_y_list, num1, num2, sample=None):
    y_x_y = []
    for x, y_list in x_y_list.items():
        if sample is not None:
            if len(y_list) == 0:
                continue
            candidate_list1 = np.random.choice(len(y_list), min(len(y_list), sample), replace=False)
            candidate_list2 = np.random.choice(len(y_list), min(len(y_list), sample), replace=False)
            y_list1 = y_list[candidate_list1]
            y_list2 = y_list[candidate_list2]
            y_x_y.extend([(y1, x, y2) for y1 in y_list1 for y2 in y_list2])
        else:
            y_x_y.extend([(y1, x, y2) for y1 in y_list for y2 in y_list])

    # 转成 numpy 数组
    y_x_y = np.array(y_x_y, dtype=np.int64)

    # 空结果保护：返回 (0,3) 的二维数组
    if y_x_y.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    # 正常偏移与排序
    y_x_y[:, [0, 2]] += num1
    y_x_y[:, 1] += num2
    sorted_index = sorted(
        range(len(y_x_y)),
        key=lambda i: y_x_y[i, [0, 2, 1]].tolist()
    )
    y_x_y = y_x_y[sorted_index]
    return y_x_y


def metapath_yxxy(x_x, x_y_list, num1, num2, sample=None):
    """
    生成 y-x-x-y 的四元组：
    - x_x: 可迭代的 (x1, x2) 对
    - x_y_list: dict 或 list-like，按 (x - num2) 下标/键取到 y 的邻居数组
    - num1: 给 y 索引加的偏移（用于拼大矩阵时的行/列偏移）
    - num2: 给 x 索引加的偏移
    - sample: 限制每个 x 的 y 邻居采样数量（None 表示全量）
    """
    y_x_x_y = []
    for x1, x2 in x_x:
        # 取 x1/x2 的 y 邻居；不存在则视为空
        y_nei1 = x_y_list.get(x1 - num2, []) if isinstance(x_y_list, dict) else x_y_list[x1 - num2]
        y_nei2 = x_y_list.get(x2 - num2, []) if isinstance(x_y_list, dict) else x_y_list[x2 - num2]

        # 转 numpy，统一处理
        y_nei1 = np.asarray(y_nei1, dtype=np.int64)
        y_nei2 = np.asarray(y_nei2, dtype=np.int64)
        if y_nei1.size == 0 or y_nei2.size == 0:
            continue

        if sample is not None:
            k1 = min(len(y_nei1), sample)
            k2 = min(len(y_nei2), sample)
            if k1 == 0 or k2 == 0:
                continue
            idx1 = np.random.choice(len(y_nei1), k1, replace=False)
            idx2 = np.random.choice(len(y_nei2), k2, replace=False)
            y_nei1 = y_nei1[idx1]
            y_nei2 = y_nei2[idx2]

        # 笛卡尔积
        y_x_x_y.extend((int(y1), int(x1), int(x2), int(y2)) for y1 in y_nei1 for y2 in y_nei2)

    y_x_x_y = np.asarray(y_x_x_y, dtype=np.int64)
    if y_x_x_y.size == 0:
        return np.zeros((0, 4), dtype=np.int64)

    # 偏移并排序（按 y1, y2, x1, x2 排）
    y_x_x_y[:, [0, 3]] += num1
    # x1, x2 已经是全局索引的话就不再偏移；若需要也可在此加：y_x_x_y[:, [1, 2]] += num2
    order = np.lexsort((y_x_x_y[:, 2], y_x_x_y[:, 1], y_x_x_y[:, 3], y_x_x_y[:, 0]))  # 次序可按需调整
    y_x_x_y = y_x_x_y[order]
    return y_x_x_y


def metapath_zyxyz(y_x_y, y_z_list, num1, num2, ratio):
    """
    生成 z-y-x-y-z 的五元组：
    - y_x_y: 形如 (y1, x, y2) 的二维数组（可能为空）
    - y_z_list: dict 或 list-like，按 (y - num2) 下标/键取到 z 的邻居数组
    - num1: 给 z 索引加的偏移
    - num2: 给 y 索引加的偏移
    - ratio: 若 <=1 表示按比例采样（向上取整，至少 1）；>1 表示最多采样 ratio 个
    """
    y_x_y = np.asarray(y_x_y, dtype=np.int64)
    if y_x_y.size == 0:
        return np.zeros((0, 5), dtype=np.int64)

    z_y_x_y_z = []
    for y1, x, y2 in y_x_y:
        # 取 y1/y2 的 z 邻居；不存在则视为空
        z_nei1 = y_z_list.get(y1 - num2, []) if isinstance(y_z_list, dict) else y_z_list[y1 - num2]
        z_nei2 = y_z_list.get(y2 - num2, []) if isinstance(y_z_list, dict) else y_z_list[y2 - num2]

        z_nei1 = np.asarray(z_nei1, dtype=np.int64)
        z_nei2 = np.asarray(z_nei2, dtype=np.int64)
        if z_nei1.size == 0 or z_nei2.size == 0:
            continue

        # 采样规则
        if ratio is not None:
            if ratio <= 1:
                k1 = max(1, int(np.ceil(ratio * len(z_nei1))))
                k2 = max(1, int(np.ceil(ratio * len(z_nei2))))
            else:
                k1 = int(min(ratio, len(z_nei1)))
                k2 = int(min(ratio, len(z_nei2)))
            if k1 == 0 or k2 == 0:
                continue
            idx1 = np.random.choice(len(z_nei1), k1, replace=False)
            idx2 = np.random.choice(len(z_nei2), k2, replace=False)
            z_nei1 = z_nei1[idx1]
            z_nei2 = z_nei2[idx2]

        # 笛卡尔积
        z_y_x_y_z.extend((int(z1), int(y1), int(x), int(y2), int(z2)) for z1 in z_nei1 for z2 in z_nei2)

    z_y_x_y_z = np.asarray(z_y_x_y_z, dtype=np.int64)
    if z_y_x_y_z.size == 0:
        return np.zeros((0, 5), dtype=np.int64)

    # 偏移并排序（按 z1, z2, y1, x, y2 排）
    z_y_x_y_z[:, [0, 4]] += num1
    order = np.lexsort((z_y_x_y_z[:, 3], z_y_x_y_z[:, 2], z_y_x_y_z[:, 1], z_y_x_y_z[:, 4], z_y_x_y_z[:, 0]))
    z_y_x_y_z = z_y_x_y_z[order]
    return z_y_x_y_z


def sampling(array_list, num, offset):
    target_list = np.arange(num).tolist()
    sampled_list = []
    k = 100 # number of samiling

    left = 0
    right = 0
    for target_idx in target_list:
        while right < len(array_list) and array_list[right, 0] == target_idx + offset:
            right += 1
        target_array = array_list[left:right, :]

        if len(target_array) > 0:
            samples = min(k, len(target_array))
            sampled_idx = np.random.choice(len(target_array), samples, replace=False)
            target_array = target_array[sampled_idx]

        sampled_list.append(target_array)
        left = right
    sampled_array = np.concatenate(sampled_list, axis=0)
    sorted_index = sorted(list(range(len(sampled_array))), key=lambda i: sampled_array[i, [0, 2, 1]].tolist())
    sampled_array = sampled_array[sorted_index]

    return sampled_array


def get_metapath(metapath, num_drug, num_protein, num_disease, num_se, save_prefix):
    if len(metapath) == 2:
        # (0, 0)
        if metapath == (0, 0):
            metapath_indices = metapath_xx(drug_drug_list, num=0)
        # (1, 1)
        elif metapath == (1, 1):
            metapath_indices = metapath_xx(protein_protein_list, num=num_drug)

    elif len(metapath) == 3:
        # (0, 1, 0)
        if metapath == (0, 1, 0):
            metapath_indices = metapath_yxy(protein_drug_list, num1=0, num2=num_drug)
        # (0, 2, 0)
        elif metapath == (0, 2, 0):
            metapath_indices = metapath_yxy(disease_drug_list, num1=0, num2=num_drug + num_protein, sample=100)
        # (0, 3, 0)
        elif metapath == (0, 3, 0):
            metapath_indices = metapath_yxy(se_drug_list, num1=0, num2=num_drug + num_protein + num_disease, sample=100)
        # (1, 0, 1)
        elif metapath == (1, 0, 1):
            metapath_indices = metapath_yxy(drug_protein_list, num1=num_drug, num2=0)
        # (1, 2, 1)
        elif metapath == (1, 2, 1):
            metapath_indices = metapath_yxy(disease_protein_list, num1=num_drug, num2=num_drug + num_protein, sample=100)

    elif len(metapath) == 4:
        # (0, 1, 1, 0)
        if metapath == (0, 1, 1, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (1, 1))) + '.npy'):
            #     p_p = np.load(save_prefix + '-'.join(map(str, (1, 1))) + '.npy')
            # # else:
            #     p_p = metapath_xx(protein_protein_list, num=num_drug)
            #     np.save(save_prefix + '-'.join(map(str, (1, 1))) + '.npy', p_p)
            p_p = metapath_xx(protein_protein_list, num=num_drug, sample=50)
            metapath_indices = metapath_yxxy(p_p, protein_drug_list, num1=0, num2=num_drug, sample=30)
        # (1, 0, 0, 1)
        elif metapath == (1, 0, 0, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (0, 0))) + '.npy'):
            #     d_d = np.load(save_prefix + '-'.join(map(str, (0, 0))) + '.npy')
            # else:
            #     d_d = metapath_xx(drug_drug_list, num=0)
            #     np.save(save_prefix + '-'.join(map(str, (0, 0))) + '.npy', d_d)
            d_d = metapath_xx(drug_drug_list, num=0, sample=100)
            metapath_indices = metapath_yxxy(d_d, drug_protein_list, num1=num_drug, num2=0, sample=10)

    elif len(metapath) == 5:
        # 0-1-0-1-0
        if metapath == (0, 1, 0, 1, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (1, 0, 1))) + '.npy'):
            #     p_d_p = np.load(save_prefix + '-'.join(map(str, (1, 0, 1))) + '.npy')
            # else:
            #     p_d_p = metapath_yxy(drug_protein_list, num1=num_drug, num2=0)
            #     np.save(save_prefix + '-'.join(map(str, (1, 0, 1))) + '.npy', p_d_p)
            p_d_p = metapath_yxy(drug_protein_list, num1=num_drug, num2=0, sample=20)
            p_d_p = sampling(p_d_p, num=num_protein, offset=num_drug)
            metapath_indices = metapath_zyxyz(p_d_p, protein_drug_list, num1=0, num2=num_drug, ratio=5)
        # 0-1-2-1-0
        elif metapath == (0, 1, 2, 1, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (1, 2, 1))) + '.npy'):
            #     p_i_p = np.load(save_prefix + '-'.join(map(str, (1, 2, 1))) + '.npy')
            # else:
            #     p_i_p = metapath_yxy(disease_protein_list, num1=num_drug, num2=num_drug + num_protein, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (1, 2, 1))) + '.npy', p_i_p)
            p_i_p = metapath_yxy(disease_protein_list, num1=num_drug, num2=num_drug + num_protein, sample=80)
            p_i_p = sampling(p_i_p, num=num_protein, offset=num_drug)
            metapath_indices = metapath_zyxyz(p_i_p, protein_drug_list, num1=0, num2=num_drug, ratio=5)
        # 0-2-0-2-0
        elif metapath == (0, 2, 0, 2, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy'):
            #     i_d_i = np.load(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy')
            # else:
            #     i_d_i = metapath_yxy(drug_disease_lit, num1=num_drug + num_protein, num2=0, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy', i_d_i)
            i_d_i = metapath_yxy(drug_disease_list, num1=num_drug + num_protein, num2=0, sample=80)
            i_d_i = sampling(i_d_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_d_i, disease_drug_list, num1=0, num2=num_drug + num_protein, ratio=5)
        # 0-3-0-3-0
        elif metapath == (0, 3, 0, 3, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (3, 0, 3))) + '.npy'):
            #     s_d_s = np.load(save_prefix + '-'.join(map(str, (3, 0, 3))) + '.npy')
            # else:
            #     s_d_s = metapath_yxy(drug_se_list, num1=num_drug + num_protein + num_disease, num2=0, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (3, 0, 3))) + '.npy', s_d_s)
            s_d_s = metapath_yxy(drug_se_list, num1=num_drug + num_protein + num_disease, num2=0, sample=80)
            s_d_s = sampling(s_d_s, num=num_se, offset=num_drug + num_protein + num_disease)
            metapath_indices = metapath_zyxyz(s_d_s, se_drug_list, num1=0, num2=num_drug + num_protein + num_disease, ratio=5)
        # 0-2-1-2-0
        elif metapath == (0, 2, 1, 2, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy'):
            #     i_p_i = np.load(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy')
            # else:
            #     i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy', i_p_i)
            i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            i_p_i = sampling(i_p_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_p_i, disease_drug_list, num1=0, num2=num_drug + num_protein, ratio=5)
        # 1-0-1-0-1
        elif metapath == (1, 0, 1, 0, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (0, 1, 0))) + '.npy'):
            #     d_p_d = np.load(save_prefix + '-'.join(map(str, (0, 1, 0))) + '.npy')
            # else:
            #     d_p_d = metapath_yxy(protein_drug_list, num1=0, num2=num_drug)
            #     np.save(save_prefix + '-'.join(map(str, (0, 1, 0))) + '.npy', d_p_d)
            d_p_d = metapath_yxy(protein_drug_list, num1=0, num2=num_drug, sample=10)
            d_p_d = sampling(d_p_d, num=num_drug, offset=0)
            metapath_indices = metapath_zyxyz(d_p_d, drug_protein_list, num1=num_drug, num2=0, ratio=10)
        # 1-0-2-0-1
        elif metapath == (1, 0, 2, 0, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (0, 2, 0))) + '.npy'):
            #     d_i_d = np.load(save_prefix + '-'.join(map(str, (0, 2, 0))) + '.npy')
            # else:
            #     d_i_d = metapath_yxy(disease_drug_list, num1=0, num2=num_drug + num_protein, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (0, 2, 0))) + '.npy', d_i_d)
            d_i_d = metapath_yxy(disease_drug_list, num1=0, num2=num_drug + num_protein, sample=80)
            d_i_d = sampling(d_i_d, num=num_drug, offset=0)
            metapath_indices = metapath_zyxyz(d_i_d, drug_protein_list, num1=num_drug, num2=0, ratio=5)
        # 1-2-0-2-1
        elif metapath == (1, 2, 0, 2, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy'):
            #     i_d_i = np.load(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy')
            # else:
            #     i_d_i = metapath_yxy(drug_disease_lit, num1=num_drug + num_protein, num2=0, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy', i_d_i)
            i_d_i = metapath_yxy(drug_disease_list, num1=num_drug + num_protein, num2=0, sample=80)
            i_d_i = sampling(i_d_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_d_i, disease_protein_list, num1=num_drug, num2=num_drug + num_protein, ratio=5)
        # 1-2-1-2-1
        elif metapath == (1, 2, 1, 2, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy'):
            #     i_p_i = np.load(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy')
            # else:
            #     i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy', i_p_i)
            i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            i_p_i = sampling(i_p_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_p_i, disease_protein_list, num1=num_drug, num2=num_drug + num_protein, ratio=5)

    return metapath_indices

def target_metapath_and_neightbors(edge_metapath_idx_array, target_idx_list, offset:int =0 ):
    # write all things
    offset = int(offset)
    target_metapaths_mapping = {}
    target_neighbors = {}
    left = 0
    right = 0
    for target_idx in target_idx_list:
        # target_metapaths_mapping = {}
        # target_neighbors = {}
        while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset:
            right += 1
        target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
        neighbors = edge_metapath_idx_array[left:right, -1] - offset
        # neighbors = list(map(str, neighbors))
        target_neighbors[target_idx] = [target_idx] + neighbors.tolist()
        left = right

    return target_metapaths_mapping, target_neighbors


def Load_Adj_Togerther(dir_lists, ratio=0.01):
    a = np.loadtxt(dir_lists[0])
    print('Before Interactions: ', sum(sum(a)))

    for i in range(len(dir_lists) - 1):
        b_new = np.zeros_like(a)

        b = np.loadtxt(dir_lists[i + 1])
        # remove diagonal elements
        b = b - np.diag(np.diag(b))
        # if the matrix are symmetrical, get the triu matrix
        if (b == b.T).all():
            b = np.triu(b)
        index = np.nonzero(b)
        values = b[index]
        index = np.transpose(index)
        edgelist = np.concatenate([index, values.reshape(-1, 1)], axis=1)
        topK_idx = np.argpartition(edgelist[:, 2], int(ratio * len(edgelist)))[-(int(ratio * len(edgelist))):]
        print(len(topK_idx))
        select_idx = index[topK_idx]
        b_new[select_idx[:, 0], select_idx[:, 1]] = b[select_idx[:, 0], select_idx[:, 1]]
        a = a + b_new

    a = a + a.T
    a[a > 0] = 1
    a[a <= 0] = 0
    a = a + np.eye(a.shape[0], a.shape[1])
    a = a.astype(int)
    print('After Interactions: ', sum(sum(a)))

    return a

def get_adjM(drug_drug, drug_protein, drug_disease, drug_sideEffect, protein_protein, protein_disease,
             num_drug, num_protein, num_disease, num_se):
    # Drug-0, Protein-1, Disease-2, Side-effect-3
    dim = num_drug + num_protein + num_disease + num_se
    adjM = np.zeros((dim, dim), dtype=int)
    adjM[:num_drug, :num_drug] = drug_drug
    adjM[:num_drug, num_drug: num_drug + num_protein] = drug_protein
    adjM[:num_drug, num_drug + num_protein: num_drug + num_protein + num_disease] = drug_disease
    adjM[:num_drug, num_drug + num_protein + num_disease:] = drug_sideEffect
    adjM[num_drug: num_drug + num_protein, num_drug: num_drug + num_protein] = protein_protein
    adjM[num_drug: num_drug + num_protein, num_drug + num_protein: num_drug + num_protein + num_disease] = protein_disease

    adjM[num_drug: num_drug + num_protein, :num_drug] = drug_protein.T
    adjM[num_drug + num_protein: num_drug + num_protein + num_disease, :num_drug] = drug_disease.T
    adjM[num_drug + num_protein + num_disease:, :num_drug] = drug_sideEffect.T
    adjM[num_drug + num_protein: num_drug + num_protein + num_disease, num_drug: num_drug + num_protein] = protein_disease.T
    
    return adjM


def build_and_cache_for_split(
    split_name, dp_idx, *,
    drug_drug, drug_protein_full, drug_disease, drug_sideEffect,
    protein_protein, protein_disease,
    num_drug, num_protein, num_disease, num_se,
    expected_metapaths, counter, save_prefix,
    save_adjM_npz=False,
    coldstart_mode="drug"  # "drug" | "protein"
):
    # === A) 计算“当前 split 出现的药物/蛋白质集合”并做掩码 ===
    split_drugs, split_prots = split_node_sets(dp_idx)  # 仍然可用，但蛋白质将不用于屏蔽
    drug_mask = np.zeros(num_drug, dtype=bool);
    drug_mask[list(split_drugs)] = True

    if coldstart_mode == "drug":
        # 药物侧冷启动：蛋白质不做屏蔽（全量可见）
        protein_mask = np.ones(num_protein, dtype=bool)
    else:
        # 蛋白质侧冷启动：相反（可选）
        protein_mask = np.zeros(num_protein, dtype=bool);
        protein_mask[list(split_prots)] = True

    # === B) D–P：只保留本 split 的正边（与原逻辑一致），并对“非本 split 的行/列”清零 ===
    # === B) D–P：train 才保留正边；valid/test 一律不写，避免泄露 ===
    fold_drug_protein = np.zeros_like(drug_protein_full, dtype=drug_protein_full.dtype)

    if split_name == 'train' and dp_idx is not None and dp_idx.size > 0:
        d_idx, p_idx = dp_idx[0], dp_idx[1]
        fold_drug_protein[d_idx, p_idx] = drug_protein_full[d_idx, p_idx]

    # 屏蔽非本 split 的药物/蛋白质行列（药物侧 CS 下 protein_mask=全 True）
    fold_drug_protein = rows_cols_mask(fold_drug_protein, drug_mask, protein_mask)

    # === C) 其它关系也要同步切掉“非本 split 的药物/蛋白质” ===
    # 药物相关：D–D, D–Disease, D–SE 仅保留本 split 的药物行（D–D 还需列）
    drug_drug_cs      = rows_cols_mask(drug_drug, drug_mask, drug_mask)
    drug_disease_cs   = rows_cols_mask(drug_disease, drug_mask, np.ones(num_disease, dtype=bool))
    drug_se_cs        = rows_cols_mask(drug_sideEffect, drug_mask, np.ones(num_se, dtype=bool))
    # 蛋白质相关：P–P, P–Disease 仅保留本 split 的蛋白质行（P–P 还需列）
    protein_protein_cs = rows_cols_mask(protein_protein, protein_mask, protein_mask)
    protein_disease_cs = rows_cols_mask(protein_disease, protein_mask, np.ones(num_disease, dtype=bool))

    # === D) 拼异质邻接（此时“非本 split 的药物/蛋白质”在所有块都被切干净了） ===
    adjM = get_adjM(
        drug_drug_cs, fold_drug_protein, drug_disease_cs, drug_se_cs,
        protein_protein_cs, protein_disease_cs,
        num_drug, num_protein, num_disease, num_se
    )

    # === E) 基于本 split 的 adjM 生成一跳邻居字典（与原逻辑一致） ===
    global drug_drug_list, drug_protein_list, drug_disease_list, drug_se_list
    global protein_drug_list, protein_protein_list, protein_disease_list
    global disease_drug_list, disease_protein_list, se_drug_list

    drug_drug_list     = {i: adjM[i, :num_drug].nonzero()[0] for i in range(num_drug)}
    drug_protein_list  = {i: adjM[i, num_drug:num_drug + num_protein].nonzero()[0] for i in range(num_drug)}
    drug_disease_list  = {i: adjM[i, num_drug + num_protein:num_drug + num_protein + num_disease].nonzero()[0] for i in range(num_drug)}
    drug_se_list       = {i: adjM[i, num_drug + num_protein + num_disease:].nonzero()[0] for i in range(num_drug)}

    protein_drug_list     = {i: adjM[num_drug + i, :num_drug].nonzero()[0] for i in range(num_protein)}
    protein_protein_list  = {i: adjM[num_drug + i, num_drug:num_drug + num_protein].nonzero()[0] for i in range(num_protein)}
    protein_disease_list  = {i: adjM[num_drug + i, num_drug + num_protein:num_drug + num_protein + num_disease].nonzero()[0] for i in range(num_protein)}

    disease_drug_list     = {i: adjM[num_drug + num_protein + i, :num_drug].nonzero()[0] for i in range(num_disease)}
    disease_protein_list  = {i: adjM[num_drug + num_protein + i, num_drug:num_drug + num_protein].nonzero()[0] for i in range(num_disease)}

    se_drug_list          = {i: adjM[num_drug + num_protein + num_disease + i, : num_drug].nonzero()[0] for i in range(num_se)}

    if save_adjM_npz:
        split_adj_dir = os.path.join(save_prefix, f"repeat{counter}", "adjM")
        make_dir(split_adj_dir)
        np.save(os.path.join(split_adj_dir, f"adjM_{split_name}.npy"), adjM)

    # ——健壮性检查：确认 D–P 只含本 split 的边，且屏蔽生效——
    dp_block = adjM[:num_drug, num_drug:num_drug + num_protein]
    if dp_idx is not None and dp_idx.size > 0:
        mask = np.zeros_like(dp_block, dtype=bool)
        d_idx, p_idx = dp_idx[0], dp_idx[1]
        mask[d_idx, p_idx] = True
        leak = (dp_block.astype(bool) & ~mask).sum()
        assert leak == 0, f"[{split_name}] D–P block 有 {leak} 条非本 split 的边"
    assert dp_block[~drug_mask, :].sum() == 0 and dp_block[:, ~protein_mask].sum() == 0, \
        f"[{split_name}] 非本 split 的药物/蛋白质在 D–P 中未清零"


    if coldstart_mode == "drug":
        target_idx_lists = [sorted(list(split_drugs)), list(range(num_protein))]
    else:
        target_idx_lists = [list(range(num_drug)), sorted(list(split_prots))]
    offset_list = [0, num_drug]

    for i, metapaths in enumerate(expected_metapaths):  # i=0 Drug端, i=1 Protein端
        for metapath in metapaths:
            metapath_dir = save_prefix + f'repeat{counter}/{i}/{split_name}/_'
            make_dir(metapath_dir)

            cache_base = metapath_dir + '-'.join(map(str, metapath))
            npy_path = cache_base + '.npy'
            idx_pkl = cache_base + '.idx.pkl'
            adj_pkl = cache_base + '.adjlist.pkl'

            # 强烈建议：冷启动时不要直接复用旧 .npy，因为抽样/掩码可能变化
            if os.path.isfile(npy_path):
                edge_metapath_idx_array = np.load(npy_path, allow_pickle=True)
            else:
                edge_metapath_idx_array = get_metapath(
                    metapath, num_drug, num_protein, num_disease, num_se, metapath_dir
                )
                np.save(npy_path, edge_metapath_idx_array)

            target_metapaths, target_neighbors = target_metapath_and_neightbors(
                edge_metapath_idx_array, target_idx_lists[i], offset=offset_list[i]
            )
            pickle.dump(target_metapaths, open(idx_pkl, 'wb'))
            pickle.dump(target_neighbors, open(adj_pkl, 'wb'))

    return adjM


def get_type_mask(num_drug, num_protein, num_disease, num_se):
    # Drug-0, Protein-1, Disease-2, Side-effect-3
    dim = num_drug + num_protein + num_disease + num_se
    type_mask = np.zeros((dim), dtype=int)
    type_mask[num_drug: num_drug + num_protein] = 1
    type_mask[num_drug + num_protein: num_drug + num_protein + num_disease] = 2
    type_mask[num_drug + num_protein + num_disease:] = 3
    return type_mask

if __name__ == '__main__':
    data_set = 'data_luo'
    data_dir = './hetero_dataset/{}/'.format(data_set)
    num_repeats = 1


    def load_pairs_to_idx2(path):
        """读取 [ [d,p], ... ] -> shape(2, N) 的 int 索引数组"""
        pairs = json.load(open(path, 'r'))
        arr = np.array(pairs, dtype=int).T  # (2, N)
        return arr


    train_idx = load_pairs_to_idx2(os.path.join(data_dir, 'train_drug_coldstart.json'))  # shape (2, N_train)
    valid_idx = load_pairs_to_idx2(os.path.join(data_dir, 'valid_drug_coldstart.json'))  # shape (2, N_valid)
    test_idx = load_pairs_to_idx2(os.path.join(data_dir, 'test_drug_coldstart.json'))  # shape (2, N_test)

    save_prefix = data_dir + '/processed_coldstart_drug/'
    os.makedirs(save_prefix, exist_ok=True)

    expected_metapaths = [[(0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 1, 1, 0),
                          (0, 1, 0, 1, 0), (0, 2, 0, 2, 0), (0, 3, 0, 3, 0), (0, 1, 2, 1, 0), (0, 2, 1, 2, 0)],
                          [(1, 1), (1, 0, 1), (1, 2, 1), (1, 0, 0, 1),
                           (1, 0, 1, 0, 1), (1, 0, 2, 0, 1), (1, 2, 0, 2, 1), (1, 2, 1, 2, 1)]]

    ## Step 1: Reconstruct Drug-Drug interaction network and Protein-Protein interaxtion network
    # Reconstruct Drug-Drug interaction network
    # 1 interaction + 4 sim
    drug_drug_path = data_dir + '/mat_data/mat_drug_drug.txt'
    drug_drug_sim_chemical_path = data_dir + '/sim_network/Sim_mat_drugs.txt'
    drug_drug_sim_interaction_path = data_dir + '/sim_network/Sim_mat_drug_drug.txt'
    drug_drug_sim_se_path = data_dir + '/sim_network/Sim_mat_drug_se.txt'
    drug_drug_sim_disease_path = data_dir + '/sim_network/Sim_mat_drug_disease.txt'

    # Reconstruct Protein-Protein interaxtion network
    # 1interaction + 3 sim
    protein_protein_path = data_dir + '/mat_data/mat_protein_protein.txt'
    protein_protein_sim_sequence_path = data_dir + '/sim_network/Sim_mat_proteins.txt'
    protein_protein_sim_disease_path = data_dir + '/sim_network/Sim_mat_protein_disease.txt'
    protein_protein_sim_interaction_path = data_dir + '/sim_network/Sim_mat_protein_protein.txt'

    # About drug and protein (others)...
    drug_protein_path = data_dir + '/mat_data/mat_drug_protein.txt'
    drug_disease_path = data_dir + '/mat_data/mat_drug_disease.txt'
    drug_sideEffect_path = data_dir + '/mat_data/mat_drug_se.txt'
    protein_disease_path = data_dir + '/mat_data/mat_protein_disease.txt'

    # drug_drug and protein_protein combine the simNets and interactions
    # print('Load_Drug_Adj_Togerther ...')
    # drug_drug = Load_Adj_Togerther(dir_lists=[drug_drug_path, drug_drug_sim_chemical_path,
    #                                           drug_drug_sim_interaction_path, drug_drug_sim_se_path,
    #                                           drug_drug_sim_disease_path], ratio=0.01)
    #
    # print('Load_Protein_Adj_Togerther ...')
    # protein_protein = Load_Adj_Togerther(dir_lists=[protein_protein_path, protein_protein_sim_sequence_path,
    #                                                 protein_protein_sim_disease_path, protein_protein_sim_interaction_path],
    #                                      ratio=0.005)

    drug_drug = np.loadtxt(drug_drug_path, dtype=int)
    drug_protein = np.loadtxt(drug_protein_path, dtype=int)
    drug_disease = np.loadtxt(drug_disease_path, dtype=int)
    protein_protein = np.loadtxt(protein_protein_path, dtype=int)
    drug_sideEffect = np.loadtxt(drug_sideEffect_path, dtype=int)
    protein_disease = np.loadtxt(protein_disease_path, dtype=int)

    num_drug, num_protein = drug_protein.shape
    num_disease = drug_disease.shape[1]
    num_se = drug_sideEffect.shape[1]
    type_mask = get_type_mask(num_drug, num_protein, num_disease, num_se)
    np.save(save_prefix + 'node_types.npy', type_mask)

    for counter in range(num_repeats):
        print('\nThis is the {} repeat...'.format(counter))

        _adj_train = build_and_cache_for_split(
            'train', train_idx,
            drug_drug=drug_drug, drug_protein_full=drug_protein,
            drug_disease=drug_disease, drug_sideEffect=drug_sideEffect,
            protein_protein=protein_protein, protein_disease=protein_disease,
            num_drug=num_drug, num_protein=num_protein, num_disease=num_disease, num_se=num_se,
            expected_metapaths=expected_metapaths, counter=counter, save_prefix=save_prefix
        )

        _adj_valid = build_and_cache_for_split(
            'valid', valid_idx,
            drug_drug=drug_drug, drug_protein_full=drug_protein,
            drug_disease=drug_disease, drug_sideEffect=drug_sideEffect,
            protein_protein=protein_protein, protein_disease=protein_disease,
            num_drug=num_drug, num_protein=num_protein, num_disease=num_disease, num_se=num_se,
            expected_metapaths=expected_metapaths, counter=counter, save_prefix=save_prefix
        )

        _adj_test = build_and_cache_for_split(
            'test', test_idx,
            drug_drug=drug_drug, drug_protein_full=drug_protein,
            drug_disease=drug_disease, drug_sideEffect=drug_sideEffect,
            protein_protein=protein_protein, protein_disease=protein_disease,
            num_drug=num_drug, num_protein=num_protein, num_disease=num_disease, num_se=num_se,  # 注意实参别写错
            expected_metapaths=expected_metapaths, counter=counter, save_prefix=save_prefix
        )

