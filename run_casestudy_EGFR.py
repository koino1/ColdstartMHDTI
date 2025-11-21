import os
import sys
import time
import argparse
import numpy as np
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


from utils.data_casestudy_EGFR import load_data, fold_train_test_idx, mydataset, collate_fc, get_features
from model.MAGNN_dti_xiugai1 import MAGNN_lp
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch

from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef
)


loss_bec = nn.BCELoss()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

setup_seed(20)



def _predict_topk_by_ratio(pos_prob: np.ndarray, ratio: float):
    """
    让“预测为正”的样本占比 = ratio（例如 0.5）。
    做法：对分数从大到小排序，前 K=⌊N*ratio⌋ 个置为正。
    这样可避免分数并列导致的“超过/少于 K 个”的问题。
    返回：y_pred（二值向量）, kth_score（第K名的分数；仅供参考）
    """
    n = len(pos_prob)
    K = int(np.floor(n * ratio))
    y_pred = np.zeros(n, dtype=int)
    if K <= 0:
        return y_pred, float("-inf")
    idx_desc = np.argsort(-pos_prob)  # 从大到小
    top_idx = idx_desc[:K]
    y_pred[top_idx] = 1
    kth_score = pos_prob[top_idx[-1]]
    return y_pred, float(kth_score)

def _gather_pairs(batch_list):
    """将 batch_list 中的 (drug_idx, protein_idx) 配对转为 (N,2) numpy 数组。"""
    import numpy as _np, torch as _torch
    if isinstance(batch_list, (list, tuple)):
        return _np.array(batch_list, dtype=int)
    if _torch.is_tensor(batch_list):
        return batch_list.detach().cpu().numpy().astype(int)
    raise TypeError(f"Unsupported batch_list type: {type(batch_list)}")

def save_scores_csv(save_dir, split, epoch, pairs_np, y_true_np, y_proba_np):
    """
    保存逐样本分数到 CSV：
    y_proba_np 形状 (N,2)，[:,0] 为负类概率，[:,1] 为正类概率。
    """
    import os, pandas as pd
    out_dir = os.path.join(save_dir, "scores")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        "drug_index":   pairs_np[:, 0],
        "protein_index":pairs_np[:, 1],
        "y_true":       y_true_np.astype(int),
        "y_score":      y_proba_np[:, 1],
        "y_score_neg":  y_proba_np[:, 0],
        "epoch":        [epoch] * len(y_true_np),
        "split":        [split] * len(y_true_np),
    })
    fn = "test_scores.csv" if split == "test" else f"valid_scores_epoch{epoch}.csv"
    df.to_csv(os.path.join(out_dir, fn), index=False)


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)

def get_MSE(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def get_adj(dti):
    len_dti = len(dti)
    dpp_adj = np.zeros((len_dti, len_dti), dtype=int)
    for i, dpp1 in enumerate(dti):
        for j, dpp2 in enumerate(dti):
            if (dpp1[0] == dpp2[0]) | (dpp1[1] == dpp2[1]):
                dpp_adj[i][j] = 1
    return dpp_adj


def training(net, optimizer, train_loader, features_list, type_mask, scheduler=None):
    net.train()
    train_loss = 0
    total = 0
    for i, (train_g_lists, train_indices_lists, train_idx_batch_mapped_lists, y_train, batch_list) in enumerate(train_loader):
        y_train = torch.tensor(y_train).long().to(features_list[0].device)
        adj_i = get_adj(batch_list)
        adj_i = torch.FloatTensor(adj_i).to(features_list[0].device)
        # forward
        output = net((train_g_lists, features_list, type_mask, train_indices_lists, train_idx_batch_mapped_lists, adj_i))
        eps = 1e-12
        output = torch.clamp(output, eps, 1.0 - eps)  #
        loss = F.nll_loss(torch.log(output), y_train)
        # loss = loss_bec(output[:, 1], y_train.float()) # the same to above

        # autograd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item() * len(y_train)
        total = total + len(y_train)

    return train_loss/total

def evaluate(net, loader, features_list, type_mask,
             y_true_unused=None, threshold: float = 0.1,
             topk_ratio: float = None,
             split: str = "valid",                 # "valid" 或 "test"
             save_scores: bool = False,            # 是否保存逐样本分数
             save_dir: str = None, epoch="final"): # 保存所需信息
    net.eval()
    prob_chunks, ytrue_chunks, pairs_chunks = [], [], []
    with torch.no_grad():
        for (g_lists, indices_lists, idx_batch_mapped_lists, y_true, batch_list) in loader:
            adj_i = torch.FloatTensor(get_adj(batch_list)).to(features_list[0].device)
            out = net((g_lists, features_list, type_mask, indices_lists, idx_batch_mapped_lists, adj_i))
            # 若是 logits，就转概率；若已是概率，clamp 即可
            if out.dim() == 2 and (out.max() > 1.0 or out.min() < 0.0):
                out = torch.softmax(out, dim=1)
            prob_chunks.append(out)
            ytrue_chunks.append(torch.tensor(y_true).long().to(features_list[0].device))
            pairs_chunks.append(_gather_pairs(batch_list))

    probs = torch.cat(prob_chunks, dim=0)  # (N,2)
    y_true_t = torch.cat(ytrue_chunks, dim=0)

    eps = 1e-12
    probs = torch.clamp(probs, eps, 1 - eps)
    val_loss = F.nll_loss(torch.log(probs), y_true_t)

    y_proba_np = probs.detach().cpu().numpy()   # (N,2)
    y_true_np  = y_true_t.detach().cpu().numpy()
    pairs_np   = np.concatenate(pairs_chunks, axis=0) if len(pairs_chunks) else np.empty((0,2), dtype=int)

    pos_prob = y_proba_np[:, 1]
    if topk_ratio is not None:
        y_pred, thr_used = _predict_topk_by_ratio(pos_prob, topk_ratio)
    else:
        y_pred = (pos_prob >= threshold).astype(int)
        thr_used = float(threshold)  # 仅供你调试打印

    acc  = accuracy_score(y_true_np, y_pred)
    auc  = roc_auc_score(y_true_np, pos_prob)
    aupr = average_precision_score(y_true_np, pos_prob)
    f1   = f1_score(y_true_np, y_pred)
    mcc  = matthews_corrcoef(y_true_np, y_pred)

    # 可选：落盘逐样本 CSV
    if save_scores and save_dir is not None:
        save_scores_csv(save_dir, split=split, epoch=epoch,
                        pairs_np=pairs_np, y_true_np=y_true_np, y_proba_np=y_proba_np)

    # 兼容你原有返回，并额外带 acc/概率/配对
    return val_loss, acc, auc, aupr, f1, mcc, y_proba_np, y_true_np, pairs_np



def save_embeddings_for_loader(net, data_loader, features_list, type_mask, save_dir, split: str, epoch):
    """
    在 data_loader 上前向一次，使用 net.encode_batch(...) 拿到 (drug_emb, protein_emb)，
    并把索引与嵌入保存为 {save_dir}/embed/{split}_embeddings_epoch{epoch}.npz
    """
    import numpy as np, torch, os
    out_dir = os.path.join(save_dir, 'embed')
    os.makedirs(out_dir, exist_ok=True)

    all_drug_idx, all_prot_idx = [], []
    all_drug_emb, all_prot_emb = [], []

    device = features_list[0].device if hasattr(features_list[0], 'device') else torch.device('cpu')

    net.eval()
    with torch.no_grad():
        for (g_lists, indices_lists, idx_batch_mapped_lists, y_true, batch_list) in data_loader:
            adj_i = torch.FloatTensor(get_adj(batch_list)).to(device)
            # 期望模型提供 net.encode_batch(...) -> (h_user, h_item)
            h_user, h_item = net.encode_batch((g_lists, features_list, type_mask, indices_lists, idx_batch_mapped_lists, adj_i))

            drug_idx = torch.tensor([pair[0] for pair in batch_list], device=h_user.device)
            prot_idx = torch.tensor([pair[1] for pair in batch_list], device=h_item.device)

            all_drug_idx.append(drug_idx.cpu())
            all_prot_idx.append(prot_idx.cpu())
            all_drug_emb.append(h_user.detach().cpu())
            all_prot_emb.append(h_item.detach().cpu())

    if all_drug_idx:
        all_drug_idx = torch.cat(all_drug_idx).numpy()
        all_prot_idx = torch.cat(all_prot_idx).numpy()
        all_drug_emb = torch.cat(all_drug_emb).numpy()
        all_prot_emb = torch.cat(all_prot_emb).numpy()
    else:
        all_drug_idx = np.empty((0,), dtype=int)
        all_prot_idx = np.empty((0,), dtype=int)
        all_drug_emb = np.empty((0, 0), dtype=float)
        all_prot_emb = np.empty((0, 0), dtype=float)

    np.savez(os.path.join(out_dir, f'{split}_embeddings_epoch{epoch}.npz'),
             drug_idx=all_drug_idx, protein_idx=all_prot_idx,
             drug_emb=all_drug_emb, protein_emb=all_prot_emb)



def testing(net, test_loader, features_list, type_mask, y_true_test, threshold: float = 0.7):
    net.eval()
    proba_list = []
    with torch.no_grad():
        for i, (test_g_lists, test_indices_lists, test_idx_batch_mapped_lists, y_test, batch_list) in enumerate(test_loader):
            adj_i = torch.FloatTensor(get_adj(batch_list)).to(features_list[0].device)
            output = net((test_g_lists, features_list, type_mask, test_indices_lists, test_idx_batch_mapped_lists, adj_i))
            proba_list.append(output)

    y_proba_test = torch.cat(proba_list)
    eps = 1e-12
    y_proba_test = torch.clamp(y_proba_test, eps, 1 - eps)  # 若已是概率，避免log(0)风险
    y_proba_test = y_proba_test.cpu().numpy()

    pos_prob = y_proba_test[:, 1]
    y_pred   = (pos_prob >= threshold).astype(int)

    auc  = roc_auc_score(y_true_test, pos_prob)
    aupr = average_precision_score(y_true_test, pos_prob)
    f1   = f1_score(y_true_test, y_pred)
    mcc  = matthews_corrcoef(y_true_test, y_pred)

    return auc, aupr, f1, mcc, y_true_test, y_proba_test


def _read_pairs(json_path: str) -> np.ndarray:
    with open(json_path, 'r') as f:
        arr = np.array(json.load(f), dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f'Bad pair shape in {json_path}: expect (N,2), got {arr.shape}')
    return arr

def _stack_idx_and_labels(pos_idx: np.ndarray, neg_idx: np.ndarray, mat_drug_protein: np.ndarray):
    all_idx = np.vstack([pos_idx, neg_idx])
    y_true = mat_drug_protein[all_idx[:, 0], all_idx[:, 1]].astype(np.int64)  # 正=1, 负=0
    return all_idx, y_true

### -----------------------
def run_model(args):
    # 读取基础数据
    type_mask = np.load(os.path.join(args.data_dir, 'processed_case_study_EGFR', 'node_types.npy'))
    drug_protein = np.loadtxt(os.path.join(args.data_dir, 'mat_data', 'mat_drug_protein.txt'), dtype=int)
    # 读取三份 split 的正负索引
    # 格式化 data_dir
    args.data_dir = args.data_dir.format(args.dataset)

    split_dir = "/home/yanghongyang/MHGNN-DTI/hetero_dataset/data"
    train_pos = _read_pairs(os.path.join(split_dir, 'train_case_study_EGFR.json'))
    train_neg = _read_pairs(os.path.join(split_dir, 'train_case_study_EGFR_neg.json'))
    valid_pos = _read_pairs(os.path.join(split_dir, 'valid_case_study_EGFR.json'))
    valid_neg = _read_pairs(os.path.join(split_dir, 'valid_case_study_EGFR_neg.json'))
    test_pos  = _read_pairs(os.path.join(split_dir, 'test_case_study_EGFR.json'))
    test_neg  = _read_pairs(os.path.join(split_dir, 'test_case_study_EGFR_neg.json'))

    # 拼接索引并由真实矩阵取标签（正=1，负=0）
    train_idx, y_train = _stack_idx_and_labels(train_pos, train_neg, drug_protein)
    valid_idx, y_valid = _stack_idx_and_labels(valid_pos, valid_neg, drug_protein)
    test_idx, y_test = _stack_idx_and_labels(test_pos, test_neg, drug_protein)

    # 加载三份 split 的元路径缓存（与新版 load_data 签名一致）
    train_adjlists, train_edge_metapath_indices_list = load_data(args, rp=args.rp, train_test='train')
    valid_adjlists, valid_edge_metapath_indices_list = load_data(args, rp=args.rp, train_test='valid')
    test_adjlists,  test_edge_metapath_indices_list  = load_data(args, rp=args.rp, train_test='test')


    [num_metapaths_drug, num_metapaths_protein] = len(train_adjlists[0]), len(train_adjlists[1])

    # 数据集/加载器
    train_dataset = mydataset(train_idx, y_train)
    valid_dataset = mydataset(valid_idx, y_valid)
    test_dataset  = mydataset(test_idx,  y_test)

    num_drug = drug_protein.shape[0]  # offset=药物节点数
    train_collate = collate_fc(train_adjlists, train_edge_metapath_indices_list,
                               num_samples=args.samples, offset=num_drug, device=args.device)
    valid_collate = collate_fc(valid_adjlists, valid_edge_metapath_indices_list,
                               num_samples=args.samples, offset=num_drug, device=args.device)
    test_collate = collate_fc(test_adjlists, test_edge_metapath_indices_list,
                              num_samples=args.samples, offset=num_drug, device=args.device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False,
                              collate_fn=train_collate.collate_func)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False,
                              collate_fn=valid_collate.collate_func)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False,
                             collate_fn=test_collate.collate_func)

#####
    features_list, in_dims = get_features(args, type_mask)
    print("=== Features list info ===")
    for i, feat in enumerate(features_list):
        # 稀疏张量没有 .size()，所以要区分
        if feat.is_sparse:
            print(f"Node type {i}: sparse tensor, size = {feat.shape}")
        else:
            print(f"Node type {i}: dense tensor, size = {feat.size()}")

    net = MAGNN_lp([num_metapaths_drug, num_metapaths_protein], args.num_etypes, args.etypes_lists, in_dims,
                   args.hidden_dim, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                   args.dropout_rate, args.attn_switch, args).to(args.device)

    head_params = list(net.classifier.parameters())  # 头部
    backbone_params = [p for n, p in net.named_parameters()
                       if not n.startswith('classifier.')]  # 除头部外的其它参数

    # 可选：检查不重复
    head_ids = {id(p) for p in head_params}
    assert not any(id(p) in head_ids for p in backbone_params)

    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR (per-batch update)
    total_updates = len(train_loader) * args.epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(args.lr) * 0.01)

    # === 训练/验证（把“test_loader 当 valid”改为 “用 valid_loader 做验证”）===
    os.makedirs(args.save_dir + '/checkpoint', exist_ok=True)
    best_acc = 0.0
    best_auc = 0.0
    best_aupr = 0.0
    pred = None
    counter = 0

    if args.only_test:
        # 只测试：直接在 test 上评估
        ckpt_path = os.path.join(args.save_dir, "checkpoint", "checkpoint.pt")
        net.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        # 用新的 evaluate：返回 acc/auc/aupr/f1/mcc + 逐样本概率与配对
        _, test_acc, test_auc, test_aupr, test_f1, test_mcc, y_proba_test, y_true_test, pairs_test = evaluate(
            net, test_loader, features_list, type_mask,
            split="test", save_scores=True, save_dir=args.save_dir, epoch="final",
            topk_ratio=0.5  # ← 前 50% 判正
        )
        # 为了兼容你后续代码中的变量名：
        best_auc, best_aupr, pred = test_auc, test_aupr, y_proba_test  # pred 现在是 (N,2) 概率矩阵

        # （可选）同时保存测试集 embedding，保持原脚本行为
        save_embeddings_for_loader(
            net, test_loader, features_list, type_mask,
            args.save_dir, split="test", epoch="final"
        )
        print(
            f"[test] ACC={test_acc:.4f}, AUC={test_auc:.4f}, AUPR={test_aupr:.4f}, F1={test_f1:.4f}, MCC={test_mcc:.4f}")

    else:
        # 如有已有权重可先加载
        ckpt_path = args.save_dir + '/checkpoint/checkpoint.pt'
        if os.path.exists(ckpt_path):
            print(f'Load model weights from {ckpt_path}')
            net.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        for epoch in range(args.epoch):
            best_epoch = -1
            # training（沿用你的 training(...) 接口）
            train_loss = training(net, optimizer, train_loader, features_list, type_mask, scheduler)

            # validation（这次用 valid_loader）

            val_loss, acc, auc, aupr, f1, mcc, y_proba_val, y_true_val, pairs_val = evaluate(
                net, valid_loader, features_list, type_mask,
                split="valid", save_scores=False,
                topk_ratio=0.5  # ← 前 50% 判正
            )

            print(
                'Epoch {:d} | Train loss {:.6f} | Val loss {:.6f} | acc {:.4f} | auc {:.4f} | aupr {:.4f} | f1 {:.4f} | mcc {:.4f}'.format(
                    epoch, train_loss, val_loss, acc, auc, aupr, f1, mcc
                )
            )

            # early stopping criterion（保留你原有逻辑：AUPR 提升即保存）

            min_delta = 1e-4
            if (aupr > best_aupr + min_delta):
                best_acc = acc  # 可留作记录（不参与判定）
                best_auc = auc
                best_aupr, pred = aupr, y_proba_val
                torch.save(net.state_dict(), ckpt_path)
                counter = 0

                # 记录最佳 epoch
                best_epoch = epoch

                best_epoch = epoch
                save_embeddings_for_loader(net, train_loader, features_list, type_mask, args.save_dir, split='train',
                                           epoch=best_epoch)
                save_embeddings_for_loader(net, valid_loader, features_list, type_mask, args.save_dir, split='valid',
                                           epoch=best_epoch)
                save_scores_csv(args.save_dir, split="valid", epoch=best_epoch,
                                pairs_np=pairs_val, y_true_np=y_true_val, y_proba_np=y_proba_val)
            else:
                counter += 1

            # if scheduler is not None:
            #     scheduler.step()  # ← 放在 epoch 末尾

            if counter > args.patience:
                print('Early stopping!')
                break

        net.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        # —— 评测+保存 test_scores.csv
        _, test_acc, test_auc, test_aupr, test_f1, test_mcc, y_proba_test, y_true_test, pairs_test = evaluate(
            net, test_loader, features_list, type_mask,
            split="test", save_scores=True, save_dir=args.save_dir, epoch="final",
            topk_ratio=0.5  # ← 前 50% 判正
        )

        # —— 保存测试集 embedding（延续你原来的函数）
        save_embeddings_for_loader(
            net, test_loader, features_list, type_mask,
            args.save_dir, split="test", epoch="final"
        )

        # —— 打印（保持你想要的格式）
        print(
            f"[test] ACC={test_acc:.4f}, AUC={test_auc:.4f}, AUPR={test_aupr:.4f}, F1={test_f1:.4f}, MCC={test_mcc:.4f}")

        # 结果落盘（改成 valid/test 两行更清晰）
        os.makedirs(args.save_dir, exist_ok=True)
        with open(args.save_dir + '/results.csv', 'a') as f:
            if os.stat(args.save_dir + '/results.csv').st_size == 0:
                f.write('Split,AUC,AUPR\n')
            f.write(','.join(map(str, ['valid', best_auc, best_aupr])) + '\n')
            f.write(','.join(map(str, ['test', test_auc, test_aupr])) + '\n')

        # 同时保留一个“best”快照（可选）
        torch.save(net.state_dict(), args.save_dir + '/checkpoint/checkpoint_best.pt')

        # 也把测试集的结果保存成 json（字段名与原逻辑类似）
        results = {
            # 'pred': test_pred.tolist(),
            # 'ground_truth': y_test.tolist(),
            'AUC': float(test_auc),
            'AUPR': float(test_aupr),
        }
        json.dump(results, open(os.path.join(args.save_dir, 'test_pred_results.json'), 'w'))


# Params
def parser():
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--feats_type', type=int, default=3,
                    help='Type of node features: 0 - all one-hot (id) vectors; 1 - all zero 10-d; 2 - drug/protein custom features and others one-hot. Default is 0.')
    ap.add_argument('--dataset', default='data',
                    help='Dataset name, used to format data_dir and split_dir')

    ap.add_argument('--drug_feat_path', default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data/embeddings_drug.pt', help='Path to drug feature file (.npy or .txt) when feats_type=2')
    ap.add_argument('--protein_feat_path', default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data/embeddings_prot.pt', help='Path to protein feature file (.npy or .txt) when feats_type=2')
    ap.add_argument('--disease_feat_path',
                    default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data/embeddings_dise.npy',  # ← 用你的实际路径替换
                    help='Path to disease feature file (.npy or .pt) when feats_type=2/3')
    ap.add_argument('--side_feat_path',
                    default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data/embeddings_se.npy',  # ← 用你的实际路径替换
                    help='Path to side-effect feature file (.npy or .pt) when feats_type=2/3')
    ap.add_argument('--hidden_dim', type=int, default=128, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn_vec_dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--attn_switch', type=bool, default=True, help='attention considers the center node embedding or not')
    ap.add_argument('--rnn_type', default='average', help='Type of the aggregator. max-pooling, average, linear, neighbor, RotatE0.')
    ap.add_argument('--predictor', default='gcn', help='options: linear, gcn.')
    ap.add_argument('--semantic_fusion', default='attention', help='options: concatenation, attention, max-pooling, average.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 5.')
    ap.add_argument('--batch_size', type=int, default=256, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num_ntype', default=4, type=int, help='Number of node types')
    ap.add_argument('--lr', default=0.0005)
    ap.add_argument('--weight_decay', default=1e-4)
    ap.add_argument('--dropout_rate', default=0.3)
    ap.add_argument('--num_workers', default=0, type=int)

    # ap.add_argument('--nFold', default=10, type=int)
    ap.add_argument('--neg_times', default=1, type=int, help='The ratio between positive samples and negative samples')
    # ap.add_argument('--data_dir', default='/yanghongyang/MHGNN-DTI/hetero_dataset/{}/')
    ap.add_argument('--data_dir', default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/{}/',
                    help='Root dataset directory. May contain {} placeholder for --dataset.')
    # ap.add_argument('--train_file', default='train_pairs.json', help='Training set file name')
    # ap.add_argument('--valid_file', default='valid_pairs.json', help='Validation set file name')
    # ap.add_argument('--test_file', default='test_pairs.json', help='Test set file name')
    # 负样本文件名
    # ap.add_argument('--train_neg_file', default='train_neg.json')
    # ap.add_argument('--valid_neg_file', default='valid_neg.json')
    # ap.add_argument('--test_neg_file',  default='test_neg.json')

    args = ap.parse_args()

    ap.add_argument('--save_dir',
                    default='./results_dpp/{}/repeat{}/LLM_datasetB_casestudy_EGFR/neg_times{}_{}_{}_{}_num_head{}_hidden_dim{}_batch_sz{}_LLM_{}'
                            '_predictor_{}',
                    help='Postfix for the saved model and result. Default is LastFM.')
    ap.add_argument('--only_test', default=False, type=bool)
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parser()
    args.dataset = 'data'
    args.data_dir = args.data_dir.format(args.dataset)
    # args.save_dir = args.save_dir.format(args.dataset)
    # make_dir(args.save_dir)

    etypes_lists = [
        [[None], [0, 1], [2, 3], [4, 5], [0, None, 1], [2, 3, 2, 3], [2, 7, 6, 3]],
        # [0, 1, 0, 1], [2, 3, 2, 3], [4, 5, 4, 5], [0, 6, 7, 1], [2, 7, 6, 3]],
        [[None], [1, 0], [6, 7], [1, None, 0], [6, 7, 6, 7], [6, 3, 2, 7]]
        # , [1, 0, 1, 0], [1, 2, 3, 0], [6, 3, 2, 7], [6, 7, 6, 7]]
    ]

    expected_metapaths = [
        [(0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 1, 1, 0), (0, 2, 0, 2, 0), (0, 2, 1, 2, 0)],
        # (0, 1, 0, 1, 0), (0, 2, 0, 2, 0), (0, 3, 0, 3, 0), (0, 1, 2, 1, 0), (0, 2, 1, 2, 0)],
        [(1, 1), (1, 0, 1), (1, 2, 1), (1, 0, 0, 1), (1, 2, 1, 2, 1), (1, 2, 0, 2, 1)],
        # (1, 0, 1, 0, 1), (1, 0, 2, 0, 1), (1, 2, 0, 2, 1), (1, 2, 1, 2, 1)]
    ]


    args.etypes_lists = etypes_lists
    args.num_etypes = 8
    args.expected_metapaths = expected_metapaths

    for rp in range(args.repeat):
        print('This is repeat ', rp)
        args.rp = rp
        save_dir = args.save_dir
        args.save_dir = args.save_dir.format(args.dataset, args.rp, args.neg_times, args.rnn_type.capitalize(),
                                             len(args.expected_metapaths[0]), len(args.expected_metapaths[1]),
                                             args.num_heads, args.hidden_dim, args.batch_size,
                                             args.semantic_fusion, args.predictor)
        print('Save path ', args.save_dir)
        make_dir(args.save_dir)
        sys.stdout = Logger(args.save_dir + 'log.txt')

        run_model(args)

        print('Save path ', args.save_dir)
        args.save_dir = save_dir