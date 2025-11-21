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

from utils.data_dataluo_change_metapath import load_data, fold_train_test_idx, mydataset, collate_fc, get_features
from model.MAGNN_dti_xiugai_transformer import MAGNN_lp
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import numpy as np, torch, os
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


def get_MSE(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_adj(dti):
    len_dti = len(dti)
    dpp_adj = np.zeros((len_dti, len_dti), dtype=int)
    for i, dpp1 in enumerate(dti):
        for j, dpp2 in enumerate(dti):
            if (dpp1[0] == dpp2[0]) | (dpp1[1] == dpp2[1]):
                dpp_adj[i][j] = 1
    return dpp_adj


def training(net, optimizer, train_loader, features_list, type_mask):
    net.train()
    train_loss = 0
    total = 0
    for i, (train_g_lists, train_indices_lists, train_idx_batch_mapped_lists, y_train, batch_list) in enumerate(
            train_loader):
        y_train = torch.tensor(y_train).long().to(features_list[0].device)
        adj_i = get_adj(batch_list)
        adj_i = torch.FloatTensor(adj_i).to(features_list[0].device)
        # forward
        output = net(
            (train_g_lists, features_list, type_mask, train_indices_lists, train_idx_batch_mapped_lists, adj_i))
        loss = F.nll_loss(torch.log(output), y_train)
        # loss = loss_bec(output[:, 1], y_train.float()) # the same to above

        # autograd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item() * len(y_train)
        total = total + len(y_train)

    return train_loss / total


def evaluate(net, test_loader, features_list, type_mask, _y_true_unused, threshold: float = 0.5):
    net.eval()
    pred_val = []
    y_true_s = []
    with torch.no_grad():
        for i, (val_g_lists, val_indices_lists, val_idx_batch_mapped_lists, y_true, batch_list) in enumerate(
                test_loader):
            adj_i = torch.FloatTensor(get_adj(batch_list)).to(features_list[0].device)
            output = net((val_g_lists, features_list, type_mask, val_indices_lists, val_idx_batch_mapped_lists, adj_i))
            pred_val.append(output)
            y_true_s.append(torch.tensor(y_true).long().to(features_list[0].device))

    val_pred = torch.cat(pred_val)  # (N, 2) 概率或 logits（按你模型的输出）
    y_true = torch.cat(y_true_s)

    # 原先写法（假定 val_pred 已是概率）：
    # val_loss = F.nll_loss(torch.log(val_pred), y_true)
    # 更稳妥一点（避免 log(0)）：
    eps = 1e-12
    val_prob = torch.clamp(val_pred, eps, 1 - eps)
    val_loss = F.nll_loss(torch.log(val_prob), y_true)

    val_prob_np = val_prob.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    pos_prob = val_prob_np[:, 1]
    y_pred_np = (pos_prob >= threshold).astype(int)

    acc = accuracy_score(y_true_np, y_pred_np)
    auc = roc_auc_score(y_true_np, pos_prob)
    aupr = average_precision_score(y_true_np, pos_prob)
    f1 = f1_score(y_true_np, y_pred_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_np)

    return val_loss, acc, auc, aupr, f1, mcc, val_prob_np


def testing(net, test_loader, features_list, type_mask, y_true_test, threshold: float = 0.5):
    net.eval()
    proba_list = []
    with torch.no_grad():
        for i, (test_g_lists, test_indices_lists, test_idx_batch_mapped_lists, y_test, batch_list) in enumerate(
                test_loader):
            adj_i = torch.FloatTensor(get_adj(batch_list)).to(features_list[0].device)
            output = net(
                (test_g_lists, features_list, type_mask, test_indices_lists, test_idx_batch_mapped_lists, adj_i))
            proba_list.append(output)

    y_proba_test = torch.cat(proba_list)
    eps = 1e-12
    y_proba_test = torch.clamp(y_proba_test, eps, 1 - eps)  # 若已是概率，避免log(0)风险
    y_proba_test = y_proba_test.cpu().numpy()

    pos_prob = y_proba_test[:, 1]
    y_pred = (pos_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true_test, pos_prob)
    aupr = average_precision_score(y_true_test, pos_prob)
    f1 = f1_score(y_true_test, y_pred)
    mcc = matthews_corrcoef(y_true_test, y_pred)

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
def save_embeddings_for_loader(net, data_loader, features_list, type_mask, save_dir, split: str, epoch):
    import os, numpy as np, torch
    out_dir = os.path.join(save_dir, 'embed')
    os.makedirs(out_dir, exist_ok=True)

    all_drug_idx, all_prot_idx = [], []
    all_drug_emb, all_prot_emb = [], []
    all_x_out = []
    all_x_cat = []  # 拼接后的联合向量 x（可选）
    all_z_joint = []  # 分类器倒数第二层的联合嵌入 z（128 维）

    net.eval()
    device = features_list[0].device
    with torch.no_grad():
        for (g_lists, indices_lists, idx_batch_mapped_lists, y_true, batch_list) in data_loader:
            adj_i = torch.as_tensor(get_adj(batch_list), dtype=torch.float32, device=device)

            # 你新的 encode_batch：返回 (h_user, h_item, x_cat, z_joint, logits)
            h_user, h_item, x_cat, z_joint, x_out = net.encode_batch(
                (g_lists, features_list, type_mask, indices_lists, idx_batch_mapped_lists, adj_i)
            )

            drug_idx = torch.tensor([pair[0] for pair in batch_list], device=device)
            prot_idx = torch.tensor([pair[1] for pair in batch_list], device=device)

            all_drug_idx.append(drug_idx.cpu())
            all_prot_idx.append(prot_idx.cpu())
            all_drug_emb.append(h_user.cpu())
            all_prot_emb.append(h_item.cpu())
            all_x_out.append(x_out.cpu())
            all_x_cat.append(x_cat.cpu())
            all_z_joint.append(z_joint.cpu())

    # 拼接为整体
    all_drug_idx = torch.cat(all_drug_idx).numpy() if all_drug_idx else np.empty((0,), dtype=int)
    all_prot_idx = torch.cat(all_prot_idx).numpy() if all_prot_idx else np.empty((0,), dtype=int)
    all_drug_emb = torch.cat(all_drug_emb).numpy() if all_drug_emb else np.empty((0, 0), dtype=float)
    all_prot_emb = torch.cat(all_prot_emb).numpy() if all_prot_emb else np.empty((0, 0), dtype=float)
    all_x_out = torch.cat(all_x_out).numpy() if all_x_out else np.empty((0, 0), dtype=float)
    all_x_cat = torch.cat(all_x_cat).numpy() if all_x_cat else np.empty((0, 0), dtype=float)
    all_z_joint = torch.cat(all_z_joint).numpy() if all_z_joint else np.empty((0, 0), dtype=float)

    np.savez(
        os.path.join(out_dir, f'{split}_embeddings_epoch{epoch}.npz'),
        drug_idx=all_drug_idx,
        protein_idx=all_prot_idx,
        drug_emb=all_drug_emb,
        protein_emb=all_prot_emb,
        x_out=all_x_out,  # 通常形状 (N, 2)
        joint_concat=all_x_cat,  # x = concat(h_user, h_item)
        joint_after_classifier=all_z_joint  # z（128 维）
    )


### -----------------------
def run_model(args):
    # 读取基础数据
    type_mask = np.load(os.path.join(args.data_dir, 'processed_coldstart_drug', 'node_types.npy'))
    drug_protein = np.loadtxt(os.path.join(args.data_dir, 'mat_data', 'mat_drug_protein.txt'), dtype=int)
    # 读取三份 split 的正负索引
    # 格式化 data_dir
    args.data_dir = args.data_dir.format(args.dataset)

    split_dir = "/home/yanghongyang/MHGNN-DTI/hetero_dataset/data_luo"
    train_pos = _read_pairs(os.path.join(split_dir, 'train_drug_coldstart.json'))
    train_neg = _read_pairs(os.path.join(split_dir, 'train_neg_drug_coldstart.json'))
    valid_pos = _read_pairs(os.path.join(split_dir, 'valid_drug_coldstart.json'))
    valid_neg = _read_pairs(os.path.join(split_dir, 'valid_neg_drug_coldstart.json'))
    test_pos  = _read_pairs(os.path.join(split_dir, 'test_drug_coldstart.json'))
    test_neg  = _read_pairs(os.path.join(split_dir, 'test_neg_drug_coldstart.json'))


    # 拼接索引并由真实矩阵取标签（正=1，负=0）
    train_idx, y_train = _stack_idx_and_labels(train_pos, train_neg, drug_protein)
    valid_idx, y_valid = _stack_idx_and_labels(valid_pos, valid_neg, drug_protein)
    test_idx, y_test = _stack_idx_and_labels(test_pos, test_neg, drug_protein)

    # 加载三份 split 的元路径缓存（与新版 load_data 签名一致）
    train_adjlists, train_edge_metapath_indices_list = load_data(args, rp=args.rp, train_test='train')
    valid_adjlists, valid_edge_metapath_indices_list = load_data(args, rp=args.rp, train_test='valid')
    test_adjlists, test_edge_metapath_indices_list = load_data(args, rp=args.rp, train_test='test')

    [num_metapaths_drug, num_metapaths_protein] = len(train_adjlists[0]), len(train_adjlists[1])

    # 数据集/加载器
    train_dataset = mydataset(train_idx, y_train)
    valid_dataset = mydataset(valid_idx, y_valid)
    test_dataset = mydataset(test_idx, y_test)

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
    # ==== 1) 准备参数桶 ====
    featproj_decay, trans_decay, fusion_decay, head_decay = [], [], [], []
    no_decay = []

    # 显式抓取所有 LayerNorm 参数（包括 TypeTower.ln）
    layernorm_param_ids = set()
    for m in net.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters(recurse=False):
                layernorm_param_ids.add(id(p))

    def is_no_decay(name: str, p: torch.nn.Parameter):
        name_l = name.lower()
        return (
                id(p) in layernorm_param_ids  # 任何 LayerNorm 参数都不做衰减
                or name_l.endswith("bias")  # bias 不衰减
                or ".pos_emb" in name_l  # 位置向量（如有）
                or "embedding" in name_l  # 词嵌入类参数（如有）
        )

    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if is_no_decay(n, p):
            no_decay.append(p)
            continue

        # 头部（分类器）
        if n.startswith("classifier."):
            head_decay.append(p)
            continue

        # TypeTower（fc_list）：只给 Linear 权重 decay（proj.weight / fc2.weight）
        if ".fc_list." in n:
            if n.endswith(".proj.weight") or n.endswith(".fc2.weight"):
                featproj_decay.append(p)
            else:
                # 其余（比如 fc bias、ln.*）已经在 is_no_decay 被拦住；走到这里的兜底也放 no_decay
                no_decay.append(p)
            continue

        # 语义融合层（若命名与你代码一致）
        if any(key in n for key in [
            "user_layer.fc1", "user_layer.fc2",
            "item_layer.fc1", "item_layer.fc2"
        ]):
            fusion_decay.append(p)
            continue

        # Transformer/MAGNN 内部注意力与前馈
        if any(key in n for key in [
            "self_mha", "cross_mha",
            "sa_ff", "ca_ff",
            "avg_to_heads", "inst_to_heads",
            "attn"
        ]):
            trans_decay.append(p)
            continue

        # 兜底：未归类的 backbone 参数归入 transformer decay
        head_decay.append(p)

    # ==== 2) 组装 param_groups ====
    param_groups = []
    if featproj_decay: param_groups.append({"params": featproj_decay, "weight_decay": 5e-4})
    if trans_decay:     param_groups.append({"params": trans_decay, "weight_decay": 1e-3})
    if fusion_decay:    param_groups.append({"params": fusion_decay, "weight_decay": 1e-3})
    if head_decay:      param_groups.append({"params": head_decay, "weight_decay": 1e-4})
    if no_decay:        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    # # Cosine LR (per-batch update)
    # total_updates = len(train_loader) * args.epoch
    # scheduler = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(args.lr) * 0.01)

    # === 训练/验证（把“test_loader 当 valid”改为 “用 valid_loader 做验证”）===
    os.makedirs(args.save_dir + '/checkpoint', exist_ok=True)
    best_acc = 0.0
    best_auc = 0.0
    best_aupr = 0.0
    pred = None
    counter = 0

    if args.only_test:
        # 只测试：直接在 test 上评估
        net.load_state_dict(torch.load(args.save_dir + '/checkpoint/checkpoint.pt', map_location=args.device))
        # 注意这里用 test_loader
        _, _, auc, aupr, y_pred = evaluate(net, test_loader, features_list, type_mask, None)
        best_auc, best_aupr, pred = auc, aupr, y_pred

    else:
        # 如有已有权重可先加载
        ckpt_path = args.save_dir + '/checkpoint/checkpoint.pt'
        if os.path.exists(ckpt_path):
            print(f'Load model weights from {ckpt_path}')
            net.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        for epoch in range(args.epoch):
            best_epoch = -1
            # training（沿用你的 training(...) 接口）
            train_loss = training(net, optimizer, train_loader, features_list, type_mask)

            # validation（这次用 valid_loader）
            val_loss, acc, auc, aupr, f1, mcc, y_pred = evaluate(net, valid_loader, features_list, type_mask, None)

            print(
                'Epoch {:d} | Train loss {:.6f} | acc {:.4f} | auc {:.4f} | aupr {:.4f} | f1 {:.4f} | mcc {:.4f}'.format(
                    epoch, train_loss, acc, auc, aupr, f1, mcc
                )
            )

            # early stopping criterion（保留你原有逻辑：AUPR/ACC 提升即保存）

            if (best_aupr < aupr) or (best_acc < acc):
                best_acc = acc
                best_auc = auc
                best_aupr, pred = aupr, y_pred
                torch.save(net.state_dict(), ckpt_path)
                counter = 0

                # 记录最佳 epoch
                best_epoch = epoch

                # 仅在出现“最优”时：保存训练/验证集的嵌入
                save_embeddings_for_loader(net, train_loader, features_list, type_mask, args.save_dir, split='train',
                                           epoch=best_epoch)
                save_embeddings_for_loader(net, valid_loader, features_list, type_mask, args.save_dir, split='valid',
                                           epoch=best_epoch)
            else:
                counter += 1

            if counter > args.patience:
                print('Early stopping!')
                break

        net.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        _, test_acc, test_auc, test_aupr, test_f1, test_mcc, _ = evaluate(
            net, test_loader, features_list, type_mask, None
        )
        save_embeddings_for_loader(net, test_loader, features_list, type_mask, args.save_dir, split='test',
                                   epoch=best_epoch if best_epoch >= 0 else 'final')
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
    ap.add_argument('--dataset', default='data_luo',
                    help='Dataset name, used to format data_dir and split_dir')

    ap.add_argument('--drug_feat_path',
                    default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data_luo/embeddings_drug.pt',
                    help='Path to drug feature file (.npy or .txt) when feats_type=2')
    ap.add_argument('--protein_feat_path',
                    default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data_luo/embeddings_prot.pt',
                    help='Path to protein feature file (.npy or .txt) when feats_type=2')
    ap.add_argument('--disease_feat_path',
                    default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data_luo/embeddings_dise.npy',  # ← 用你的实际路径替换
                    help='Path to disease feature file (.npy or .pt) when feats_type=2/3')
    ap.add_argument('--side_feat_path',
                    default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/data_luo/embeddings_se.npy',  # ← 用你的实际路径替换
                    help='Path to side-effect feature file (.npy or .pt) when feats_type=2/3')
    ap.add_argument('--hidden_dim', type=int, default=128, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn_vec_dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--attn_switch', type=bool, default=True,
                    help='attention considers the center node embedding or not')
    ap.add_argument('--rnn_type', default='self-attn',
                    help='Type of the aggregator. max-pooling, average, linear, neighbor, RotatE0.')
    ap.add_argument('--predictor', default='product_mlp', help='options: linear, product_mlp.')
    ap.add_argument('--semantic_fusion', default='attention',
                    help='options: concatenation, attention, max-pooling, average.')
    ap.add_argument('--epoch', type=int, default=200, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 5.')
    ap.add_argument('--batch_size', type=int, default=256, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num_ntype', default=4, type=int, help='Number of node types')
    ap.add_argument('--lr', default=0.0002)
    ap.add_argument('--weight_decay', default=1e-4)
    ap.add_argument('--dropout_rate', default=0.3)
    ap.add_argument('--num_workers', default=0, type=int)

    # ap.add_argument('--nFold', default=10, type=int)
    ap.add_argument('--neg_times', default=1, type=int, help='The ratio between positive samples and negative samples')
    # ap.add_argument('--data_dir', default='/yanghongyang/MHGNN-DTI/hetero_dataset/{}/')
    ap.add_argument('--data_dir', default='/home/yanghongyang/MHGNN-DTI/hetero_dataset/{}/',
                    help='Root dataset directory. May contain {} placeholder for --dataset.')

    args = ap.parse_args()

    ap.add_argument('--save_dir',
                    default='./results_dpp/{}/repeat{}/LLM_Drugstart_trans_1113_改变投影层_固定学习率/neg_times{}_{}_{}_{}_num_head{}_hidden_dim{}_batch_sz{}_LLM_{}'
                            '_predictor_{}',
                    help='Postfix for the saved model and result. Default is LastFM.')
    ap.add_argument('--only_test', default=False, type=bool)
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    args.dataset = 'data_luo'
    if args.dataset == 'data_luo':
        args.lr = 0.0002
    elif args.dataset == 'data':
        args.lr = 0.0002
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
