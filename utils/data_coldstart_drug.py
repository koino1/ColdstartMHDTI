import networkx as nx
import numpy as np
import scipy
import pickle
import json
from torch.utils.data import DataLoader, Dataset
import dgl
import torch
import os

def read_adjlist(fp):
    in_file = open(fp, 'r')
    adjlist = [line.strip() for line in in_file]
    in_file.close()
    return adjlist

def read_pickle(fp):
    in_file = open(fp, 'rb')
    idx = pickle.load(in_file)
    in_file.close()
    return idx

def read_json(fp):
    in_file = open(fp, 'r')
    idx = json.load(in_file)
    in_file.close()
    return idx

def fold_train_test_idx(pos_folds, neg_folds, nFold, foldID):
    train_pos_idx = []
    train_neg_idx = []
    test_fold_idx = []
    for fold in range(nFold):
        if fold == foldID:
            continue
        train_pos_idx.append(pos_folds['fold_' + str(fold)])
        train_neg_idx.append(neg_folds['fold_' + str(fold)])
    train_pos_idx = np.concatenate(train_pos_idx, axis=1)
    train_neg_idx = np.concatenate(train_neg_idx, axis=1)
    train_fold_idx = np.concatenate([train_pos_idx, train_neg_idx], axis=1)

    test_fold_idx.append(pos_folds['fold_' + str(foldID)])
    test_fold_idx.append(neg_folds['fold_' + str(foldID)])
    test_fold_idx = np.concatenate(test_fold_idx, axis=1)
    return train_fold_idx.T, test_fold_idx.T

def load_data(args, rp, train_test='train'):
    """
    适配新目录：
    {data_dir}/processed/repeat{rp}/{0|1}/{train|valid|test}/_{metapath}.{adjlist,idx}.pkl
    """
    # 构造前缀：注意 '_' 是目录
    prefix0 = os.path.join(args.data_dir, 'processed_coldstart_drug', f'repeat{rp}', '0', train_test)
    prefix1 = os.path.join(args.data_dir, 'processed_coldstart_drug', f'repeat{rp}', '1', train_test)

    # --- Drug 端 ---
    adjlist00 = read_pickle(os.path.join(prefix0, '_0-0.adjlist.pkl'))
    adjlist01 = read_pickle(os.path.join(prefix0, '_0-1-0.adjlist.pkl'))
    adjlist02 = read_pickle(os.path.join(prefix0, '_0-2-0.adjlist.pkl'))
    adjlist03 = read_pickle(os.path.join(prefix0, '_0-3-0.adjlist.pkl'))
    adjlist04 = read_pickle(os.path.join(prefix0, '_0-1-1-0.adjlist.pkl'))
    adjlist05 = read_pickle(os.path.join(prefix0, '_0-2-0-2-0.adjlist.pkl'))
    adjlist06 = read_pickle(os.path.join(prefix0, '_0-2-1-2-0.adjlist.pkl'))


    idx00 = read_pickle(os.path.join(prefix0, '_0-0.idx.pkl'))
    idx01 = read_pickle(os.path.join(prefix0, '_0-1-0.idx.pkl'))
    idx02 = read_pickle(os.path.join(prefix0, '_0-2-0.idx.pkl'))
    idx03 = read_pickle(os.path.join(prefix0, '_0-3-0.idx.pkl'))
    idx04 = read_pickle(os.path.join(prefix0, '_0-1-1-0.idx.pkl'))
    idx05 = read_pickle(os.path.join(prefix0, '_0-2-0-2-0.idx.pkl'))
    idx06 = read_pickle(os.path.join(prefix0, '_0-2-1-2-0.idx.pkl'))


    # --- Protein 端 ---
    adjlist10 = read_pickle(os.path.join(prefix1, '_1-1.adjlist.pkl'))
    adjlist11 = read_pickle(os.path.join(prefix1, '_1-0-1.adjlist.pkl'))
    adjlist12 = read_pickle(os.path.join(prefix1, '_1-2-1.adjlist.pkl'))
    adjlist13 = read_pickle(os.path.join(prefix1, '_1-0-0-1.adjlist.pkl'))
    adjlist14 = read_pickle(os.path.join(prefix1, '_1-2-1-2-1.adjlist.pkl'))
    adjlist15 = read_pickle(os.path.join(prefix1, '_1-2-0-2-1.adjlist.pkl'))

    idx10 = read_pickle(os.path.join(prefix1, '_1-1.idx.pkl'))
    idx11 = read_pickle(os.path.join(prefix1, '_1-0-1.idx.pkl'))
    idx12 = read_pickle(os.path.join(prefix1, '_1-2-1.idx.pkl'))
    idx13 = read_pickle(os.path.join(prefix1, '_1-0-0-1.idx.pkl'))
    idx14 = read_pickle(os.path.join(prefix1, '_1-2-1-2-1.idx.pkl'))
    idx15 = read_pickle(os.path.join(prefix1, '_1-2-0-2-1.idx.pkl'))

    return [[adjlist00, adjlist01, adjlist02, adjlist03, adjlist04, adjlist05, adjlist06],
            [adjlist10, adjlist11, adjlist12, adjlist13,adjlist14, adjlist15]], \
           [[idx00, idx01, idx02, idx03, idx04, idx05, idx06],
            [idx10, idx11, idx12, idx13, idx14, idx15]]



class mydataset(Dataset):
    def __init__(self, drug_protein_idx, y_true):
        self.drug_protein_idx = drug_protein_idx
        self.Y = y_true

    def __len__(self):
        return len(self.drug_protein_idx)

    def __getitem__(self, index):
        d_p_idx = self.drug_protein_idx[index].tolist()
        y = self.Y[index]

        return d_p_idx, y

class collate_fc(object):
    def __init__(self, adjlists, edge_metapath_indices_list, num_samples, offset, device):
        self.adjlists = adjlists
        self.edge_metapath_indices_list = edge_metapath_indices_list
        self.num_samples = num_samples
        self.offset = offset
        self.device = device

    def collate_func(self, batch_list):
        y_true = [y for _, y in batch_list]
        batch_list = [idx for idx, _ in batch_list]

        g_lists = [[], []]
        result_indices_lists = [[], []]
        idx_batch_mapped_lists = [[], []]
        for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(self.adjlists, self.edge_metapath_indices_list)):
            for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
                edges, result_indices, num_nodes, mapping = parse_adjlist([adjlist[row[mode]] for row in batch_list],
                                                                          [indices[row[mode]] for row in batch_list],
                                                                          self.num_samples, offset=self.offset, mode=mode)

                g = dgl.DGLGraph()

                g.add_nodes(num_nodes)
                if len(edges) > 0:
                    sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                    g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                    result_indices = torch.LongTensor(result_indices[sorted_index]).to(self.device)
                else:
                    result_indices = torch.LongTensor(result_indices).to(self.device)
                g = g.to(self.device)
                g_lists[mode].append(g)
                result_indices_lists[mode].append(result_indices)
                idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in batch_list]))

        return g_lists, result_indices_lists, idx_batch_mapped_lists, y_true, batch_list

def parse_adjlist(adjlist, edge_metapath_indices, samples=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping

def _load_feat_any(path):
    import torch, numpy as np
    if path.endswith(".npy"):
        arr = np.load(path)
        return torch.tensor(arr, dtype=torch.float32)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "emb" in obj:
        return obj["emb"].to(torch.float32)
    if isinstance(obj, torch.Tensor):
        return obj.to(torch.float32)
    raise TypeError(f"Unsupported feature file: {path} (expect .npy or .pt with key 'emb')")

def get_features(args, type_mask):
    features_list = []
    in_dims = []
    if args.feats_type == 0:
        for i in range(args.num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(args.device))
    elif args.feats_type == 1:
        for i in range(args.num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(args.device))

    elif args.feats_type == 2:
        # 只支持 .pt：Drug(type 0) / Protein(type 1) 从 .pt 文件加载，其余类型用 one-hot
        for i in range(args.num_ntype):
            num_nodes = int((type_mask == i).sum())
            if i == 0:
                if not hasattr(args, 'drug_feat_path') or args.drug_feat_path is None:
                    raise ValueError('feats_type=2 requires args.drug_feat_path for drug features (.pt)')
                obj = torch.load(args.drug_feat_path, map_location="cpu")
                if isinstance(obj, dict):
                    if "emb" not in obj:
                        raise KeyError(f"Drug .pt missing key 'emb'. Available keys: {list(obj.keys())}")
                    arr = obj["emb"]
                elif isinstance(obj, torch.Tensor):
                    arr = obj
                else:
                    raise TypeError(f"Drug features must be dict or Tensor, got {type(obj)}")
                if arr.dim() != 2:
                    raise ValueError(f"Drug features must be 2D [N, D], got shape {tuple(arr.shape)}")
                if arr.shape[0] != num_nodes:
                    raise ValueError(f'Drug feature rows ({arr.shape[0]}) != number of drug nodes ({num_nodes})')
                in_dims.append(int(arr.shape[1]))
                features_list.append(arr.to(args.device, dtype=torch.float32))
            elif i == 1:
                if not hasattr(args, 'protein_feat_path') or args.protein_feat_path is None:
                    raise ValueError('feats_type=2 requires args.protein_feat_path for protein features (.pt)')
                obj = torch.load(args.protein_feat_path, map_location="cpu")
                if isinstance(obj, dict):
                    if "emb" not in obj:
                        raise KeyError(f"Protein .pt missing key 'emb'. Available keys: {list(obj.keys())}")
                    arr = obj["emb"]
                elif isinstance(obj, torch.Tensor):
                    arr = obj
                else:
                    raise TypeError(f"Protein features must be dict or Tensor, got {type(obj)}")
                if arr.dim() != 2:
                    raise ValueError(f"Protein features must be 2D [N, D], got shape {tuple(arr.shape)}")
                if arr.shape[0] != num_nodes:
                    raise ValueError(f'Protein feature rows ({arr.shape[0]}) != number of protein nodes ({num_nodes})')
                in_dims.append(int(arr.shape[1]))
                features_list.append(arr.to(args.device, dtype=torch.float32))
            else:
                # 其他类型使用 one-hot 稀疏特征
                dim = num_nodes
                in_dims.append(dim)
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(args.device))

    elif args.feats_type == 3:
        # 四类都从文件加载；缺任意一个直接报错
        for i in range(args.num_ntype):
            raw = (type_mask == i).sum()
            num_nodes = int(raw.item()) if torch.is_tensor(raw) else int(raw)

            if i == 0:   # Drug
                if not getattr(args, 'drug_feat_path', None):
                    raise ValueError("feats_type=3 requires --drug_feat_path (no fallback)")
                arr = _load_feat_any(args.drug_feat_path)

            elif i == 1: # Protein
                if not getattr(args, 'protein_feat_path', None):
                    raise ValueError("feats_type=3 requires --protein_feat_path (no fallback)")
                arr = _load_feat_any(args.protein_feat_path)

            elif i == 2: # Disease
                if not getattr(args, 'disease_feat_path', None):
                    raise ValueError("feats_type=3 requires --disease_feat_path (no fallback)")
                arr = _load_feat_any(args.disease_feat_path)

            elif i == 3: # SideEffect
                if not getattr(args, 'side_feat_path', None):
                    raise ValueError("feats_type=3 requires --side_feat_path (no fallback)")
                arr = _load_feat_any(args.side_feat_path)

            else:
                raise ValueError(f"Unknown node type index {i} for feats_type=3")

            # 校验形状与节点数一致
            if arr.dim() != 2:
                raise ValueError(f"Type {i} features must be 2D [N,D], got {tuple(arr.shape)}")
            if arr.shape[0] != num_nodes:
                raise ValueError(
                    f"Type {i} rows ({arr.shape[0]}) != nodes ({num_nodes}); "
                    "请检查索引一致性或先重排特征行顺序"
                )

            in_dims.append(int(arr.shape[1]))
            features_list.append(arr.to(args.device, dtype=torch.float32))

    else:
        raise ValueError(f"Unknown feats_type: {args.feats_type}")

    return features_list, in_dims


