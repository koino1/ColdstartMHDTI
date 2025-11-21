#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DistMult KGE pretraining with full coverage for diseases/side effects, optional extra relations.

Required index-based relation files (two columns: idx0, idx1):
  --drug_se PATH                 # (drug_idx, side_idx)
  --drug_disease PATH            # (drug_idx, disease_idx)
  --protein_disease PATH         # (protein_idx, disease_idx)

Optional extra relations you asked to add:
  --drug_drug PATH               # (drug_idx, drug_idx)
  --protein_protein PATH         # (protein_idx, protein_idx)
  --drug_protein PATH            # (drug_idx, protein_idx)  [CAUTION: exclude eval DTI edges to avoid leakage]

Full coverage of Disease/SideEffect:
  Provide either --disease_list/--sideeffect_list (one name per line), or --n_disease/--n_side.

Imputation for isolated diseases/side effects (never appear in any triple):
  --fill_strategy {keep_random,global_mean,gaussian}  [default: gaussian]

Outputs in --out_dir:
  - disease_embeddings.npy, sideeffect_embeddings.npy  (rows aligned to indices 0..N-1)
  - disease_entities.tsv, sideeffect_entities.tsv      (names aligned to rows if lists are provided)
  - entity_embeddings.npy, relation_embeddings.npy, entities.tsv, relations.tsv
"""

import argparse, os, random
from typing import List, Tuple, Dict, Sequence, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_index_edges(path: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    if path is None: return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    sep = "\t" if path.endswith(".tsv") else ","
    try:
        df = pd.read_csv(path, sep=sep, header=None, usecols=[0,1])
    except Exception:
        df = pd.read_csv(path, sep=sep)
        df = df.iloc[:, :2]
    df = df.dropna()
    # enforce integer
    df.iloc[:,0] = pd.to_numeric(df.iloc[:,0], errors="coerce").astype("Int64")
    df.iloc[:,1] = pd.to_numeric(df.iloc[:,1], errors="coerce").astype("Int64")
    df = df.dropna().astype(int).drop_duplicates().reset_index(drop=True)
    return list(zip(df.iloc[:,0].tolist(), df.iloc[:,1].tolist()))

def read_list(path: Optional[str]) -> Optional[List[str]]:
    if path is None: return None
    names = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            names.append(line.rstrip("\n\r"))
    return names

# ----------------------------
# Build triples & mappings
# ----------------------------
def build_triples_from_indices(
    drug_se: List[Tuple[int,int]],
    drug_disease: List[Tuple[int,int]],
    protein_disease: List[Tuple[int,int]],
    drug_drug: Optional[List[Tuple[int,int]]] = None,
    protein_protein: Optional[List[Tuple[int,int]]] = None,
    drug_protein: Optional[List[Tuple[int,int]]] = None,
    n_disease: Optional[int] = None,
    n_side: Optional[int] = None,
    disease_list: Optional[List[str]] = None,
    side_list: Optional[List[str]] = None,
    add_inverse: bool = False
):
    triples = []
    D = lambda d: f"Drug:{int(d)}"
    P = lambda p: f"Protein:{int(p)}"
    Z = lambda z: f"Disease:{int(z)}"
    S = lambda s: f"SideEffect:{int(s)}"

    # Core relations
    for d, s in drug_se:         triples.append((D(d), "causes",          S(s)))
    for d, z in drug_disease:    triples.append((D(d), "hasIndication",   Z(z)))
    for p, z in protein_disease: triples.append((P(p), "associatedWith",  Z(z)))

    # Extra relations (optional)
    if drug_drug:
        for d1, d2 in drug_drug:
            triples.append((D(d1), "drugDrug", D(d2)))
    if protein_protein:
        for p1, p2 in protein_protein:
            triples.append((P(p1), "proteinProtein", P(p2)))
    if drug_protein:
        for d, p in drug_protein:
            triples.append((D(d), "drugProtein", P(p)))

    if add_inverse:
        inv = []
        for h, r, t in triples:
            inv.append((t, f"{r}_inv", h))
        triples += inv

    # maps
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    def get_id(dct, k):
        if k not in dct: dct[k] = len(dct)
        return dct[k]

    triples_idx = []
    for h, r, t in triples:
        hid = get_id(ent2id, h)
        rid = get_id(rel2id, r)
        tid = get_id(ent2id, t)
        triples_idx.append((hid, rid, tid))

    # determine totals for full coverage of Disease & Side
    max_d_csv = -1
    if drug_disease:    max_d_csv = max(max_d_csv, max(z for _,z in drug_disease))
    if protein_disease: max_d_csv = max(max_d_csv, max(z for _,z in protein_disease))
    max_s_csv = max([s for _,s in drug_se]) if drug_se else -1

    if n_disease is None:
        n_disease_total = len(disease_list) if disease_list is not None else (max_d_csv + 1 if max_d_csv >= 0 else 0)
    else:
        n_disease_total = n_disease

    if n_side is None:
        n_side_total = len(side_list) if side_list is not None else (max_s_csv + 1 if max_s_csv >= 0 else 0)
    else:
        n_side_total = n_side

    # Expand to ensure Disease:0..N-1 and SideEffect:0..N-1 exist
    for i in range(n_disease_total): _ = get_id(ent2id, Z(i))
    for i in range(n_side_total):    _ = get_id(ent2id, S(i))

    # Build type buckets after full coverage
    drug_ids, prot_ids, disease_ids, side_ids = [], [], [], []
    for e, eid in ent2id.items():
        if e.startswith("Drug:"):       drug_ids.append(eid)
        elif e.startswith("Protein:"):  prot_ids.append(eid)
        elif e.startswith("Disease:"):  disease_ids.append(eid)
        elif e.startswith("SideEffect:"): side_ids.append(eid)

    # typed negative sampling ranges
    rel2types: Dict[int, Tuple[Sequence[int], Sequence[int]]] = {}
    for r, rid in rel2id.items():
        if r == "hasIndication":
            rel2types[rid] = (drug_ids, disease_ids)
        elif r == "causes":
            rel2types[rid] = (drug_ids, side_ids)
        elif r == "associatedWith":
            rel2types[rid] = (prot_ids, disease_ids)
        elif r == "drugDrug":
            rel2types[rid] = (drug_ids, drug_ids)
        elif r == "proteinProtein":
            rel2types[rid] = (prot_ids, prot_ids)
        elif r == "drugProtein":
            rel2types[rid] = (drug_ids, prot_ids)
        elif r.endswith("_inv"):
            base = r[:-4]
            if base == "hasIndication":
                rel2types[rid] = (disease_ids, drug_ids)
            elif base == "causes":
                rel2types[rid] = (side_ids, drug_ids)
            elif base == "associatedWith":
                rel2types[rid] = (disease_ids, prot_ids)
            elif base == "drugProtein":
                rel2types[rid] = (prot_ids, drug_ids)
            elif base == "drugDrug":
                rel2types[rid] = (drug_ids, drug_ids)
            elif base == "proteinProtein":
                rel2types[rid] = (prot_ids, prot_ids)
            else:
                rel2types[rid] = (list(ent2id.values()), list(ent2id.values()))
        else:
            rel2types[rid] = (list(ent2id.values()), list(ent2id.values()))

    # connected sets for imputation stats
    connected_disease = set()
    if drug_disease:
        connected_disease.update(z for _,z in drug_disease)
    if protein_disease:
        connected_disease.update(z for _,z in protein_disease)
    connected_side = set(s for _,s in drug_se) if drug_se else set()

    meta = {
        "n_disease": n_disease_total,
        "n_side": n_side_total,
        "connected_disease": connected_disease,
        "connected_side": connected_side,
        "disease_list": disease_list,
        "side_list": side_list,
    }
    return np.asarray(triples_idx, dtype=np.int64), ent2id, rel2id, rel2types, meta

# ----------------------------
# Dataset & Model
# ----------------------------
class TripleDataset(Dataset):
    def __init__(self, triples: np.ndarray, rel2types: Dict[int, Tuple[Sequence[int], Sequence[int]]], num_negs: int = 5):
        self.triples = triples
        self.rel2types = rel2types
        self.num_negs = num_negs

    def __len__(self): return len(self.triples)

    def __getitem__(self, idx: int):
        h, r, t = self.triples[idx]
        negs = []
        for _ in range(self.num_negs):
            if random.random() < 0.5:
                nh = random.choice(self.rel2types[r][0])
                negs.append((nh, r, t))
            else:
                nt = random.choice(self.rel2types[r][1])
                negs.append((h, r, nt))
        return (h, r, t), negs

def collate(batch):
    pos = [x[0] for x in batch]
    neg = sum([x[1] for x in batch], [])
    pos = torch.tensor(pos, dtype=torch.long)
    neg = torch.tensor(neg, dtype=torch.long) if len(neg)>0 else None
    return pos, neg

class DistMult(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int):
        super().__init__()
        self.ent = nn.Embedding(num_entities, emb_dim)
        self.rel = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def scores(self, triples_b: torch.LongTensor) -> torch.Tensor:
        h = self.ent(triples_b[:,0])
        r = self.rel(triples_b[:,1])
        t = self.ent(triples_b[:,2])
        return torch.sum(h * r * t, dim=-1)

# ----------------------------
# Train + Export + Impute
# ----------------------------
def train_and_export(
    triples, ent2id, rel2id, rel2types, meta,
    emb_dim=128, epochs=200, batch_size=4096, num_negs=5,
    lr=1e-3, l2=1e-6, seed=123, add_inverse=False, out_dir="./kge_outputs",
    fill_strategy="gaussian", device=None
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TripleDataset(triples, rel2types, num_negs=num_negs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate)

    model = DistMult(num_entities=len(ent2id), num_relations=len(rel2id), emb_dim=emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        model.train()
        total, count = 0.0, 0
        for pos, neg in loader:
            pos = pos.to(device)
            if neg is not None: neg = neg.to(device)

            s_pos = model.scores(pos)
            y_pos = torch.ones_like(s_pos)

            if neg is not None and neg.size(0) > 0:
                s_neg = model.scores(neg)
                y_neg = torch.zeros_like(s_neg)
                s = torch.cat([s_pos, s_neg], dim=0)
                y = torch.cat([y_pos, y_neg], dim=0)
            else:
                s, y = s_pos, y_pos

            loss = bce(s, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            total += loss.item() * pos.size(0)
            count += pos.size(0)

        print(f"Epoch {ep:03d} | loss={total/max(1,count):.4f}")

    # Export tables
    ent_mat = model.ent.weight.detach().cpu().numpy()
    rel_mat = model.rel.weight.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "entity_embeddings.npy"), ent_mat)
    np.save(os.path.join(out_dir, "relation_embeddings.npy"), rel_mat)

    inv_ent = {v:k for k,v in ent2id.items()}
    with open(os.path.join(out_dir, "entities.tsv"), "w", encoding="utf-8") as f:
        for i in range(len(inv_ent)):
            f.write(f"{i}\t{inv_ent[i]}\n")
    inv_rel = {v:k for k,v in rel2id.items()}
    with open(os.path.join(out_dir, "relations.tsv"), "w", encoding="utf-8") as f:
        for i in range(len(inv_rel)):
            f.write(f"{i}\t{inv_rel[i]}\n")

    # Slice disease/side in 0..N-1 order
    n_dis, n_side = meta["n_disease"], meta["n_side"]
    dis_row_ids  = [ent2id[f"Disease:{i}"] for i in range(n_dis)]
    side_row_ids = [ent2id[f"SideEffect:{i}"] for i in range(n_side)]
    disease_emb  = ent_mat[dis_row_ids]
    side_emb     = ent_mat[side_row_ids]

    # Impute isolates
    connected_disease = set(meta["connected_disease"])
    connected_side    = set(meta["connected_side"])
    iso_dis_idx   = [i for i in range(n_dis)  if i not in connected_disease]
    iso_side_idx  = [i for i in range(n_side) if i not in connected_side]
    conn_dis_idx  = [i for i in range(n_dis)  if i in  connected_disease]
    conn_side_idx = [i for i in range(n_side) if i in  connected_side]

    def impute(target_emb: np.ndarray, conn_idx: List[int], iso_idx: List[int]):
        if len(iso_idx) == 0 or len(conn_idx) == 0: return target_emb
        sub = target_emb[conn_idx]
        if fill_strategy == "keep_random":
            return target_emb
        elif fill_strategy == "global_mean":
            mean = sub.mean(axis=0, keepdims=True)
            target_emb[iso_idx] = mean
            return target_emb
        elif fill_strategy == "gaussian":
            mean = sub.mean(axis=0, keepdims=True)
            std  = sub.std(axis=0, keepdims=True) + 1e-6
            noise = np.random.normal(loc=0.0, scale=std, size=(len(iso_idx), sub.shape[1]))
            target_emb[iso_idx] = mean + noise
            return target_emb
        else:
            raise ValueError("fill_strategy must be keep_random/global_mean/gaussian")

    disease_emb = impute(disease_emb, conn_dis_idx, iso_dis_idx)
    side_emb    = impute(side_emb,    conn_side_idx, iso_side_idx)

    # L2 normalize
    def l2norm(E):
        n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
        return E / n
    disease_emb = l2norm(disease_emb)
    side_emb    = l2norm(side_emb)

    # Save final
    np.save(os.path.join(out_dir, "disease_embeddings_dataB.npy"), disease_emb)
    np.save(os.path.join(out_dir, "sideeffect_embeddings_dataB.npy"), side_emb)

    with open(os.path.join(out_dir, "disease_entities.tsv"), "w", encoding="utf-8") as f:
        if meta["disease_list"] is not None:
            for name in meta["disease_list"]:
                f.write((name if name is not None else "") + "\n")
        else:
            for i in range(n_dis): f.write(f"Disease:{i}\n")

    with open(os.path.join(out_dir, "sideeffect_entities.tsv"), "w", encoding="utf-8") as f:
        if meta["side_list"] is not None:
            for name in meta["side_list"]:
                f.write((name if name is not None else "") + "\n")
        else:
            for i in range(n_side): f.write(f"SideEffect:{i}\n")

    print("[DONE] Saved embeddings to:", out_dir)
    print("  disease_embeddings.npy shape:", disease_emb.shape, f"| isolates filled: {len(iso_dis_idx)}")
    print("  sideeffect_embeddings.npy shape:", side_emb.shape, f"| isolates filled: {len(iso_side_idx)}")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # core files
    ap.add_argument("--drug_se", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/mat_data/drug_se.csv")
    ap.add_argument("--drug_disease", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/mat_data/drug_disease.csv")
    ap.add_argument("--protein_disease", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/mat_data/protein_disease.csv")
    ap.add_argument("--disease_list", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/diseaseID.txt", help="TXT: one disease name per line (index = line number)")
    ap.add_argument("--sideeffect_list", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/side_effect.txt", help="TXT: one side-effect name per line (index = line number)")

    # extra files (optional)
    ap.add_argument("--drug_drug", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/mat_data/drug_drug.csv")
    ap.add_argument("--protein_protein", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/mat_data/protein_protein.csv")
    ap.add_argument("--drug_protein", default="/Users/hongyangyang/Downloads/MHGNN-DTI-main/hetero_dataset/data/mat_data/drug_protein.csv")
    # full coverage options

    ap.add_argument("--n_disease", type=int, default=None)
    ap.add_argument("--n_side", type=int, default=None)
    # training
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--negs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--add_inverse", action="store_true")
    ap.add_argument("--fill_strategy", choices=["keep_random","global_mean","gaussian"], default="gaussian")
    ap.add_argument("--out_dir", default="./kge_outputs")
    args = ap.parse_args()

    set_seed(args.seed)

    # read all files
    drug_se = read_index_edges(args.drug_se)
    drug_disease = read_index_edges(args.drug_disease)
    protein_disease = read_index_edges(args.protein_disease)
    drug_drug = read_index_edges(args.drug_drug) if args.drug_drug else None
    protein_protein = read_index_edges(args.protein_protein) if args.protein_protein else None
    drug_protein = read_index_edges(args.drug_protein) if args.drug_protein else None

    disease_list = read_list(args.disease_list)
    side_list = read_list(args.sideeffect_list)

    triples, ent2id, rel2id, rel2types, meta = build_triples_from_indices(
        drug_se=drug_se, drug_disease=drug_disease, protein_disease=protein_disease,
        drug_drug=drug_drug, protein_protein=protein_protein, drug_protein=drug_protein,
        n_disease=args.n_disease, n_side=args.n_side,
        disease_list=disease_list, side_list=side_list,
        add_inverse=args.add_inverse
    )

    print(f"#entities={len(ent2id)}  #relations={len(rel2id)}  #triples={len(triples)}")
    print(f"  #disease(total)={meta['n_disease']}  #side(total)={meta['n_side']}")
    print(f"  connected disease={len(meta['connected_disease'])}  side={len(meta['connected_side'])}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_export(
        triples, ent2id, rel2id, rel2types, meta,
        emb_dim=args.emb_dim, epochs=args.epochs, batch_size=args.batch_size,
        num_negs=args.negs, lr=args.lr, l2=args.l2, seed=args.seed,
        add_inverse=args.add_inverse, out_dir=args.out_dir, device=device,
        fill_strategy=args.fill_strategy
    )

if __name__ == "__main__":
    main()
