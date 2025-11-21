import argparse
import os
import sys
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def infer_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find the column for {what}. "
        f"Please specify it via command-line arguments. "
        f"Candidate column names: {candidates}"
    )


# Compatible with older PyTorch versions (e.g., 1.8 without inference_mode)
INFER = getattr(torch, "inference_mode", torch.no_grad)


@INFER()
def embed_texts(texts, tok, model, device, batch_size=32, pooling="mean", max_length=None):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch = tok(
            batch_texts,
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        )
        # Move batch to the target device (BatchEncoding supports .to)
        batch = batch.to(device)

        # Forward pass to get hidden states
        out = model(**batch)
        hidden = out.last_hidden_state   # [B, T, H]

        # Convert mask to the same floating dtype as hidden, 1/0 -> 1.0/0.0
        mask = batch["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)  # [B, T, 1]

        if pooling == "mean":
            num = (hidden * mask).sum(dim=1)          # [B, H]
            den = mask.sum(dim=1).clamp(min=1e-8)     # [B, 1], avoid division by zero
            emb = num / den
        elif pooling == "cls":
            # For models like RoBERTa/ESM: use the first token as the sequence representation
            emb = hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        embs.append(emb.detach().cpu())
    return torch.cat(embs, dim=0) if embs else torch.empty(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug_csv", default="drug_info_datasetB.csv",
                        help="Path to the drug SMILES CSV file")
    parser.add_argument("--prot_csv", default="protein_info_datasetB.csv",
                        help="Path to the protein primary-sequence CSV file")
    parser.add_argument("--output_pt", default="embeddings_datasetB.pt",
                        help="Output .pt filepath for joint embeddings")
    parser.add_argument("--drug_output", default="drug_only.pt",
                        help="Optional .pt filepath to separately save drug embeddings "
                             "(if not specified, it will be derived from output_pt)")
    parser.add_argument("--prot_output", default="prot_only.pt",
                        help="Optional .pt filepath to separately save protein embeddings "
                             "(if not specified, it will be derived from output_pt)")

    parser.add_argument("--smiles_col", default=None,
                        help="Column name for SMILES in the drug CSV")
    parser.add_argument("--drug_id_col", default=None,
                        help="Drug ID column name (optional; used for traceability)")
    parser.add_argument("--seq_col", default=None,
                        help="Sequence column name in the protein CSV "
                             "(primary structure, e.g., 'MKT...')")
    parser.add_argument("--prot_id_col", default=None,
                        help="Protein ID column name (optional; used for traceability)")

    # You can change these to your preferred model names
    parser.add_argument("--drug_model", default="DeepChem/ChemBERTa-77M-MTR",
                        help="Hugging Face model name for ChemBERTa")
    parser.add_argument("--prot_model", default="facebook/esm2_t33_650M_UR50D",
                        help="Hugging Face model name for ESM-2")

    parser.add_argument("--drug_bs", type=int, default=64,
                        help="Batch size for drug (SMILES) encoding")
    parser.add_argument("--prot_bs", type=int, default=2,
                        help="Batch size for protein encoding "
                             "(650M parameters, so default is small)")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean",
                        help="Pooling strategy for sequence embeddings")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 for inference on supported CUDA devices to save memory")
    parser.add_argument("--l2norm", action="store_true",
                        help="Apply L2 normalization to the final embeddings")
    parser.add_argument("--proj_dim", type=int, default=0,
                        help="If >0, apply a linear projection to this dimension before saving "
                             "(e.g., 64); 0 means no projection (currently not used).")

    args = parser.parse_args()

    # Read CSVs
    drug_df = pd.read_csv(args.drug_csv)
    prot_df = pd.read_csv(args.prot_csv)

    # Infer or use explicit column names
    smiles_col = args.smiles_col or infer_col(
        drug_df, ["smiles", "SMILES", "canonical_smiles", "CanonicalSMILES"], "SMILES"
    )
    seq_col = args.seq_col or infer_col(
        prot_df,
        ["sequence", "Sequence", "protein_sequence", "target_aa_code", "seq"],
        "protein sequence",
    )

    drug_ids = (drug_df[args.drug_id_col].astype(str).tolist()
                if args.drug_id_col and args.drug_id_col in drug_df.columns
                else [str(i) for i in range(len(drug_df))])
    prot_ids = (prot_df[args.prot_id_col].astype(str).tolist()
                if args.prot_id_col and args.prot_id_col in prot_df.columns
                else [str(i) for i in range(len(prot_df))])

    drug_texts = drug_df[smiles_col].astype(str).fillna("").tolist()
    prot_texts = prot_df[seq_col].astype(str).fillna("").tolist()

    # Load models/tokenizers (tested with PyTorch 1.8+ and Transformers 4.26+)
    # Only use half precision when on CUDA and --fp16 is enabled
    torch_dtype = torch.float16 if (args.fp16 and str(args.device).startswith("cuda")) else None

    drug_tok = AutoTokenizer.from_pretrained(args.drug_model)
    drug_tok.model_max_length = 512
    drug_model = AutoModel.from_pretrained(
        args.drug_model,
        torch_dtype=torch_dtype
    ).to(args.device).eval()

    prot_tok = AutoTokenizer.from_pretrained(args.prot_model)
    prot_tok.model_max_length = 1022
    prot_model = AutoModel.from_pretrained(
        args.prot_model,
        torch_dtype=torch_dtype
    ).to(args.device).eval()

    # Generate embeddings
    print(f"[Drug] {len(drug_texts)} SMILES -> {args.drug_model} ...")
    drug_emb = embed_texts(
        drug_texts, drug_tok, drug_model,
        args.device, args.drug_bs, args.pooling, max_length=512
    )

    print(f"[Prot] {len(prot_texts)} sequences -> {args.prot_model} ...")
    prot_emb = embed_texts(
        prot_texts, prot_tok, prot_model,
        args.device, args.prot_bs, args.pooling, max_length=1022
    )

    # Optional: L2 normalization
    if args.l2norm:
        drug_emb = torch.nn.functional.normalize(drug_emb, p=2, dim=1)
        prot_emb = torch.nn.functional.normalize(prot_emb, p=2, dim=1)

    # Save joint embeddings
    save_obj = {
        "meta": {
            "drug_model": args.drug_model,
            "prot_model": args.prot_model,
            "pooling": args.pooling,
            "l2norm": bool(args.l2norm),
            "proj_dim": int(args.proj_dim),
            "smiles_col": smiles_col,
            "seq_col": seq_col,
        },
        "drug": {"ids": drug_ids, "emb": drug_emb},   # [N_d, D]
        "prot": {"ids": prot_ids, "emb": prot_emb},   # [N_p, D]
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pt)) or ".", exist_ok=True)
    torch.save(save_obj, args.output_pt)
    print(f"Saved joint embeddings -> {args.output_pt}")

    base = os.path.splitext(args.output_pt)[0]
    # If user did not explicitly set drug_output/prot_output,
    # derive them from the base of output_pt.
    drug_out = args.drug_output if args.drug_output else (base + "_drug.pt")
    prot_out = args.prot_output if args.prot_output else (base + "_prot.pt")

    torch.save({"ids": drug_ids, "emb": drug_emb}, drug_out)
    torch.save({"ids": prot_ids, "emb": prot_emb}, prot_out)

    print(f"Also saved separate embeddings -> drug: {drug_out} | prot: {prot_out}")
    print(f"Drug emb shape: {tuple(drug_emb.shape)}  |  Prot emb shape: {tuple(prot_emb.shape)}")


if __name__ == "__main__":
    main()

