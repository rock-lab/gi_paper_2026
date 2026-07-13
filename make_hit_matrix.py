#!/usr/bin/env python3
"""Build a gene-by-gene BOOLEAN hit matrix.

Takes a merged per-pair results TSV (default: the half-smeared model) and
writes a square, symmetric gene x gene matrix of True/False values marking
whether each pair is a significant interaction hit.

A pair is a hit (True) if ANY of the directional class probabilities exceeds
the threshold:

    prob_aggravating >= thr  OR  prob_alleviating >= thr  OR  prob_discordant >= thr

(These classes are mutually exclusive in practice, so at most one can clear a
threshold >= 0.5.) Comparison uses >= to match the hit counts used elsewhere
in this project; at continuous probabilities >= vs > differ only at an exact
tie at the threshold.

Output format matches the existing matrix files: tab-separated square matrix
with a gene label in the corner, gene IDs as the column header row, and gene
IDs as the first column. Intended as input to a heatmap tool that uses the
boolean matrix to mark hits.

Usage:
    python make_hit_matrix.py
    python make_hit_matrix.py --threshold 0.99
    python make_hit_matrix.py --as-int --labels name
"""

import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MERGED = os.path.join(
    SCRIPT_DIR, "example_data", "expected",
    "merged_quad_me_trunc_halfsmeared_toy.tsv")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged-tsv", default=DEFAULT_MERGED,
                   help="Merged per-pair results TSV (default: half-smeared model)")
    p.add_argument("--output", default=None,
                   help="Output TSV path (default: derived from input name + threshold)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for a hit (default: 0.5 = paper "
                        "call threshold; use 0.95 for a stricter 'confident' cutoff)")
    p.add_argument("--labels", choices=["orf", "name"], default="orf",
                   help="Use ORF IDs (default) or gene names as row/col labels")
    p.add_argument("--index-name", default="orf2",
                   help="Corner label / index name (default: orf2)")
    p.add_argument("--diagonal", choices=["data", "false", "true"], default="data",
                   help="Self-pair diagonal: hit status from data (default), "
                        "all False, or all True")
    p.add_argument("--as-int", action="store_true",
                   help="Write 1/0 instead of True/False")
    return p.parse_args()


def main():
    args = parse_args()
    thr = args.threshold

    print(f"Reading: {args.merged_tsv}")
    df = pd.read_csv(args.merged_tsv, sep="\t")
    need = {"orf1", "orf2", "prob_aggravating", "prob_alleviating",
            "prob_discordant"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["hit"] = ((df["prob_aggravating"] >= thr)
                 | (df["prob_alleviating"] >= thr)
                 | (df["prob_discordant"] >= thr))

    # Map ORF -> display label
    orfs = sorted(set(df["orf1"]) | set(df["orf2"]))
    if args.labels == "name":
        orf2name = {}
        for o, n in zip(df["orf1"], df["name1"]):
            orf2name.setdefault(o, n)
        for o, n in zip(df["orf2"], df["name2"]):
            orf2name.setdefault(o, n)
        labels = [orf2name.get(o, o) for o in orfs]
        if len(set(labels)) != len(labels):
            dup = pd.Series(labels)[pd.Series(labels).duplicated()].unique()
            print(f"  WARNING: {len(dup)} gene names are not unique "
                  f"(e.g. {list(dup[:5])}); matrix labels will be duplicated.")
    else:
        labels = orfs

    n = len(orfs)
    idx = {o: i for i, o in enumerate(orfs)}
    print(f"  {len(df):,} pairs, {n} genes -> {n}x{n} matrix  (threshold {thr})")

    # Missing pairs default to False (no hit)
    M = np.zeros((n, n), dtype=bool)
    i = df["orf1"].map(idx).to_numpy()
    j = df["orf2"].map(idx).to_numpy()
    v = df["hit"].to_numpy()
    M[i, j] = v
    M[j, i] = v

    # Diagonal handling (self-pairs)
    if args.diagonal == "false":
        np.fill_diagonal(M, False)
    elif args.diagonal == "true":
        np.fill_diagonal(M, True)
    # "data" -> leave as filled from self-pair rows (i == j)

    out = pd.DataFrame(M, index=labels, columns=labels)
    out.index.name = args.index_name
    if args.as_int:
        out = out.astype(int)

    if args.output:
        out_path = args.output
    else:
        base = os.path.basename(args.merged_tsv)
        tag = base[len("merged_"):-len(".tsv")] if base.startswith("merged_") else "model"
        thr_str = f"{thr:.2f}".replace(".", "")   # 0.95 -> "095"
        out_path = os.path.join(
            os.path.dirname(args.merged_tsv),
            f"hit_matrix_thr{thr_str}_{tag}.tsv")

    out.to_csv(out_path, sep="\t")
    print(f"  Wrote: {out_path}")

    # Sanity summary (off-diagonal, counts each pair twice)
    offdiag_hits = int(M[~np.eye(n, dtype=bool)].sum()) // 2
    n_aggr = int((df["prob_aggravating"] >= thr).sum())
    n_allev = int((df["prob_alleviating"] >= thr).sum())
    n_disc = int((df["prob_discordant"] >= thr).sum())
    self_hits = int(np.diag(M).sum())
    print(f"  Hit pairs (off-diagonal, unique): {offdiag_hits:,}")
    print(f"    aggravating: {n_aggr:,}  alleviating: {n_allev:,}  discordant: {n_disc:,}")
    print(f"  Self-pair (diagonal) hits: {self_hits}")


if __name__ == "__main__":
    main()
