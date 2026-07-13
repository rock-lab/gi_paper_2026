#!/usr/bin/env python3
"""Build a gene-by-gene SIGNED probability-of-interaction matrix.

Takes a merged per-pair results TSV (default: the half-smeared model) and
writes a square, symmetric gene x gene matrix whose entries are a *signed*
probability of interaction:

    aggravating / negative interaction  -> NEGATIVE value
    alleviating / positive interaction  -> POSITIVE value
    null or discordant                  -> near zero

This is intended as input to an external heatmap tool (diverging colormap
centered at 0). The output format matches the existing
`result_mean_delta_prime_median_matrix_df_*.tsv` files: a tab-separated
square matrix with a gene label in the corner, gene IDs as the column
header row, and gene IDs as the first column.

Value modes (--value-mode):
  conc_diff  (default)  signed = P(alleviating) - P(aggravating)
                        Bounded [-1, 1]. Magnitude = net directional
                        interaction probability. Discordant/null -> ~0.
  signed_interaction    signed = sign(P_allev - P_aggr) * P(interaction)
                        Magnitude is the full P(interaction) (includes the
                        discordant component), with the sign taken from the
                        dominant concordant direction.
  signed_maxmag         signed = sign(larger-|GI| experiment) * P(interaction)
                        Magnitude is the full P(interaction); the sign is the
                        sign of whichever experiment has the larger-magnitude
                        GI score. For discordant pairs (opposite signs) this
                        picks the stronger direction, e.g. (-10, +1) -> NEG.
                        Concordant pairs are unchanged vs signed_interaction.

Usage:
    python make_signed_prob_matrix.py
    python make_signed_prob_matrix.py --value-mode signed_interaction
    python make_signed_prob_matrix.py --labels name --output my_matrix.tsv
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
                   help="Output TSV path (default: derived from input name)")
    p.add_argument("--value-mode",
                   choices=["conc_diff", "signed_interaction", "signed_maxmag"],
                   default="conc_diff",
                   help="How to compute the signed value (default: conc_diff = "
                        "P_alleviating - P_aggravating)")
    p.add_argument("--labels", choices=["orf", "name"], default="orf",
                   help="Use ORF IDs (default) or gene names as row/col labels")
    p.add_argument("--index-name", default="orf2",
                   help="Corner label / index name (default: orf2, matching the "
                        "delta_prime matrix format)")
    p.add_argument("--diagonal", choices=["data", "nan", "zero"], default="data",
                   help="Self-pair diagonal: use values from data (default), "
                        "blank (nan), or zero")
    p.add_argument("--fill", choices=["nan", "zero"], default="nan",
                   help="Fill for any gene pair missing from the data (default: nan)")
    p.add_argument("--decimals", type=int, default=6,
                   help="Round values to this many decimals (default: 6)")
    return p.parse_args()


def compute_signed(df, mode):
    """Return the signed value per row (Series aligned to df)."""
    p_aggr = df["prob_aggravating"].to_numpy()
    p_allev = df["prob_alleviating"].to_numpy()
    if mode == "conc_diff":
        return p_allev - p_aggr

    p_int = df["prob_interaction"].to_numpy()
    if mode == "signed_maxmag":
        # Sign = sign of whichever experiment has the larger |GI score|.
        # For concordant pairs both GIs share a sign so this matches the
        # concordant direction; for discordant pairs it picks the stronger one.
        gi1 = df["gi_score_exp1"].to_numpy()
        gi2 = df["gi_score_exp2"].to_numpy()
        dominant = np.where(np.abs(gi1) >= np.abs(gi2), gi1, gi2)
        sign = np.where(dominant >= 0, 1.0, -1.0)
        return sign * p_int

    # signed_interaction: sign from dominant concordant direction, magnitude
    # = total P(interaction). Ties (p_allev == p_aggr) -> +1 (arbitrary; only
    # affects perfectly symmetric, effectively-null cases).
    sign = np.where(p_allev >= p_aggr, 1.0, -1.0)
    return sign * p_int


def main():
    args = parse_args()

    print(f"Reading: {args.merged_tsv}")
    df = pd.read_csv(args.merged_tsv, sep="\t")
    need = {"orf1", "orf2", "prob_aggravating", "prob_alleviating",
            "prob_interaction"}
    if args.value_mode == "signed_maxmag":
        need |= {"gi_score_exp1", "gi_score_exp2"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["signed"] = compute_signed(df, args.value_mode)

    # Map ORF -> display label
    if args.labels == "name":
        orf2name = {}
        for o, n in zip(df["orf1"], df["name1"]):
            orf2name.setdefault(o, n)
        for o, n in zip(df["orf2"], df["name2"]):
            orf2name.setdefault(o, n)
        orfs = sorted(set(df["orf1"]) | set(df["orf2"]))
        labels = [orf2name.get(o, o) for o in orfs]
        if len(set(labels)) != len(labels):
            dup = pd.Series(labels)[pd.Series(labels).duplicated()].unique()
            print(f"  WARNING: {len(dup)} gene names are not unique "
                  f"(e.g. {list(dup[:5])}); matrix labels will be duplicated.")
    else:
        orfs = sorted(set(df["orf1"]) | set(df["orf2"]))
        labels = orfs

    n = len(orfs)
    idx = {o: i for i, o in enumerate(orfs)}
    print(f"  {len(df):,} pairs, {n} genes -> {n}x{n} matrix")

    fill_val = np.nan if args.fill == "nan" else 0.0
    M = np.full((n, n), fill_val, dtype=float)

    i = df["orf1"].map(idx).to_numpy()
    j = df["orf2"].map(idx).to_numpy()
    v = df["signed"].to_numpy()
    # Symmetric fill (each unordered pair listed once in the data)
    M[i, j] = v
    M[j, i] = v

    # Diagonal handling (self-pairs)
    if args.diagonal == "nan":
        np.fill_diagonal(M, np.nan)
    elif args.diagonal == "zero":
        np.fill_diagonal(M, 0.0)
    # "data" -> leave as filled from self-pair rows (i == j)

    if args.decimals is not None:
        M = np.round(M, args.decimals)

    out = pd.DataFrame(M, index=labels, columns=labels)
    out.index.name = args.index_name

    if args.output:
        out_path = args.output
    else:
        base = os.path.basename(args.merged_tsv)
        tag = base[len("merged_"):-len(".tsv")] if base.startswith("merged_") else "model"
        out_path = os.path.join(
            os.path.dirname(args.merged_tsv),
            f"signed_prob_interaction_matrix_{args.value_mode}_{tag}.tsv")

    out.to_csv(out_path, sep="\t")
    print(f"  Wrote: {out_path}")

    # Quick sanity summary
    offdiag = M[~np.eye(n, dtype=bool)]
    finite = offdiag[np.isfinite(offdiag)]
    if finite.size:
        print(f"  Off-diagonal range: [{finite.min():.4f}, {finite.max():.4f}]")
        print(f"    negative (aggravating-leaning): {(finite < 0).sum():,}")
        print(f"    positive (alleviating-leaning): {(finite > 0).sum():,}")
        print(f"    |value| >= 0.95: {(np.abs(finite) >= 0.95).sum():,}  "
              f"(of {finite.size:,} off-diagonal cells, counts each pair twice)")
    else:
        print("  Off-diagonal: no finite values")


if __name__ == "__main__":
    main()
