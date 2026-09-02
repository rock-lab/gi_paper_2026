#!/usr/bin/env python3
"""
make_example_data.py -- (re)build the committed toy dataset from the REAL sources.

WHAT THIS IS
------------
This is a MAINTAINER-ONLY script.  It is run ONCE (on a machine that can see the
Rockefeller HPC filesystem) to populate everything under ``example_data/`` from the
REAL screen outputs, row-filtered down to the 15 published genes in
``TOY_GENES`` ("Set A"; 105 unordered ORF pairs + 15 self-pairs).  The
resulting slices are tiny and OK to commit.

Readers of this repository DO NOT need to run this: the ``example_data/`` outputs are
shipped in the repo (the "golden" checkpoints).  This script exists only so the
provenance of the toy data is fully reproducible from the raw sources.

It produces:
  1. example_data/gi_input/result_summary_long_df_exp{1,2}_toy.tsv
        Per-screen GI scores (+ their SE) for the 120 example pairs.  These are
        PRE-DERIVED GOLDEN values: they are the *input* to the joint model, and
        are what the joint-model notebook/runner consumes.
  2. example_data/counts/exp1/*  and  example_data/counts/exp2/*
        The pooled +ATc / -ATc barcode count tables, row-filtered to constructs
        whose BOTH sides target a toy gene or a Negative control.
  3. example_data/expected/merged_quad_me_trunc_halfsmeared_toy.tsv
        The golden joint-model output (primary model), sliced to the toy pairs.
  4. example_data/guide_name_map.tsv
        The single-sgRNA library map (guide_name / orf_id / name / seq / ...),
        sliced to the toy genes + Negative controls.

The known HPC source paths are hard-coded as argparse defaults; override them
with the corresponding flags if your mounts differ.

Deps: numpy, pandas.

Run:
    python make_example_data.py
"""

import argparse
import glob
import os

import pandas as pd


# ------------------------------------------------------------------
# The 15-gene toy set "Set A"  (ORF id -> gene name)
#   3 negative (aggravating) pairs = redundant-isoenzyme synthetic lethals
#   2 positive (alleviating) pairs
#   5 inert filler genes (essential; ~no interaction within the set)
# ------------------------------------------------------------------
TOY_GENES = {
    "RVBD1854c": "ndh",     "RVBD0392c": "ndhA",    # neg: two NADH dehydrogenases
    "RVBD2754c": "thyX",    "RVBD2764c": "thyA",    # neg: two thymidylate synthases
    "RVBD0050":  "ponA1",   "RVBD3682":  "ponA2",   # neg: two class-A PBPs
    "RVBD2193":  "ctaE",    "RVBD2200c": "ctaC",    # pos: cytochrome-c oxidase
    "RVBD1304":  "atpB",    "RVBD3795":  "embB",    # pos: ATP synthase x arabinosyltransferase
    "RVBD1025":  "RVBD1025", "RVBD1822": "pgsA2",  "RVBD3261": "fbiA",   # inert filler
    "RVBD2447c": "folC",     "RVBD2794c": "pptT",                        # inert filler
}
SET = set(TOY_GENES)                 # ORF ids only (used for GI-score / merged rows)
ALLOWED = SET | {"Negative"}         # ORF ids + NT control (used for construct rows)

# The 9 columns kept in each per-screen GI-input slice, in this order.
GI_INPUT_COLS = [
    "orf1", "orf2", "orf_pair",
    "delta_prime_median", "sd_delta_prime_median",
    "prob_interaction_median", "correlation",
    "name1", "name2",
]


# ------------------------------------------------------------------
# Default (real) source paths
# ------------------------------------------------------------------
# Per-screen result_summary_long_df TSVs -- these HPC copies carry the SE column
# (sd_delta_prime_median) that the measurement-error joint model needs.
DEF_EXP1_GI = os.path.expanduser(
    "~/gi/results/H37Rv/tb_lowcas9_100ng/01_22_2025/modeling/"
    "indiv_guide_pair_twoline_model/sampling/"
    "result_summary_long_df_H37Rv_tb_lowcas9_100ng_01_22_2025.tsv"
)
DEF_EXP2_GI = os.path.expanduser(
    "~/gi/results/H37Rv/tb_lowcas9_500ng/06_10_2025/modeling/"
    "indiv_guide_pair_twoline_model/sampling/"
    "result_summary_long_df_H37Rv_tb_lowcas9_500ng_06_10_2025.tsv"
)

# Pooled barcode count tables (one file per timepoint x ATc condition).
DEF_EXP1_COUNTS_DIR = (
    "/store/home/mad/PROJECTS/CRISPRi/gi/tb_ess_library/sequencing/nocarb_full"
)
DEF_EXP2_COUNTS_DIR = (
    "/store/home/mad/PROJECTS/CRISPRi/gi/tb_ess_library/sequencing/medcas9"
)

# Golden joint-model output for the PRIMARY model (quad_me_trunc_halfsmeared, w010).
DEF_MERGED = (
    "/store/home/mad/PROJECTS/CRISPRi/gi/tb_ess_library/paper/analyses/prob_gi/"
    "joint_mixture/merged_quad_me_trunc_halfsmeared_w010.tsv"
)

# Single-sgRNA library map (guide_name / orf_id / name / seq / ... columns).
DEF_GUIDE_MAP = (
    "/store/home/mad/PROJECTS/CRISPRi/gi/tb_ess_library/paper/"
    "H37Rv_single_sgrna_library.tsv"
)


# ------------------------------------------------------------------
# Construct-id parsing
# ------------------------------------------------------------------
# A construct id is  LEFT_RIGHT  where each side is either
#   'ORF:name_SEQ'   (e.g. RVBD0001:dnaA_GATGACGATTTGCTTG), or
#   'Negative_SEQ'   (the NT control).
# Neither ORF ids, gene names, nor SEQs contain '_', so splitting on '_' with
# a cap of 3 splits always yields exactly [left_tok, seq1, right_tok, seq2].
# The ORF of a side is the text before the first ':' ('Negative' has no ':' and
# maps to itself).

def _side_orfs(ids):
    """Vectorized: given a Series of construct ids, return (orf_left, orf_right)."""
    parts = ids.str.split("_", n=3, expand=True)
    orf_l = parts[0].str.split(":").str[0]
    if 2 in parts.columns:
        orf_r = parts[2].str.split(":").str[0]
    else:  # malformed row with <3 underscores; will fail the ALLOWED test
        orf_r = pd.Series([None] * len(ids), index=ids.index)
    return orf_l, orf_r


# ------------------------------------------------------------------
# Step 1: per-screen GI-input slices
# ------------------------------------------------------------------

def filter_gi_input(src, out_path):
    """Row-filter one result_summary_long_df TSV to the toy pairs.

    The source has a leading UNNAMED index column (read via index_col=0) and many
    extra columns; we keep only GI_INPUT_COLS and re-emit with a fresh leading
    unnamed index so the file interoperates with the joint runner / merge scripts
    (both read these with ``index_col=0`` and merge on ``orf_pair``).
    """
    df = pd.read_csv(src, sep="\t", index_col=0)
    df = df[df["orf1"].isin(SET) & df["orf2"].isin(SET)]
    df = df[GI_INPUT_COLS].reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t")  # index=True -> leading unnamed index column
    print(f"  wrote {out_path}  ({len(df)} pairs)")
    return df


# ------------------------------------------------------------------
# Step 2: pooled count-table slices
# ------------------------------------------------------------------

def filter_counts_file(src, out_path, chunksize=500_000):
    """Row-filter one pooled count table; drop any stray ``test_counts_*`` column.

    Read as strings so integer counts are preserved verbatim, and stream in
    chunks (the real tables have ~2.2M rows)."""
    first = True
    n_in = n_out = 0
    for chunk in pd.read_csv(src, sep="\t", dtype=str, chunksize=chunksize):
        n_in += len(chunk)
        drop = [c for c in chunk.columns if c.startswith("test_counts")]
        if drop:
            chunk = chunk.drop(columns=drop)
        orf_l, orf_r = _side_orfs(chunk["Id"])
        mask = orf_l.isin(ALLOWED) & orf_r.isin(ALLOWED)
        sub = chunk[mask]
        n_out += len(sub)
        sub.to_csv(out_path, sep="\t", index=False,
                   mode="w" if first else "a", header=first)
        first = False
    if first:  # empty source -> still emit an empty file with no rows
        pd.DataFrame(columns=["Id"]).to_csv(out_path, sep="\t", index=False)
    return n_in, n_out


def filter_counts_dir(src_dir, out_dir, prefix):
    """Filter every ``counts_<prefix>*_{plus,minus}.txt`` pooled table in a dir."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(src_dir, f"counts_{prefix}*_plus.txt"))
        + glob.glob(os.path.join(src_dir, f"counts_{prefix}*_minus.txt"))
    )
    if not files:
        print(f"  WARNING: no pooled count files found in {src_dir}")
    for src in files:
        out_path = os.path.join(out_dir, os.path.basename(src))
        n_in, n_out = filter_counts_file(src, out_path)
        print(f"  {os.path.basename(src)}: {n_out:,} / {n_in:,} constructs kept")


# ------------------------------------------------------------------
# Step 3: golden joint-model output slice
# ------------------------------------------------------------------

def filter_merged(src, out_path):
    """Slice the golden merged joint-model output to the toy pairs (all 20 cols)."""
    df = pd.read_csv(src, sep="\t")
    df = df[df["orf1"].isin(SET) & df["orf2"].isin(SET)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  wrote {out_path}  ({len(df)} pairs, {len(df.columns)} cols)")
    return df


# ------------------------------------------------------------------
# Step 4: single-sgRNA library (guide-name) map slice
# ------------------------------------------------------------------

def filter_guide_map(src, out_path):
    """Slice the single-sgRNA library map to the toy genes + Negative controls."""
    if not os.path.exists(src):
        print(f"  WARNING: guide map not found ({src}); skipping guide_name_map.tsv")
        return None
    df = pd.read_csv(src, sep="\t", dtype=str)
    df = df[df["orf_id"].isin(ALLOWED)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  wrote {out_path}  ({len(df)} guides)")
    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--exp1-gi", default=DEF_EXP1_GI,
                   help="exp1 (100ng) result_summary_long_df TSV")
    p.add_argument("--exp2-gi", default=DEF_EXP2_GI,
                   help="exp2 (500ng) result_summary_long_df TSV")
    p.add_argument("--exp1-counts-dir", default=DEF_EXP1_COUNTS_DIR,
                   help="dir of exp1 pooled counts_P*_{plus,minus}.txt")
    p.add_argument("--exp2-counts-dir", default=DEF_EXP2_COUNTS_DIR,
                   help="dir of exp2 pooled counts_G*_{plus,minus}.txt")
    p.add_argument("--merged", default=DEF_MERGED,
                   help="golden merged joint-model output (primary model)")
    p.add_argument("--guide-map", default=DEF_GUIDE_MAP,
                   help="single-sgRNA library map (H37Rv_single_sgrna_library.tsv)")
    p.add_argument("--out-dir", default=os.path.join(here, "example_data"),
                   help="output example_data/ directory (default: alongside this script)")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out_dir

    print("Toy genes:", ", ".join(f"{o}:{n}" for o, n in TOY_GENES.items()))
    print(f"Output dir: {out}\n")

    print("[1/4] Per-screen GI-input slices (pre-derived golden values)")
    filter_gi_input(args.exp1_gi,
                    os.path.join(out, "gi_input",
                                 "result_summary_long_df_exp1_toy.tsv"))
    filter_gi_input(args.exp2_gi,
                    os.path.join(out, "gi_input",
                                 "result_summary_long_df_exp2_toy.tsv"))

    print("\n[2/4] Pooled count-table slices")
    print(" exp1 (100ng, nocarb_full):")
    filter_counts_dir(args.exp1_counts_dir,
                      os.path.join(out, "counts", "exp1"), prefix="P")
    print(" exp2 (500ng, medcas9):")
    filter_counts_dir(args.exp2_counts_dir,
                      os.path.join(out, "counts", "exp2"), prefix="G")

    print("\n[3/4] Golden joint-model output slice (primary model)")
    filter_merged(args.merged,
                  os.path.join(out, "expected",
                               "merged_quad_me_trunc_halfsmeared_toy.tsv"))

    print("\n[4/4] Guide-name map slice")
    filter_guide_map(args.guide_map, os.path.join(out, "guide_name_map.tsv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
