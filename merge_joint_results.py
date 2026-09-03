#!/usr/bin/env python3
"""
Merge joint genetic-interaction (GI) model results into one flat TSV.

Combines the two per-screen GI summaries (result_summary_long_df TSVs) with the
per-pair posterior class probabilities emitted by the joint quadrant mixture model
(run_joint_model.py) into a single 20-column table, one row per shared orf_pair.

This is the quadrant-model merge only (the primary model
`joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan`).  The joint model's
five biological-class log-probabilities (concordant / discordant / null / aggravating
/ alleviating) plus overall interaction probability are renamed to human-readable
columns and joined onto the per-screen GI scores.

Usage:
    python merge_joint_results.py \
        --exp1-tsv result_summary_long_df_exp1.tsv \
        --exp2-tsv result_summary_long_df_exp2.tsv \
        --summary-tsv joint_out_quad_me_trunc_halfsmeared_w010_summary.tsv \
        --output merged_quad_me_trunc_halfsmeared_w010.tsv

Output: 20-column TSV
    orf1, orf2, name1, name2,
    gi_score_exp1, se_exp1, gi_score_overlaps_zero_exp1,
    gi_score_exp2, se_exp2, gi_score_overlaps_zero_exp2,
    correlation_exp1, correlation_exp2,
    prob_interaction_median_exp1, prob_interaction_median_exp2,
    prob_interaction, prob_aggravating, prob_alleviating,
    prob_discordant, prob_no_interaction, prob_concordant
"""

import argparse
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# Columns to load from per-screen summary TSVs (result_summary_long_df)
# ──────────────────────────────────────────────────────────────────

EXP_COLS = [
    "orf_pair", "orf1", "orf2", "name1", "name2",
    "delta_prime_median", "sd_delta_prime_median",
    "prob_interaction_median",
    "correlation",
]

# Columns that MUST be present in each per-screen input (the join key, the pair
# identity, and the GI score + its SE). A missing one is fatal rather than
# silently dropped by the EXP_COLS selection below.
REQUIRED_EXP_COLS = [
    "orf_pair", "orf1", "orf2", "delta_prime_median", "sd_delta_prime_median",
]


# ──────────────────────────────────────────────────────────────────
# Base DataFrame: per-screen GI scores from both experiments
# ──────────────────────────────────────────────────────────────────

def _read_per_screen_tsv(path):
    """Read a per-screen result_summary TSV, tolerating (and dropping) a leading
    unnamed index column WITHOUT index_col=0 — which would otherwise consume the
    first real column (orf1) of an index-free TSV."""
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) and str(df.columns[0]).startswith("Unnamed:"):
        df = df.drop(columns=df.columns[0])
    return df


def _require_cols(df, label, path):
    """Fail loudly if a per-screen table is missing a required column, rather than
    letting the EXP_COLS selection silently drop it."""
    missing = [c for c in REQUIRED_EXP_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} ({path}) is missing required column(s): {missing}. "
            f"Present columns: {list(df.columns)}")


def load_base_df(exp1_path, exp2_path):
    """Load the two per-screen result_summary TSVs and inner-join on orf_pair.

    Renames per-screen GI columns to gi_score_exp{1,2} / se_exp{1,2}, carries
    correlation and prob_interaction_median (auto-suffixed _exp1/_exp2 by the
    merge), and adds gi_score_overlaps_zero_exp{1,2} = |gi| < 1.96*se.
    """
    print("Loading Exp1...")
    exp1 = _read_per_screen_tsv(exp1_path)
    _require_cols(exp1, "Exp1", exp1_path)
    exp1 = exp1[[c for c in EXP_COLS if c in exp1.columns]]
    print(f"  {len(exp1)} pairs")

    print("Loading Exp2...")
    exp2 = _read_per_screen_tsv(exp2_path)
    _require_cols(exp2, "Exp2", exp2_path)
    exp2 = exp2[[c for c in EXP_COLS if c in exp2.columns]]
    print(f"  {len(exp2)} pairs")

    # Guard the join: duplicate orf_pair rows would multiply out (a silent
    # cartesian blow-up), and pairs present in only one screen are dropped by the
    # inner join — report how many so that loss is never invisible.
    for df, lbl in [(exp1, "Exp1"), (exp2, "Exp2")]:
        dup = df["orf_pair"].duplicated()
        if dup.any():
            raise ValueError(
                f"{lbl} has {int(dup.sum())} duplicate orf_pair rows "
                f"(e.g. {df.loc[dup, 'orf_pair'].head(3).tolist()}). Aggregate to "
                f"one row per gene pair before merging.")
    only1 = set(exp1["orf_pair"]) - set(exp2["orf_pair"])
    only2 = set(exp2["orf_pair"]) - set(exp1["orf_pair"])
    if only1 or only2:
        print(f"  NOTE: {len(only1)} pairs only in Exp1, {len(only2)} only in Exp2; "
              f"inner join keeps the {len(set(exp1['orf_pair']) & set(exp2['orf_pair']))} shared.")

    print("Merging experiments (inner join on orf_pair)...")
    base = exp1.merge(exp2, on="orf_pair", suffixes=("_exp1", "_exp2"),
                      how="inner", validate="one_to_one")
    print(f"  {len(base)} shared pairs")

    # Rename per-screen GI score / SE columns for clarity
    base = base.rename(columns={
        "delta_prime_median_exp1": "gi_score_exp1",
        "delta_prime_median_exp2": "gi_score_exp2",
        "sd_delta_prime_median_exp1": "se_exp1",
        "sd_delta_prime_median_exp2": "se_exp2",
    })

    # Flag pairs whose per-screen 95% CI overlaps zero. Guard on BOTH SE columns,
    # and emit <NA> (not False) wherever an SE is missing, so a NaN SE is never
    # silently recorded as "does not overlap zero" (i.e. significant).
    have_se_cols = "se_exp1" in base.columns and "se_exp2" in base.columns
    if have_se_cols:
        for exp in ["exp1", "exp2"]:
            se = base[f"se_{exp}"]
            overlaps = np.abs(base[f"gi_score_{exp}"]) < 1.96 * se
            base[f"gi_score_overlaps_zero_{exp}"] = overlaps.where(se.notna(), other=pd.NA)
        print(f"  SE1 range: [{base['se_exp1'].min():.4f}, {base['se_exp1'].max():.4f}]")
        print(f"  SE2 range: [{base['se_exp2'].min():.4f}, {base['se_exp2'].max():.4f}]")
    else:
        print("  (SE columns not found; skipping se_exp and gi_score_overlaps_zero columns)")

    # Drop pairs with a missing SE in either screen so this base matches the pair
    # set run_joint_model fits: it drops NaN-SE pairs (they are not fittable), so
    # keeping them here would make the joint-summary coverage check below flag a
    # phantom mismatch when both scripts are run directly on raw result_summary
    # copies. (The example path never hits this — run_per_screen_mixture already
    # drops NaN-SE rows upstream.)
    if have_se_cols:
        n_before = len(base)
        base = base.dropna(subset=["se_exp1", "se_exp2"]).reset_index(drop=True)
        if n_before - len(base):
            print(f"  Dropped {n_before - len(base)} pair(s) with a missing SE "
                  f"(not fittable by the joint model).")

    # `correlation` is optional upstream: the paper's HPC aggregation carries a
    # per-pair guide-level correlation, but the simplified public
    # aggregate_guide_pairs.py does not. Always surface both columns (NaN where
    # absent) so the merged table keeps a stable 20-column schema either way.
    for exp in ["exp1", "exp2"]:
        col = f"correlation_{exp}"
        if col not in base.columns:
            base[col] = np.nan

    return base


# ──────────────────────────────────────────────────────────────────
# Merge the joint quadrant-model summary onto the base DataFrame
# ──────────────────────────────────────────────────────────────────

# Rename Stan-derived per-pair probability columns -> human-readable names.
# The joint model emits <name>_joint (+ _median/_lo/_hi credible-interval variants);
# only the mean (no suffix) columns survive into the 20-column schema.
RENAME_MAP = {
    "prob_interaction_joint": "prob_interaction",
    "prob_interaction_joint_median": "prob_interaction_median",
    "prob_interaction_joint_lo": "prob_interaction_lo",
    "prob_interaction_joint_hi": "prob_interaction_hi",
    "prob_aggr_joint": "prob_aggravating",
    "prob_aggr_joint_median": "prob_aggravating_median",
    "prob_aggr_joint_lo": "prob_aggravating_lo",
    "prob_aggr_joint_hi": "prob_aggravating_hi",
    "prob_allev_joint": "prob_alleviating",
    "prob_allev_joint_median": "prob_alleviating_median",
    "prob_allev_joint_lo": "prob_alleviating_lo",
    "prob_allev_joint_hi": "prob_alleviating_hi",
    "prob_disc_joint": "prob_discordant",
    "prob_disc_joint_median": "prob_discordant_median",
    "prob_disc_joint_lo": "prob_discordant_lo",
    "prob_disc_joint_hi": "prob_discordant_hi",
    "prob_null_joint": "prob_no_interaction",
    "prob_null_joint_median": "prob_no_interaction_median",
    "prob_null_joint_lo": "prob_no_interaction_lo",
    "prob_null_joint_hi": "prob_no_interaction_hi",
    "prob_conc_joint": "prob_concordant",
    "prob_conc_joint_median": "prob_concordant_median",
    "prob_conc_joint_lo": "prob_concordant_lo",
    "prob_conc_joint_hi": "prob_concordant_hi",
}


def merge_summary(base_df, summary_path):
    """Merge base per-screen columns with the joint quadrant-model summary.

    Renames the Stan-derived <class>_joint probabilities to their final names and
    left-joins them onto the base DataFrame by orf_pair.
    """
    print(f"\nLoading joint summary: {summary_path}")
    summary = pd.read_csv(summary_path, sep="\t")
    print(f"  {len(summary)} rows, {len(summary.columns)} columns")

    summary = summary.rename(columns=RENAME_MAP)

    # Keep orf_pair + all renamed probability columns (drops log_lik_*, y1/y2, etc.)
    prob_cols = [c for c in summary.columns if c.startswith("prob_")]
    summary = summary[["orf_pair"] + prob_cols]

    # The joint summary must cover exactly the base pairs, one row each: a missing
    # pair would left-join to NaN joint probabilities (a silent hole), a duplicate
    # would multiply rows.
    dup = summary["orf_pair"].duplicated()
    if dup.any():
        raise ValueError(
            f"Joint summary has {int(dup.sum())} duplicate orf_pair rows "
            f"(e.g. {summary.loc[dup, 'orf_pair'].head(3).tolist()}).")
    missing = set(base_df["orf_pair"]) - set(summary["orf_pair"])
    if missing:
        raise ValueError(
            f"Joint summary is missing {len(missing)} of the {len(base_df)} merged "
            f"pairs (e.g. {sorted(missing)[:3]}); their joint probabilities would be "
            f"NaN. Run the joint model on the same pair set as the merge.")

    merged = base_df.merge(summary, on="orf_pair", how="left", validate="one_to_one")

    # Belt-and-suspenders: no NaN in the core class probabilities after the join.
    core = [c for c in MODEL_PROB_ORDER if c in merged.columns]
    if core:
        n_nan = int(merged[core].isna().any(axis=1).sum())
        if n_nan:
            raise ValueError(
                f"{n_nan} merged pairs have NaN joint probabilities after the merge.")
    return merged


# ──────────────────────────────────────────────────────────────────
# Output column ordering / precision
# ──────────────────────────────────────────────────────────────────

BASE_COLS = [
    "orf1", "orf2", "name1", "name2",
    "gi_score_exp1", "se_exp1", "gi_score_overlaps_zero_exp1",
    "gi_score_exp2", "se_exp2", "gi_score_overlaps_zero_exp2",
    "correlation_exp1", "correlation_exp2",
    "prob_interaction_median_exp1", "prob_interaction_median_exp2",
]

MODEL_PROB_ORDER = [
    "prob_interaction",
    "prob_aggravating",
    "prob_alleviating",
    "prob_discordant",
    "prob_no_interaction",
    "prob_concordant",
]

# Credible-interval variants dropped from the final table
DROP_SUFFIXES = ["_median", "_lo", "_hi"]

# Redundant exp2 identifier columns (identical to exp1 after the inner join)
DROP_COLS = ["orf1_exp2", "orf2_exp2", "name1_exp2", "name2_exp2"]

# Strip the _exp1 suffix from the surviving identifier columns
ID_RENAME_MAP = {
    "orf1_exp1": "orf1",
    "orf2_exp1": "orf2",
    "name1_exp1": "name1",
    "name2_exp1": "name2",
}

# Column name substring -> decimal places
PRECISION = {
    "gi_score": 2,
    "se_": 2,
    "correlation": 3,
    "prob_": 5,
}


def apply_precision(df):
    """Round float columns to the decimals specified by name substring."""
    for col in df.columns:
        for pattern, decimals in PRECISION.items():
            if pattern in col and df[col].dtype in ("float64", "float32"):
                df[col] = df[col].round(decimals)
                break
    return df


def order_columns(df):
    """Return df with the 14 base columns first, then the 6 model probability
    columns.  Drops _median/_lo/_hi variants, redundant exp2 identifiers, and
    orf_pair; strips _exp1 from identifier columns; applies precision."""
    # Drop credible-interval variants, but keep prob_interaction_median_exp1/exp2
    drop_cols = [c for c in df.columns
                 if any(c.endswith(suf) for suf in DROP_SUFFIXES)
                 and c not in BASE_COLS]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop redundant exp2 identifiers and the join key
    df = df.drop(columns=[c for c in DROP_COLS + ["orf_pair"] if c in df.columns],
                 errors="ignore")

    # Rename exp1 identifier columns to their canonical names
    df = df.rename(columns={k: v for k, v in ID_RENAME_MAP.items() if k in df.columns})

    df = apply_precision(df)

    base = [c for c in BASE_COLS if c in df.columns]
    model = [c for c in MODEL_PROB_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in base and c not in model]
    return df[base + model + extra]


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge joint quadrant-model results into one flat 20-column TSV."
    )
    parser.add_argument("--exp1-tsv", required=True,
                        help="Per-screen result_summary_long_df TSV for experiment 1")
    parser.add_argument("--exp2-tsv", required=True,
                        help="Per-screen result_summary_long_df TSV for experiment 2")
    parser.add_argument("--summary-tsv", required=True,
                        help="Joint model summary TSV from run_joint_model.py")
    parser.add_argument("--output", required=True,
                        help="Output path for the merged TSV")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Joint GI merge (quadrant model)")
    print("=" * 60)

    base_df = load_base_df(args.exp1_tsv, args.exp2_tsv)
    merged = merge_summary(base_df, args.summary_tsv)
    merged = order_columns(merged)

    # Strongest interactions first
    if "prob_no_interaction" in merged.columns:
        merged = merged.sort_values("prob_no_interaction", ascending=True)

    merged.to_csv(args.output, sep="\t", index=False)
    print(f"\n  Saved: {args.output}  ({len(merged)} rows, {len(merged.columns)} cols)")


if __name__ == "__main__":
    main()
