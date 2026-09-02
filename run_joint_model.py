#!/usr/bin/env python3
"""
Run the joint genetic-interaction (GI) probability model on two screens.

This is a slimmed, self-contained joint-model driver, hardcoded to the
single PRIMARY model used in the paper:

    joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan

The model is a 5-component Normal / half-smeared-Uniform mixture fit jointly
to two per-screen GI scores (with per-pair measurement error). It classifies
each gene pair into: no interaction (correlated bivariate-Normal null),
concordant aggravating (Q4: both < 0), concordant alleviating (Q2: both > 0),
or discordant (Q1/Q3: opposite signs). The winsorized uniform support keeps
the interaction density tight so the null competes fairly.

Inputs are two per-screen `result_summary_long_df` TSVs (one per experiment),
each carrying `orf_pair`, `delta_prime_median` (= per-screen GI score) and
`sd_delta_prime_median` (= its standard error). The two screens are inner-joined
on `orf_pair`.

Output is a per-pair summary TSV whose class-probability columns are named so
that `merge_joint_results.py` can pick them up directly:

    <output_prefix>_quad_me_trunc_halfsmeared_w<PCT>_summary.tsv

Example
-------
    python run_joint_model.py \
        example_data/gi_input/result_summary_long_df_exp1_toy.tsv \
        example_data/gi_input/result_summary_long_df_exp2_toy.tsv \
        example_out/joint

Dependencies: cmdstanpy, numpy, pandas (plus a working CmdStan install).
"""

import os
import argparse

import numpy as np
import pandas as pd
import cmdstanpy


# ──────────────────────────────────────────────────────────────────
# Fixed model spec (the PRIMARY model — no multi-model registry)
# ──────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STAN = os.path.join(
    HERE, "joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan"
)

# Pinned hyperpriors (see build_stan_data). rho_theta / kappa_theta are declared
# in the Stan data block but UNUSED by this model; they are supplied only so the
# data block validates.
HYPERPRIORS = {
    "mu_mu": 0,
    "sigma_mu": 2,
    "a_sigma": 2,
    "b_sigma": 0.1,
    "rho_theta": 0.5,
    "kappa_theta": 2,
}

# Generated-quantity log-prob vars -> short name used in the merge rename_map.
# (log_pZ1 is handled separately as prob_interaction_joint.)
EXTRA_VARS = {
    "log_p_conc": "prob_conc",   # concordant  (aggravating + alleviating)
    "log_p_disc": "prob_disc",   # discordant
    "log_p_null": "prob_null",   # no interaction
    "log_p_aggr": "prob_aggr",   # aggravating (Q4: both < 0)
    "log_p_allev": "prob_allev",  # alleviating (Q2: both > 0)
}


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def load_and_merge(tsv1_path, tsv2_path, load_se=True):
    """Load two per-screen TSVs and inner-join on ``orf_pair``.

    Keeps only ``orf_pair``, ``delta_prime_median`` (the per-screen GI score),
    and ``sd_delta_prime_median`` (its standard error). A leading unnamed index
    column is tolerated. Pairs missing an SE in either screen are dropped.

    Returns a DataFrame with columns ``orf_pair``,
    ``delta_prime_median_exp1/_exp2`` and ``sd_delta_prime_median_exp1/_exp2``.
    """
    keep_cols = {"", "orf_pair", "delta_prime_median"}
    if load_se:
        keep_cols.add("sd_delta_prime_median")

    def _load(path, label):
        print(f"Loading {label}: {path}")
        df = pd.read_csv(path, sep="\t", index_col=0,
                         usecols=lambda c: c in keep_cols)
        if "orf_pair" not in df.columns:
            df = df.reset_index()
        print(f"  {len(df)} pairs")
        return df

    df1 = _load(tsv1_path, "Exp1")
    df2 = _load(tsv2_path, "Exp2")

    print("Merging on orf_pair (inner join)...")
    merged = df1.merge(df2, on="orf_pair", suffixes=("_exp1", "_exp2"))
    print(f"  {len(merged)} shared pairs")

    if load_se:
        se1_col, se2_col = "sd_delta_prime_median_exp1", "sd_delta_prime_median_exp2"
        if se1_col not in merged.columns or se2_col not in merged.columns:
            raise ValueError(
                "This model requires sd_delta_prime_median in BOTH input TSVs.\n"
                f"  Available columns: {list(merged.columns)}"
            )
        n_missing = merged[se1_col].isna().sum() + merged[se2_col].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} missing SE values; dropping those pairs")
            merged = merged.dropna(subset=[se1_col, se2_col])
        print(f"  SE1 range: [{merged[se1_col].min():.4f}, {merged[se1_col].max():.4f}]")
        print(f"  SE2 range: [{merged[se2_col].min():.4f}, {merged[se2_col].max():.4f}]")

    return merged


# ──────────────────────────────────────────────────────────────────
# Stan data construction
# ──────────────────────────────────────────────────────────────────

def build_stan_data(merged, include_se=True, winsorize_pct=0.10):
    """Build the Stan data dict for the half-smeared quadrant model.

    The uniform interaction support on each axis is winsorized to the
    ``[pct, 100 - pct]`` percentile range (default 0.10 -> [0.10, 99.90] pct);
    this tightens the uniform bounds so the null does not win by default.
    The quadrant origin is fixed at 0. Per-pair measurement errors se1/se2 are
    passed through for the measurement-error null and origin smearing.

    ``include_se`` defaults to True and should stay True for this measurement-
    error model (se1/se2 are required by the Stan program); the flag exists only
    so callers (e.g. the demo notebook) can share this one builder explicitly.
    """
    y1 = merged["delta_prime_median_exp1"].values.copy()
    y2 = merged["delta_prime_median_exp2"].values.copy()

    lo1, hi1 = np.percentile(y1, [winsorize_pct, 100 - winsorize_pct])
    lo2, hi2 = np.percentile(y2, [winsorize_pct, 100 - winsorize_pct])
    n_clip1 = int(((y1 < lo1) | (y1 > hi1)).sum())
    n_clip2 = int(((y2 < lo2) | (y2 > hi2)).sum())
    y1 = np.clip(y1, lo1, hi1)
    y2 = np.clip(y2, lo2, hi2)
    print(f"  Winsorized to [{winsorize_pct}, {100 - winsorize_pct}] percentile:")
    print(f"    y1: [{lo1:.2f}, {hi1:.2f}], clipped {n_clip1:,} pairs")
    print(f"    y2: [{lo2:.2f}, {hi2:.2f}], clipped {n_clip2:,} pairs")

    data = {
        "N": len(merged),
        "y1": y1.tolist(),
        "y2": y2.tolist(),
        "min_y1": float(np.min(y1)),
        "max_y1": float(np.max(y1)),
        "min_y2": float(np.min(y2)),
        "max_y2": float(np.max(y2)),
        "origin_y1": 0.0,
        "origin_y2": 0.0,
    }
    if include_se:
        data["se1"] = merged["sd_delta_prime_median_exp1"].values.tolist()
        data["se2"] = merged["sd_delta_prime_median_exp2"].values.tolist()
    data.update(HYPERPRIORS)
    return data


# ──────────────────────────────────────────────────────────────────
# Fit + extraction
# ──────────────────────────────────────────────────────────────────

def run_model(stan_file, stan_data, orf_pairs, y1, y2,
              n_chains, n_warmup, n_sampling, seed):
    """Compile, sample, and extract per-pair class probabilities.

    ``y1`` / ``y2`` are the ORIGINAL (un-winsorized) per-screen GI scores; they
    are stored in the summary for readability, while the fit uses the winsorized
    support inside ``stan_data``.

    For each generated-quantity log-prob vector we take exp per draw then average
    over draws (posterior-mean probability), plus the posterior median and a 95%
    credible interval.
    """
    print(f"\nCompiling Stan model: {stan_file}")
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    # Reasonable inits for the simplex + scales (aids convergence of the mixture).
    inits = {
        "mu1": 0.0, "mu2": 0.0,
        "sigma1": float(np.std(y1)), "sigma2": float(np.std(y2)),
        "rho": 0.15,
        "mix_weights": [0.05, 0.05, 0.90],
    }

    print(f"Sampling ({n_chains} chains, {n_warmup} warmup + {n_sampling} sampling, seed={seed})...")
    fit = model.sample(
        data=stan_data,
        chains=n_chains,
        iter_warmup=n_warmup,
        iter_sampling=n_sampling,
        seed=seed,
        inits=inits,
    )

    def _probs(log_var):
        p = np.exp(fit.stan_variable(log_var))  # (n_draws, N)
        return (p.mean(axis=0),
                np.median(p, axis=0),
                np.percentile(p, 2.5, axis=0),
                np.percentile(p, 97.5, axis=0))

    print("Computing posterior class probabilities...")
    p_mean, p_med, p_lo, p_hi = _probs("log_pZ1")
    summary = pd.DataFrame({
        "orf_pair": orf_pairs,
        "delta_prime_median_exp1": y1,
        "delta_prime_median_exp2": y2,
        "prob_interaction_joint": p_mean,
        "prob_interaction_joint_median": p_med,
        "prob_interaction_joint_lo": p_lo,
        "prob_interaction_joint_hi": p_hi,
    })

    for log_var, short in EXTRA_VARS.items():
        m, med, lo, hi = _probs(log_var)
        summary[f"{short}_joint"] = m
        summary[f"{short}_joint_median"] = med
        summary[f"{short}_joint_lo"] = lo
        summary[f"{short}_joint_hi"] = hi

    return summary


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the primary joint GI probability model on two screens."
    )
    parser.add_argument("tsv1", help="result_summary_long_df TSV for experiment 1")
    parser.add_argument("tsv2", help="result_summary_long_df TSV for experiment 2")
    parser.add_argument("output_prefix", help="Output path prefix for the summary TSV")
    parser.add_argument("--winsorize-pct", type=float, default=0.10,
                        help="Winsorize each axis to [pct, 100-pct] percentiles (default: 0.10)")
    parser.add_argument("--chains", type=int, default=8, help="Number of chains (default: 8)")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup iterations (default: 1000)")
    parser.add_argument("--sampling", type=int, default=1000, help="Sampling iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=456, help="RNG seed (default: 456)")
    parser.add_argument("--stan", default=DEFAULT_STAN,
                        help="Path to the Stan model (default: the half-smeared quadrant model)")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Joint GI probability model (quad_me_trunc_halfsmeared)")
    print("=" * 60)

    merged = load_and_merge(args.tsv1, args.tsv2, load_se=True)
    stan_data = build_stan_data(merged, winsorize_pct=args.winsorize_pct)

    # Original (un-winsorized) GI scores + pair ids for the summary.
    y1 = merged["delta_prime_median_exp1"].values
    y2 = merged["delta_prime_median_exp2"].values
    orf_pairs = merged["orf_pair"].values

    summary = run_model(
        stan_file=args.stan,
        stan_data=stan_data,
        orf_pairs=orf_pairs, y1=y1, y2=y2,
        n_chains=args.chains,
        n_warmup=args.warmup,
        n_sampling=args.sampling,
        seed=args.seed,
    )

    # Tag encodes the winsorize percentile, e.g. 0.10 -> "w010" so the merge
    # step can locate this file: <prefix>_quad_me_trunc_halfsmeared_w010_summary.tsv
    wpct_str = f"{args.winsorize_pct:.2f}".replace(".", "")
    tag = f"quad_me_trunc_halfsmeared_w{wpct_str}"
    out_path = f"{args.output_prefix}_{tag}_summary.tsv"
    summary.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved summary ({len(summary)} pairs, {len(summary.columns)} cols) to {out_path}")

    # Call breakdown. Default call threshold is 0.5; 0.95 is a stricter
    # "confident" cutoff. A pair is called by its dominant interaction class.
    p_int = summary["prob_interaction_joint"].values
    for thr in (0.5, 0.95):
        n = int((p_int >= thr).sum())
        print(f"  interacting pairs (P_interaction >= {thr}): {n:,} / {len(summary):,}")


if __name__ == "__main__":
    main()
