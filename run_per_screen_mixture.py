#!/usr/bin/env python3
"""
Per-screen (1D) Normal-Uniform mixture model for genetic-interaction calls.

This is the *per-screen* step that sits between per-guide-pair GI scoring
(gi_scoring.py -> result_summary_long_df) and the *joint* two-experiment
model (run_joint_model.py). For a single screen it fits a two-
component mixture to the gene-pair GI scores:

  - Null component:        Normal(mu, sigma), variance inflated per pair by
                           the GI-score standard error (measurement error).
  - Interaction component: Uniform(min_y, max_y), optionally "smeared" by the
                           same per-pair measurement error (CDF convolution).

The posterior probability that a pair belongs to the interaction component,
`prob_interaction_median`, is written back onto the input table. That column
is one of the inputs the joint model uses to *sign* interactions
(signed_maxmag), so this driver documents/wires the previously-orphaned
per-screen mixture step.

Two Stan back-ends are supported (choose with --stan):
  - normal_uniform_mix.stan                 (DEFAULT) the model the individual
    screens were called with; uses the GI score only (SE is ignored).
  - univariate_normal_uniform_mix_me.stan   measurement-error variant; uses the
    per-pair SE for both the inflated null and the smeared uniform (the 1-D
    analog of the joint model).

Usage:
    python run_per_screen_mixture.py <input_tsv> <output_tsv> [OPTIONS]

Example:
    python run_per_screen_mixture.py \\
        example_data/gi_input/result_summary_exp1.tsv \\
        example_data/gi_input/result_summary_exp1_with_prob.tsv \\
        --winsorize-pct 0.10

Input:
    A per-screen result_summary_long_df TSV. Required columns:
      orf_pair, delta_prime_median (GI score, used as y),
      sd_delta_prime_median (its standard error, used as se).
    Any other columns (orf1, orf2, name1, name2, correlation, ...) are
    carried through unchanged.

Output:
    The same table with a `prob_interaction_median` column added/overwritten
    (= posterior mean of exp(log_pZ1), i.e. P(interaction | GI score) averaged
    over draws). The output is a drop-in result_summary_long_df for the joint
    model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cmdstanpy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default Stan back-end: the 1D measurement-error mixture.
DEFAULT_STAN = "normal_uniform_mix.stan"

# Sampling configuration is pinned to match the joint model (paper defaults).
N_WARMUP = 1000
N_SAMPLING = 1000

# Required input columns.
Y_COL = "delta_prime_median"       # per-screen GI score -> y
SE_COL = "sd_delta_prime_median"   # its standard error   -> se
PAIR_COL = "orf_pair"


def load_data(tsv_path):
    """Load a per-screen result_summary_long_df TSV.

    Keeps every column (so the output is a drop-in result_summary), but
    requires orf_pair, delta_prime_median and sd_delta_prime_median.
    Handles the leading unnamed index column present in real result_summary
    files, and drops rows missing a GI score or SE.
    """
    print(f"Loading {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)

    # Real result_summary files carry a leading unnamed index column; orf_pair
    # is then a regular column. If it landed in the index instead, recover it.
    if PAIR_COL not in df.columns:
        df = df.reset_index()

    if PAIR_COL not in df.columns:
        raise ValueError(f"Missing column: {PAIR_COL}")
    if Y_COL not in df.columns:
        raise ValueError(f"Missing column: {Y_COL}")
    if SE_COL not in df.columns:
        raise ValueError(
            f"Missing column: {SE_COL}. The measurement-error model needs the "
            f"GI-score standard error. (HPC result_summary copies carry it; "
            f"some redistributed copies do not.)"
        )

    n_before = len(df)
    df = df.dropna(subset=[Y_COL, SE_COL]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} rows with missing GI score / SE")

    print(f"  {len(df):,} gene pairs")
    print(f"  GI score range: [{df[Y_COL].min():.2f}, {df[Y_COL].max():.2f}]")
    print(f"  SE range:       [{df[SE_COL].min():.4f}, {df[SE_COL].max():.4f}]")
    return df


def build_stan_data(df, winsorize_pct=0.10):
    """Build the 1D Stan data dictionary.

    Winsorization clips the GI scores to the [pct, 100-pct] percentile range
    before setting the uniform bounds (min_y, max_y). This concentrates the
    interaction (uniform) density onto the plausible support instead of a few
    extreme outliers. The raw (unclipped) GI scores are still what get a
    probability assigned; only the fitted bounds/null see the clipped values.

    The `se` field is used by the measurement-error model and simply ignored
    by the no-ME variant, so the same data dict works for both.
    """
    y = df[Y_COL].values.copy()
    se = df[SE_COL].values.copy()

    if winsorize_pct is not None and winsorize_pct > 0:
        lo, hi = np.percentile(y, [winsorize_pct, 100 - winsorize_pct])
        n_clip = int(((y < lo) | (y > hi)).sum())
        y = np.clip(y, lo, hi)
        print(f"  Winsorized to [{winsorize_pct}, {100 - winsorize_pct}] pct: "
              f"y in [{lo:.2f}, {hi:.2f}], clipped {n_clip:,} pairs")

    data = {
        "N": len(y),
        "y": y.tolist(),
        "se": se.tolist(),
        "min_y": float(np.min(y)),
        "max_y": float(np.max(y)),
        # Null centered at the origin (no interaction => GI score 0).
        "mu_mu": 0.0,
        "sigma_mu": 2.0,
        "a_sigma": 2.0,
        "b_sigma": 0.1,
        # Prior on the interaction fraction: ~5% of pairs interact per screen.
        "rho_theta": 0.05,
        "kappa_theta": 2.0,
    }

    if data["max_y"] <= data["min_y"]:
        raise ValueError(
            f"Degenerate GI-score support: min_y == max_y == {data['min_y']:.4f}. "
            f"Need >= 2 distinct (winsorized) GI scores to fit the mixture.")

    print(f"  Uniform bounds: [{data['min_y']:.2f}, {data['max_y']:.2f}] "
          f"(density {1.0 / (data['max_y'] - data['min_y']):.4f})")
    return data


def resolve_stan_path(stan_arg):
    """Resolve a Stan model argument to a file path.

    Accepts a bare filename (resolved next to this script) or an explicit path.
    """
    if os.path.isabs(stan_arg) or os.path.exists(stan_arg):
        return stan_arg
    return os.path.join(SCRIPT_DIR, stan_arg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-screen 1D Normal-Uniform mixture model for GI calls."
    )
    parser.add_argument("input_tsv",
                        help="Per-screen result_summary_long_df TSV")
    parser.add_argument("output_tsv",
                        help="Output TSV (input + prob_interaction_median)")
    parser.add_argument("--stan", default=DEFAULT_STAN,
                        help=f"Stan model file (default: {DEFAULT_STAN}; "
                             f"use univariate_normal_uniform_mix_me.stan for the ME variant)")
    parser.add_argument("--winsorize-pct", type=float, default=0.10,
                        help="Percentile for winsorizing the uniform support "
                             "(default: 0.10 -> clip to [0.10, 99.90] pct)")
    parser.add_argument("--chains", type=int, default=8,
                        help="Number of MCMC chains (default: 8)")
    parser.add_argument("--seed", type=int, default=456,
                        help="Random seed (default: 456)")
    return parser.parse_args()


def main():
    args = parse_args()

    stan_path = resolve_stan_path(args.stan)

    print("=" * 60)
    print("  Per-screen Normal-Uniform mixture model")
    print("=" * 60)
    print(f"  Input:      {args.input_tsv}")
    print(f"  Output:     {args.output_tsv}")
    print(f"  Stan model: {stan_path}")
    print(f"  Winsorize:  {args.winsorize_pct} pct")
    print(f"  Sampling:   {args.chains} chains x "
          f"{N_WARMUP} warmup + {N_SAMPLING} sampling, seed {args.seed}")

    if not os.path.exists(stan_path):
        raise FileNotFoundError(f"Stan model not found: {stan_path}")

    # Load data and build the 1D Stan input.
    df = load_data(args.input_tsv)
    data = build_stan_data(df, winsorize_pct=args.winsorize_pct)

    out_dir = os.path.dirname(os.path.abspath(args.output_tsv))
    os.makedirs(out_dir, exist_ok=True)

    # Compile and sample.
    print(f"\n  Compiling Stan model...")
    model = cmdstanpy.CmdStanModel(stan_file=stan_path)

    inits = {
        "mu": 0.0,
        "sigma": float(np.std(df[Y_COL].values)),
        "theta": 0.05,
    }

    print(f"  Sampling...")
    fit = model.sample(
        data=data,
        chains=args.chains,
        iter_warmup=N_WARMUP,
        iter_sampling=N_SAMPLING,
        seed=args.seed,
        inits=inits,
    )

    # Report the global mixture parameters.
    print(f"\n  Global parameter estimates (median [95% CI]):")
    for name in ["mu", "sigma", "theta"]:
        vals = fit.stan_variable(name)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"    {name:8s}: {np.median(vals):7.4f}  [{lo:.4f}, {hi:.4f}]")

    # Per-pair interaction probability = posterior mean of exp(log_pZ1).
    # (Historically named *_median for schema compatibility; it is the mean of
    # the per-draw probabilities, matching the joint model's input column.)
    log_pZ1 = fit.stan_variable("log_pZ1")          # (n_draws, N)
    prob_interaction = np.exp(log_pZ1).mean(axis=0)  # (N,)
    df["prob_interaction_median"] = prob_interaction

    # Write the augmented table with a leading unnamed index column so it is a
    # drop-in result_summary_long_df for run_joint_model.py.
    df.to_csv(args.output_tsv, sep="\t", index=True)
    print(f"\n  Saved: {args.output_tsv} ({len(df):,} pairs)")

    # Call summary at the default and stricter thresholds.
    n_total = len(df)
    for thr in [0.50, 0.95]:
        n_hit = int((prob_interaction >= thr).sum())
        label = "default" if thr == 0.50 else "confident"
        print(f"    P >= {thr:.2f} ({label}): {n_hit:,} pairs "
              f"({100 * n_hit / n_total:.1f}%)")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
