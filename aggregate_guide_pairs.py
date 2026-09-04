#!/usr/bin/env python3
"""
Aggregate guide-pair GI scores up to the gene-pair level.

This is the "missing seam" between the per-guide-pair scoring step
(``gi_scoring.py`` -> ``gi_scores.tsv`` / ``gi_scores_corrected.tsv``) and the
per-screen probabilistic mixture model (``run_per_screen_mixture.py``). The
mixture model consumes a
``result_summary_long_df``-schema TSV whose rows are *gene pairs* with a point
GI score (``delta_prime_median``) and a measurement-error standard error
(``sd_delta_prime_median``). ``gi_scoring.py`` only emits *guide-pair* rows, so
this script collapses the several guide-pairs that target a given gene pair down
to one row per canonical ``orf_pair``.

-------------------------------------------------------------------------------
IMPORTANT -- this is an APPROXIMATE reproduction of the HPC aggregation
-------------------------------------------------------------------------------
The published pipeline aggregated guide-pair posteriors on the cluster with a
richer per-pair model (which also produced ``correlation``, credible intervals,
and a posterior ``sd_delta_prime_median``). This script reproduces the essential
transformation -- median GI score per gene pair plus a documented SE -- so that
readers can run the full chain on the toy data. It will NOT reproduce the HPC
numbers to the last decimal. For an exact reproduction, use the pre-derived
golden inputs shipped in ``example_data/gi_input/`` (those ARE the exact HPC
``result_summary_long_df`` values, restricted to the 15-gene toy set).

-------------------------------------------------------------------------------
What it does
-------------------------------------------------------------------------------
1. Load a guide-pair ``gi_scores(_corrected).tsv`` with the ``gi_scoring.py``
   output schema:
       guide_pair, guide1, guide2, y25_double, y25_single1, y25_single2,
       y25_expected, y25_delta [, y25_delta_corrected], y25_std, ...
   Each ``guide_pair`` follows the construct grammar
       ORF1:name1_SEQ1_ORF2:name2_SEQ2
   (a single-mutant control side is the literal ``Negative_SEQ``).
2. Parse ``orf1/orf2`` and ``name1/name2`` from the two guide sides.
3. DROP single-mutant rows (any side that is a Negative / non-targeting
   control) -- only gene x gene double mutants are aggregated.
4. Form the canonical, order-independent ``orf_pair`` = sorted "orf1_orf2"
   (self pairs orf1 == orf2 are kept), reordering names to match.
5. GROUP BY ``orf_pair`` and aggregate that gene pair's guide-pairs:
       delta_prime_median   = median of the per-guide-pair GI score
                              (GAM-corrected ``y25_delta_corrected`` if the
                              column exists, else raw ``y25_delta``;
                              overridable with --score-col)
       sd_delta_prime_median = SE estimate -- see below.

-------------------------------------------------------------------------------
SE definition (sd_delta_prime_median)  -- pooled measurement error
-------------------------------------------------------------------------------
We treat each guide-pair's posterior SD (``y25_std``) as the measurement noise
of one observation of the gene-pair GI score and pool them as

        sd_delta_prime_median = sqrt( mean(y25_std**2) ) / sqrt(n)

where n is the number of guide-pairs for the gene pair. This is the standard
error of an average of n noisy measurements with (pooled) per-measurement SD
sqrt(mean(y25_std**2)). Properties:
  * For a gene pair with a single guide-pair (n = 1) it reduces to that pair's
    own posterior SD ``y25_std`` -- a sensible, non-degenerate fallback.
  * It shrinks as more guide-pairs corroborate the gene pair (1/sqrt(n)).
  * It propagates the trajectory-model measurement error that the downstream
    mixture model expects in its ``se`` slot.

We deliberately pick this ONE definition and document it. An alternative would
be the empirical spread of the guide-pair deltas (std(delta)/sqrt(n)); we use
that only as a fallback when ``y25_std`` is unavailable. Note ``delta_prime_median``
is a *median* while this SE is the SE of a *mean*; for the small, fairly
symmetric guide-pair sets here the two centers coincide closely and this is an
acceptable approximation (again: the golden inputs carry the exact HPC SE).

-------------------------------------------------------------------------------
Output (result_summary_long_df subset consumed by run_per_screen_mixture.py)
-------------------------------------------------------------------------------
Tab-separated, with a leading unnamed integer index column (so it round-trips
through ``pd.read_csv(..., index_col=0)`` exactly like the HPC files):
    orf1, orf2, orf_pair, delta_prime_median, sd_delta_prime_median,
    prob_interaction_median, name1, name2, n_guide_pairs
``prob_interaction_median`` is left blank (NaN) -- it is filled in later by the
per-screen mixture model. ``n_guide_pairs`` is informational (ignored downstream).

Usage:
    python aggregate_guide_pairs.py <input_tsv> <output_tsv> [--score-col COL]

Example:
    python aggregate_guide_pairs.py \\
        example_out/exp1/gi/gi_scores_corrected.tsv \\
        example_out/exp1/result_summary_exp1.tsv
"""

import argparse
import sys

import numpy as np
import pandas as pd


# Control-side markers (matches gi_scoring.GIScoring.is_negative_control).
NEGATIVE_TERMS = ("Negative", "NT", "Non-targeting", "NonTargeting")

# Score columns tried, in order, when --score-col is not given.
SCORE_COL_CANDIDATES = ("y25_delta_corrected", "y25_delta")


def is_negative_orf(orf: str) -> bool:
    """True if an ORF token is a negative/non-targeting control side."""
    return any(term in orf for term in NEGATIVE_TERMS)


def parse_side(side: str):
    """
    Parse one guide side into (orf, name).

    A targeting side is 'ORF:name' (e.g. 'RVBD0001:dnaA'); a control side is
    the bare token 'Negative'. For the control case orf == name == the token,
    which is later dropped by the Negative filter.
    """
    if ":" in side:
        orf, name = side.split(":", 1)
    else:
        orf, name = side, side
    return orf, name


def side_from_guide(guide: str) -> str:
    """
    Extract the 'ORF:name' side token from a full guide string
    'ORF:name_SEQ'. The ORF:name portion never contains '_', so split on the
    first underscore.
    """
    return guide.split("_", 1)[0]


def sides_from_guide_pair(guide_pair: str):
    """
    Fallback parse of both sides directly from a 'guide_pair' string of the
    form 'ORF1:name1_SEQ1_ORF2:name2_SEQ2' (4 underscore-delimited tokens).
    Returns (side1, side2) as 'ORF:name' tokens.
    """
    parts = guide_pair.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Cannot parse guide_pair {guide_pair!r}: expected 4 "
            f"underscore-tokens (ORF1:name1_SEQ1_ORF2:name2_SEQ2), got "
            f"{len(parts)}. Provide guide1/guide2 columns."
        )
    return parts[0], parts[2]


def load_guide_pairs(input_tsv: str) -> pd.DataFrame:
    """Load a gi_scoring guide-pair TSV, tolerating a stray index column."""
    df = pd.read_csv(input_tsv, sep="\t")
    # Drop an accidental leading unnamed index column if present.
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if "guide_pair" not in df.columns and not (
        "guide1" in df.columns and "guide2" in df.columns
    ):
        raise ValueError(
            "Input must have a 'guide_pair' column or both 'guide1' and "
            f"'guide2' columns; found: {list(df.columns)}"
        )
    return df


def resolve_score_col(df: pd.DataFrame, score_col: str = None) -> str:
    """Choose the GI-score column to aggregate."""
    if score_col is not None:
        if score_col not in df.columns:
            raise ValueError(
                f"--score-col {score_col!r} not in input columns: "
                f"{list(df.columns)}"
            )
        return score_col
    for cand in SCORE_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(
        "No GI-score column found. Expected one of "
        f"{SCORE_COL_CANDIDATES} or pass --score-col; got {list(df.columns)}"
    )


def annotate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add orf1/orf2/name1/name2 (canonical, orf-sorted) and orf_pair to each
    guide-pair row; mark single-mutant (Negative) rows for dropping.
    """
    have_guides = "guide1" in df.columns and "guide2" in df.columns

    orf1s, orf2s, name1s, name2s, orf_pairs, keep = [], [], [], [], [], []
    for _, row in df.iterrows():
        if have_guides:
            side_a = side_from_guide(str(row["guide1"]))
            side_b = side_from_guide(str(row["guide2"]))
        else:
            side_a, side_b = sides_from_guide_pair(str(row["guide_pair"]))

        orf_a, name_a = parse_side(side_a)
        orf_b, name_b = parse_side(side_b)

        # Drop the row if either side is a control (single mutant).
        if is_negative_orf(orf_a) or is_negative_orf(orf_b):
            orf1s.append(None); orf2s.append(None)
            name1s.append(None); name2s.append(None)
            orf_pairs.append(None); keep.append(False)
            continue

        # Canonical, order-independent gene pair (self pairs allowed).
        if orf_a <= orf_b:
            o1, n1, o2, n2 = orf_a, name_a, orf_b, name_b
        else:
            o1, n1, o2, n2 = orf_b, name_b, orf_a, name_a

        orf1s.append(o1); orf2s.append(o2)
        name1s.append(n1); name2s.append(n2)
        orf_pairs.append(f"{o1}_{o2}"); keep.append(True)

    out = df.copy()
    out["orf1"] = orf1s
    out["orf2"] = orf2s
    out["name1"] = name1s
    out["name2"] = name2s
    out["orf_pair"] = orf_pairs
    out["_keep"] = keep
    return out


def pooled_se(std_values: np.ndarray, delta_values: np.ndarray) -> float:
    """
    sd_delta_prime_median = sqrt(mean(std^2)) / sqrt(n)  (documented above).

    Falls back to the empirical spread of the guide-pair deltas,
    std(delta, ddof=1)/sqrt(n), when no valid per-guide-pair y25_std is present.
    """
    std_values = np.asarray(std_values, dtype=float)
    std_values = std_values[np.isfinite(std_values)]
    n = len(delta_values)
    if std_values.size > 0:
        # Pool over the guide-pairs that carry a posterior SD; scale the
        # measurement error by 1/sqrt(n) over all guide-pairs in the group.
        return float(np.sqrt(np.mean(std_values ** 2)) / np.sqrt(n))
    # Fallback: empirical spread of the deltas (needs >= 2 observations).
    if n >= 2:
        return float(np.std(delta_values, ddof=1) / np.sqrt(n))
    return float("nan")


def aggregate(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Collapse annotated guide-pair rows to one row per gene pair."""
    df = df[df["_keep"]].copy()
    df = df.dropna(subset=[score_col])
    if df.empty:
        raise ValueError(
            "No double-mutant guide-pairs left after dropping controls / NaN "
            f"scores in column {score_col!r}."
        )

    has_std = "y25_std" in df.columns

    records = []
    for orf_pair, grp in df.groupby("orf_pair", sort=True):
        deltas = grp[score_col].to_numpy(dtype=float)
        std_vals = (
            grp["y25_std"].to_numpy(dtype=float)
            if has_std
            else np.array([], dtype=float)
        )
        records.append({
            "orf1": grp["orf1"].iloc[0],
            "orf2": grp["orf2"].iloc[0],
            "orf_pair": orf_pair,
            "delta_prime_median": float(np.median(deltas)),
            "sd_delta_prime_median": pooled_se(std_vals, deltas),
            "prob_interaction_median": np.nan,  # filled by the mixture model
            "name1": grp["name1"].iloc[0],
            "name2": grp["name2"].iloc[0],
            "n_guide_pairs": int(len(grp)),
        })

    out = pd.DataFrame.from_records(records, columns=[
        "orf1", "orf2", "orf_pair",
        "delta_prime_median", "sd_delta_prime_median",
        "prob_interaction_median", "name1", "name2", "n_guide_pairs",
    ])
    out = out.sort_values("orf_pair").reset_index(drop=True)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate guide-pair GI scores to gene-pair level "
                    "(approximate reproduction of the HPC aggregation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_tsv",
                        help="guide-pair gi_scores(_corrected).tsv")
    parser.add_argument("output_tsv",
                        help="output result_summary_long_df-schema TSV")
    parser.add_argument("--score-col", default=None,
                        help="GI-score column to aggregate "
                             "(default: y25_delta_corrected if present, "
                             "else y25_delta)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    df = load_guide_pairs(args.input_tsv)
    score_col = resolve_score_col(df, args.score_col)
    print(f"Loaded {len(df):,} guide-pair rows from {args.input_tsv}")
    print(f"Aggregating score column: {score_col}"
          f"{'' if score_col != 'y25_delta_corrected' else ' (GAM-corrected)'}")

    annotated = annotate_pairs(df)
    n_singles = int((~annotated["_keep"]).sum())
    if n_singles:
        print(f"Dropped {n_singles:,} single-mutant (control) guide-pair rows")

    result = aggregate(annotated, score_col)

    # Write with a leading unnamed integer index column so the file matches the
    # HPC result_summary_long_df format read via pd.read_csv(index_col=0).
    result.to_csv(args.output_tsv, sep="\t", index=True)

    print(f"Wrote {len(result):,} gene pairs to {args.output_tsv}")
    print(f"  GI score range: [{result['delta_prime_median'].min():.3f}, "
          f"{result['delta_prime_median'].max():.3f}]")
    finite_se = result["sd_delta_prime_median"].dropna()
    if len(finite_se):
        print(f"  SE range:       [{finite_se.min():.4f}, "
              f"{finite_se.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
