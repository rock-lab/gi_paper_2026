#!/usr/bin/env python3
"""
check_example.py — validate a finished run of the shipped example dataset.

This is EXAMPLE-SPECIFIC: it knows the five "Set A" signal pairs and compares a
freshly produced run against the golden tables in ``example_data/expected/``.
``run_example.sh`` calls it as its last step. If you adapt the runbook to your
own data, drop this step (there is no golden reference for arbitrary data).

For each known pair it prints the single-screen call (δ′ and P per screen) and
the joint call, and asserts the joint dominant class is correct. When the golden
table is present it also prints an INFORMATIONAL comparison to the published
probabilities — those are expected to differ (the toy re-fits the global mixture
on ~120 pairs) and do NOT gate pass/fail; only the class calls do.

Usage:
    python check_example.py --single1 S1.tsv --single2 S2.tsv --merged MERGED.tsv \\
        [--golden-dir example_data/expected]
"""

import argparse
import glob
import os
import sys

import pandas as pd

# Known Set A interactions: (orfA, orfB, expected dominant class, label).
CHECKS = [
    ("RVBD1854c", "RVBD0392c", "aggravating", "ndh-ndhA"),
    ("RVBD0050",  "RVBD3682",  "aggravating", "ponA1-ponA2"),
    ("RVBD2754c", "RVBD2764c", "aggravating", "thyX-thyA"),
    ("RVBD2193",  "RVBD2200c", "alleviating", "ctaE-ctaC"),
    ("RVBD1304",  "RVBD3795",  "alleviating", "atpB-embB"),
]
# Structural expectations for the shipped example's merged joint table.
EXPECTED_PAIRS = 120       # 15 Set A genes -> 105 unordered + 15 self-pairs
EXPECTED_COLS = 20         # the documented 20-column schema
EXPECTED_GENES = 15        # gene x gene matrices are EXPECTED_GENES square
CLASS_COL = {"aggravating": "prob_aggravating",
             "alleviating": "prob_alleviating",
             "discordant":  "prob_discordant"}
PROB_COLS = ["prob_interaction", "prob_aggravating", "prob_alleviating",
             "prob_discordant", "prob_no_interaction", "prob_concordant"]


def get_pair(frame, o1, o2):
    a, b = sorted([o1, o2])                       # canonical orf1 < orf2
    row = frame[(frame["orf1"] == a) & (frame["orf2"] == b)]
    if len(row) != 1:
        raise SystemExit(f"ERROR: pair {a}_{b} matched {len(row)} rows")
    return row.iloc[0]


def one_screen(frame, o1, o2):
    try:
        r = get_pair(frame, o1, o2)
        return f"δ′={float(r['delta_prime_median']):+.2f} P={float(r['prob_interaction_median']):.3f}"
    except SystemExit:
        return "n/a"


def structural_errors(m):
    """Return a list of structural problems with the merged joint table (empty if
    it matches the documented example contract: shape, unique pairs, full probs)."""
    errs = []
    if len(m) != EXPECTED_PAIRS:
        errs.append(f"expected {EXPECTED_PAIRS} rows, got {len(m)}")
    if m.shape[1] != EXPECTED_COLS:
        errs.append(f"expected {EXPECTED_COLS} columns, got {m.shape[1]}")
    if {"orf1", "orf2"}.issubset(m.columns) and m.duplicated(subset=["orf1", "orf2"]).any():
        errs.append(f"{int(m.duplicated(subset=['orf1', 'orf2']).sum())} duplicate (orf1,orf2) pairs")
    missing = [c for c in PROB_COLS if c not in m.columns]
    if missing:
        errs.append(f"missing probability columns: {missing}")
    else:
        n_nan = int(m[PROB_COLS].isna().any(axis=1).sum())
        if n_nan:
            errs.append(f"{n_nan} rows with NaN probability values")
    return errs


def matrix_errors(path, kind):
    """Return a list of problems with a gene x gene matrix TSV (empty if it is a
    square EXPECTED_GENES x EXPECTED_GENES table)."""
    import pandas as _pd
    try:
        mat = _pd.read_csv(path, sep="\t", index_col=0)
    except Exception as e:                       # noqa: BLE001 - report, don't crash
        return [f"{kind} matrix unreadable ({path}): {e}"]
    n = EXPECTED_GENES
    if mat.shape != (n, n):
        return [f"{kind} matrix expected {n}x{n}, got {mat.shape[0]}x{mat.shape[1]}"]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single1", required=True, help="exp1 per-screen gene-pair table")
    ap.add_argument("--single2", required=True, help="exp2 per-screen gene-pair table")
    ap.add_argument("--merged", required=True, help="merged 20-column joint table")
    ap.add_argument("--golden-dir", default="example_data/expected",
                    help="dir with the golden merged_*.tsv (default: example_data/expected)")
    ap.add_argument("--signed-matrix", help="optional signed_prob_matrix.tsv to shape-check")
    ap.add_argument("--hit-matrix", help="optional hit-matrix TSV to shape-check")
    args = ap.parse_args()

    s1 = pd.read_csv(args.single1, sep="\t")
    s2 = pd.read_csv(args.single2, sep="\t")
    m = pd.read_csv(args.merged, sep="\t")

    # Structural contract: the merged table must be the documented shape, with
    # unique pairs and no missing probabilities; optional matrices must be square.
    struct_errs = structural_errors(m)
    if args.signed_matrix:
        struct_errs += matrix_errors(args.signed_matrix, "signed")
    if args.hit_matrix:
        struct_errs += matrix_errors(args.hit_matrix, "hit")
    has_mtx = bool(args.signed_matrix or args.hit_matrix)
    print("Structural checks on the merged table" + (" + matrices" if has_mtx else "") + ":")
    if struct_errs:
        for e in struct_errs:
            print(f"  [FAIL] {e}")
    else:
        print(f"  [OK] {EXPECTED_PAIRS} rows x {EXPECTED_COLS} cols, unique pairs, "
              f"complete probabilities"
              + (f", matrices {EXPECTED_GENES}x{EXPECTED_GENES}" if has_mtx else ""))
    print()

    print("Known Set A pairs — single-screen call vs joint call:")
    print("  pass = correct interaction DIRECTION (expected class dominates the other two")
    print("  interaction classes). Clearing the 0.50 hit threshold is reported separately:")
    print("  borderline pairs can fall short when the global mixture is re-fit on only ~120")
    print("  pairs (the toy re-fit differs from the paper's full-cohort fit).")
    ok = True
    n_confident = 0
    for o1, o2, klass, label in CHECKS:
        r = get_pair(m, o1, o2)
        col = CLASS_COL[klass]
        p = float(r[col])
        other = max(float(r[c]) for c in CLASS_COL.values() if c != col)
        direction_ok = p > other            # expected class dominates the other interaction classes
        confident = p >= 0.5                # clears the 0.50 hit threshold
        n_confident += int(confident)
        ok = ok and direction_ok
        print(f"  {label:13s} expected {klass}")
        print(f"      single-screen  exp1[{one_screen(s1, o1, o2)}]  exp2[{one_screen(s2, o1, o2)}]")
        print(f"      joint          {col}={p:.3f} (next interaction class={other:.3f})  "
              f"direction={'OK' if direction_ok else 'WRONG'}, "
              f"{'confident hit (>=0.5)' if confident else 'sub-threshold (<0.5)'}")

    golden = sorted(glob.glob(os.path.join(args.golden_dir, "merged_*.tsv")))
    if golden:
        g = pd.read_csv(golden[0], sep="\t")
        print(f"\nComparison vs the PUBLISHED (golden) values in {golden[0]}:")
        print("  NOTE: these are *expected to differ*. The joint mixture is a global fit;")
        print("  the toy re-estimates it from ~120 interaction-enriched pairs, whereas the")
        print("  published probabilities were fit across all ~290k pairs. This comparison")
        print("  is informational and does NOT affect pass/fail (the class calls above do).")
        for o1, o2, klass, label in CHECKS:
            r = get_pair(m, o1, o2)
            e = get_pair(g, o1, o2)
            col = CLASS_COL[klass]
            print(f"  {label:13s} {col}: toy={float(r[col]):.3f}  published={float(e[col]):.3f}  "
                  f"(Δ={abs(float(r[col]) - float(e[col])):.3f})")
    else:
        print(f"\n(no golden table in {args.golden_dir}/; skipped published-value comparison)")

    print()
    if struct_errs:
        sys.exit("EXAMPLE CHECK: FAILED — merged table / matrix failed structural checks")
    if not ok:
        sys.exit("EXAMPLE CHECK: FAILED — a known pair has the wrong interaction class")
    print(f"EXAMPLE CHECK: PASSED — all {len(CHECKS)} known pairs have the correct interaction "
          f"class + structural checks\n              "
          f"({n_confident}/{len(CHECKS)} also clear the 0.50 hit threshold)")


if __name__ == "__main__":
    main()
