#!/usr/bin/env bash
# =============================================================================
# run_example.sh — the runbook.
#
# Runs the WHOLE genetic-interaction pipeline on the shipped example dataset:
# each screen separately (counts -> single-screen results) and then the joint
# cross-screen model. Every step below is a discrete Python script you can also
# run by hand; this shell script just calls them in order.
#
# USE IT AS A TEMPLATE FOR YOUR OWN DATA:
#   1. Copy this file (e.g. `run_myproject.sh`).
#   2. Edit the CONFIG block: point each screen's metadata at your own metadata
#      CSV (format: see example_data/experiment_metadata_exp1.csv), and set the
#      output dir / parameters. Your counts must follow the dual-construct id
#      grammar documented in example_data/README.md.
#   3. Delete the "EXAMPLE-ONLY validation" block at the very bottom (it knows
#      the shipped example's answers; it does not apply to your data).
#   4. Run it: `bash run_myproject.sh`.
#
# Requires: a CmdStan toolchain (cmdstanpy); GAM correction needs `pygam`
# (or R + mgcv). Run from the repo root.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ============================ CONFIG (edit me) ===============================
PY="${PYTHON:-python}"                   # python interpreter (override: PYTHON=...)
OUTDIR="${OUTDIR:-example_out}"          # where all outputs go
WORKERS="${WORKERS:-4}"                  # parallel workers for the per-guide-pair model
WINSORIZE_PCT="${WINSORIZE_PCT:-0.10}"   # winsorization percentile for the mixtures

# One block per screen: a short label + the path to that screen's metadata CSV.
SCREEN1_LABEL="exp1"; SCREEN1_META="example_data/experiment_metadata_exp1.csv"
SCREEN2_LABEL="exp2"; SCREEN2_META="example_data/experiment_metadata_exp2.csv"

# EXAMPLE-ONLY shortcut: set FROM_GOLDEN=1 to skip the heavy per-guide-pair
# fitting and start the joint step from the shipped golden gene-pair tables.
# (Only works for the shipped example; leave 0 for real data.)
FROM_GOLDEN="${FROM_GOLDEN:-0}"
# =============================================================================

mkdir -p "$OUTDIR"

# --- per-screen chain: counts -> single-screen gene-pair results -------------
# run_screen <label> <metadata_csv>  ->  writes $OUTDIR/<label>/single_screen_<label>.tsv
run_screen () {
  local label="$1" meta="$2" d="$OUTDIR/$1"
  mkdir -p "$d"

  echo ">>> [$label] step 1/5  counts -> log2 fold-change"
  "$PY" logfc_tools.py --metadata "$meta" --output "$d/logfc.txt" --normalize

  echo ">>> [$label] step 2/5  log2FC -> per-guide-pair GI (two-line Stan model)"
  # For very large libraries this step can be chunked (see gi_scoring.py --help).
  "$PY" gi_scoring.py --logfc_data "$d/logfc.txt" --output_dir "$d/gi" --workers "$WORKERS"

  echo ">>> [$label] step 3/5  GAM de-trending (optional; needs pygam or R+mgcv)"
  "$PY" gi_scoring.py --output_dir "$d/gi" --step 4 || echo "    (GAM skipped)"

  local scores="$d/gi/gi_scores_corrected.tsv"
  [ -f "$scores" ] || scores="$d/gi/gi_scores.tsv"

  echo ">>> [$label] step 4/5  guide-pairs -> gene-pairs (aggregate)"
  "$PY" aggregate_guide_pairs.py "$scores" "$d/result_summary_$label.tsv"

  echo ">>> [$label] step 5/5  per-screen 1-D mixture -> single-screen P(interaction)"
  "$PY" run_per_screen_mixture.py "$d/result_summary_$label.tsv" \
        "$d/single_screen_$label.tsv" --winsorize-pct "$WINSORIZE_PCT"
}

if [ "$FROM_GOLDEN" = "1" ]; then
  echo ">>> FROM_GOLDEN=1: using shipped golden gene-pair tables (skipping per-screen fitting)"
  SINGLE1="example_data/gi_input/result_summary_long_df_${SCREEN1_LABEL}_toy.tsv"
  SINGLE2="example_data/gi_input/result_summary_long_df_${SCREEN2_LABEL}_toy.tsv"
else
  run_screen "$SCREEN1_LABEL" "$SCREEN1_META"
  run_screen "$SCREEN2_LABEL" "$SCREEN2_META"
  SINGLE1="$OUTDIR/$SCREEN1_LABEL/single_screen_$SCREEN1_LABEL.tsv"
  SINGLE2="$OUTDIR/$SCREEN2_LABEL/single_screen_$SCREEN2_LABEL.tsv"
fi

# --- joint analysis: both screens together -----------------------------------
echo ">>> [joint] step 1/4  fit joint cross-screen quadrant model (Stan)"
"$PY" run_joint_model.py "$SINGLE1" "$SINGLE2" "$OUTDIR/joint" --winsorize-pct "$WINSORIZE_PCT"

# run_joint_model wrote <prefix>_<tag>_summary.tsv (tag encodes the model +
# winsorize). Reconstruct the exact name the same way the Python does
# (0.10 -> "010"), so this doesn't depend on which files happen to match a glob.
WPCT=$(printf "%.2f" "$WINSORIZE_PCT" | tr -d '.')
SUMMARY="$OUTDIR/joint_quad_me_trunc_halfsmeared_w${WPCT}_summary.tsv"
MERGED="$OUTDIR/merged.tsv"

echo ">>> [joint] step 2/4  merge -> 20-column per-pair table"
"$PY" merge_joint_results.py --exp1-tsv "$SINGLE1" --exp2-tsv "$SINGLE2" \
      --summary-tsv "$SUMMARY" --output "$MERGED"

echo ">>> [joint] step 3/4  signed-probability gene x gene matrix"
"$PY" make_signed_prob_matrix.py --merged-tsv "$MERGED" --value-mode signed_maxmag \
      --output "$OUTDIR/signed_prob_matrix.tsv"

echo ">>> [joint] step 4/4  boolean hit matrix (threshold 0.5)"
"$PY" make_hit_matrix.py --merged-tsv "$MERGED" --threshold 0.5 \
      --output "$OUTDIR/hit_matrix_thr050.tsv"

echo
echo "Done. Outputs in $OUTDIR/:"
if [ "$FROM_GOLDEN" = "1" ]; then
  echo "  merged joint table ($MERGED) and gene x gene matrices."
  echo "  (per-screen tables were read in place from example_data/gi_input/, not written here.)"
else
  echo "  per-screen single-screen tables, merged joint table ($MERGED), and gene x gene matrices."
fi

# ============ EXAMPLE-ONLY validation — DELETE for your own data =============
echo
echo ">>> validating known Set A interactions + structure vs the golden tables"
"$PY" check_example.py --single1 "$SINGLE1" --single2 "$SINGLE2" --merged "$MERGED" \
      --signed-matrix "$OUTDIR/signed_prob_matrix.tsv" \
      --hit-matrix "$OUTDIR/hit_matrix_thr050.tsv"
# =============================================================================
