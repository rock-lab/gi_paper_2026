# Genetic Interaction Screens in Mycobacterium tuberculosis and M. smegmatis

Repository containing the code and methods for genetic interaction screens analysis in M. tuberculosis and M. smegmatis.

> **In a hurry?** Jump to [Reproduce on toy data](#reproduce-on-toy-data-joint-model-quickstart) to run the joint genetic-interaction probability model end-to-end on 10 real genes with one command, or read the model write-up in [`docs/MODEL.md`](docs/MODEL.md).

## Code

Below is a short legend describing the main files, followed by examples running the code:

#### process_reads.py
Python script for processing FASTQ files containing sgRNA sequencing reads. This script:
- Builds subread alignment indices from sgRNA library FASTA files
- Aligns reads to the library using the subread aligner
- Counts aligned reads for each sgRNA
- Processes multiple samples in parallel
- Merges count and diagnostic files across samples

#### subread.py
Python wrapper functions for interacting with the subread aligner. Provides functions for:
- Building alignment indices (`build_index`)
- Aligning reads to references (`align`)
- Feature counting (`featureCounts`)

**Note:** Requires subread to be installed via your OS package manager (e.g., `sudo apt install subread`)

#### counting_tools.py
Comprehensive Python module containing tools for:
- Reading FASTA files
- DNA sequence manipulation (reverse complement, etc.)
- Creating revised sgRNA libraries with constant sequences
- Quality control filtering of aligned reads
- Counting reads from BAM files
- Merging count files across samples
- Diagnostic reporting and statistics

#### logfc_tools.py
Python module for calculating log2 fold-changes from sgRNA count data. This script:
- Calculates log2FC between +ATC and -ATC conditions across multiple time points
- Handles multiple replicates with configurable summary statistics (mean, median, etc.)
- Applies limit of detection filtering and pseudocount corrections
- Supports negative control normalization
- Processes experimental metadata to organize passaging experiments

#### gi_scoring.py
Genetic interaction scoring pipeline using Bayesian modeling. This script:
- Processes log2FC data to identify all guide pairs for interaction analysis
- Prepares model data in JSON format for Stan (memory-efficient chunked processing)
- Runs Bayesian two-line fitness models for each guide pair independently
- Calculates genetic interaction scores and confidence intervals from model posterior samples
- Designed for large-scale datasets (>2M guide pairs) with parallel processing capabilities

#### run_per_screen_mixture.py
Fits the per-screen 1D Normal-Uniform interaction mixture (`normal_uniform_mix.stan`, the model the individual screens were called with; a measurement-error variant is available via `--stan`) to a `result_summary_long_df` TSV, turning each per-screen GI score into a probability of interaction. See Step 4.

#### run_joint_model.py
Fits the joint cross-screen quadrant mixture (the primary `joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan` model) to two screens, classifying every gene pair as aggravating / alleviating / discordant / no-interaction. See Step 5.

#### merge_joint_results.py
Merges the per-screen and joint outputs into a single 20-column per-pair results table.

#### make_signed_prob_matrix.py / make_hit_matrix.py
Project the per-pair probabilities onto gene x gene matrices: a signed probability-of-interaction matrix (diverging, centered at 0) and a boolean hit matrix.

#### run_example.sh / make_example_data.py / check_example.py
`run_example.sh` is the **runbook**: a shell script that calls each discrete step script in order — for every screen it runs counts → single-screen results, then the joint analysis. Copy it and edit the CONFIG block to run your own data. `make_example_data.py` cuts the 15-gene example dataset from the source data (maintainer-only); `check_example.py` validates a finished example run against the golden tables. See [Reproduce on the example data](#reproduce-on-the-example-data).

## Dependencies

Python dependencies are pinned in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
# then, once, build the CmdStan toolchain that cmdstanpy uses for sampling:
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Core packages: `cmdstanpy` (Stan interface), `numpy`, `scipy`, `pandas`, `tqdm`.

Optional / step-specific:
- `subread` - only for the optional FASTQ -> counts Step 1 (install via your OS package manager, e.g. `sudo apt install subread`).
- `pysam` - BAM file handling in Step 1.
- `pygam` - optional Python GAM correction in Step 3 (alternative: R + `mgcv`).
- `pyarrow` - optional, faster IO for very large guide-pair datasets.
- `matplotlib`, `seaborn` - only for the demo notebooks / plotting.

External:
- `Stan` / CmdStan - probabilistic programming backend, compiled by `cmdstanpy`; needed only for the sampling steps (Steps 4-5).
- `R` with `mgcv` - optional alternative to `pygam` for the GAM correction.

## Usage Examples

### Step 1: Process FASTQ files and generate sgRNA counts

```bash
python process_reads.py sample1.fastq.gz sample2.fastq.gz --library sgRNA_library.fasta --output_dir ./results --workers 5 --mm 1
```

Key parameters:
- Input FASTQ files (positional arguments)
- `--library`: Path to sgRNA library FASTA file
- `--output_dir`: Directory for output files (default: ./BAM_and_Counts)
- `--workers`: Number of parallel workers (default: 5)
- `--mm`: Maximum mismatches allowed in alignment (default: 1)
- `--make_rev_lib`: Create reverse complement library with constant sequences
- `--upstream`: Upstream constant sequence for library construction
- `--downstream`: Downstream constant sequence for library construction

### Step 2: Calculate log2 fold-changes from count data

First, create an experiment metadata file describing your passaging experiment:

```bash
python logfc_tools.py --template --metadata experiment_metadata.csv
```

This creates a template CSV file with the required columns:
- `strain`: Strain name (e.g., 'H37Rv', 'Msm')
- `experiment`: Experiment identifier
- `condition`: Sample/timepoint identifier
- `atc`: ATC condition ('plus' or 'minus')
- `generations`: Number of generations/passages
- `replicate`: Replicate number
- `count_file_path`: Path to the corresponding .counts file

Then calculate log2FC values:

```bash
python logfc_tools.py --metadata experiment_metadata.csv --output logfc_results.txt --normalize --lod_limit 20
```

Key parameters:
- `--metadata`: Path to experiment metadata CSV file
- `--output`: Path to output log2FC dataframe
- `--normalize`: Apply negative control normalization
- `--lod_limit`: Limit of detection for filtering low counts (default: 20.0)
- `--summary_metric`: How to summarize replicates (mean, median, etc.)
- `--pseudo`: Pseudocount for log2FC calculation (default: 1.0)

### Step 3: Calculate genetic interaction scores

Run the full genetic interaction scoring pipeline:

```bash
python gi_scoring.py --logfc_data logfc_results.txt --output_dir ./gi_analysis --workers 8
```

For large datasets (>2M guide pairs), process in chunks:

```bash
# Step 1: Prepare model data in chunks
python gi_scoring.py --logfc_data logfc_results.txt --step 1 --start 0 --end 500000 --workers 8
python gi_scoring.py --logfc_data logfc_results.txt --step 1 --start 500000 --end 1000000 --workers 8
# ... continue for all chunks

# Step 2: Run Bayesian models (memory-intensive, use fewer workers)
python gi_scoring.py --step 2 --start 0 --end 500000 --workers 4
python gi_scoring.py --step 2 --start 500000 --end 1000000 --workers 4

# Step 3: Calculate Y25_delta (uncorrected GI scores)
python gi_scoring.py --step 3 --output_dir ./gi_analysis

# Step 4 (Optional): Apply GAM correction to GI scores
python gi_scoring.py --step 4 --output_dir ./gi_analysis
```

The GAM correction (Step 4) adjusts for systematic biases in the genetic interaction scores based on the expected fitness values. This step can use either Python (pygam) or R (mgcv) for the correction.

Key parameters:
- `--logfc_data`: Path to log2FC dataframe from Step 2
- `--output_dir`: Directory for all GI analysis outputs
- `--step`: Run specific step only (1: data prep, 2: modeling, 3: Y25_delta calculation, 4: GAM correction)
- `--start/--end`: Indices for chunked processing of large datasets
- `--workers`: Number of parallel workers (use fewer for Stan models)
- `--force`: Overwrite existing files

### Step 4: Per-screen interaction probability

Steps 1-3 produce, for each gene pair, a per-screen GI score
(`delta_prime_median`) and its standard error (`sd_delta_prime_median`). To turn
each score into a **probability of interaction**, fit a 1D Normal-Uniform
mixture (`normal_uniform_mix.stan`, the model the individual screens were called
with): the bulk of non-interacting pairs is a Normal centered near zero, and
genuine interactions form the diffuse Uniform component.

```bash
python run_per_screen_mixture.py \
    result_summary_long_df_<screen>.tsv \
    per_screen_out \
    --winsorize-pct 0.10
```

Input contract - the `result_summary_long_df` TSV must contain (a leading
unnamed index column may be present):
- `orf1`, `orf2`, `orf_pair` - the canonical sorted `"orf1_orf2"` gene-pair id
- `delta_prime_median` - the per-screen GI score
- `sd_delta_prime_median` - its standard error
- `name1`, `name2` - gene names (carried through)
- `correlation` - mean guide-guide correlation for the pair (carried through)

The model writes a per-pair `prob_interaction_median` (posterior probability
that the pair is an interaction), which is the column the joint model and the
merge step consume.

> A measurement-error variant, `univariate_normal_uniform_mix_me.stan` — the 1-D
> analog of the joint model, which additionally uses each pair's SE to inflate
> the null and smear the interaction component — is available via
> `--stan univariate_normal_uniform_mix_me.stan`.

### Step 5: Joint cross-screen quadrant model

Two screens (here 100 ng and 500 ng ATc) are modeled **jointly**. Each gene pair
has a GI score in each screen; the joint model is a Normal-Uniform mixture over
the 2D `(score_exp1, score_exp2)` plane with soft quadrant boundaries and
per-pair measurement error - the "half-smeared" truncated model
`joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan`. It assigns every
pair a probability of being:
- **concordant** - an interaction with the same sign in both screens, split into
  **aggravating** (both negative) and **alleviating** (both positive),
- **discordant** - an interaction with opposite signs across screens,
- **no interaction** - the central Normal null.

```bash
python run_joint_model.py \
    result_summary_long_df_exp1.tsv \
    result_summary_long_df_exp2.tsv \
    joint_out \
    --winsorize-pct 0.10
```

Parameters (pinned for reproducibility):
- Sampling: 8 chains x (1000 warmup + 1000 sampling), `seed=456`.
- Support **winsorized** per axis to the `[0.10, 99.90]` percentile range
  (`--winsorize-pct 0.10`), which tightens the Uniform bounds around the
  interaction density; the null **origin is fixed at 0**.
- Priors: `mu_mu=0`, `sigma_mu=2`, `a_sigma=2`, `b_sigma=0.1`.

Per-pair generated quantities: `log_pZ1` (any interaction) plus the class
log-probabilities `log_p_conc`, `log_p_disc`, `log_p_null`, `log_p_aggr`,
`log_p_allev`, and `log_lik`.

Merge the per-screen and joint outputs into one per-pair table:

```bash
python merge_joint_results.py \
    --exp1-tsv result_summary_long_df_exp1.tsv \
    --exp2-tsv result_summary_long_df_exp2.tsv \
    --summary-tsv joint_out_quad_me_trunc_halfsmeared_w010_summary.tsv \
    --output joint_mixture/merged_quad_me_trunc_halfsmeared_w010.tsv
```

The merged TSV (`merged_quad_me_trunc_halfsmeared_w010.tsv`) has 20 columns:

```
orf1, orf2, name1, name2,
gi_score_exp1, se_exp1, gi_score_overlaps_zero_exp1,
gi_score_exp2, se_exp2, gi_score_overlaps_zero_exp2,
correlation_exp1, correlation_exp2,
prob_interaction_median_exp1, prob_interaction_median_exp2,
prob_interaction, prob_aggravating, prob_alleviating,
prob_discordant, prob_no_interaction, prob_concordant
```

Finally, project the per-pair probabilities onto gene x gene matrices:

```bash
# signed probability (aggravating negative, alleviating positive); the sign is
# taken from whichever screen has the larger-magnitude GI score:
python make_signed_prob_matrix.py --value-mode signed_maxmag

# boolean hit matrix; a pair is a hit if ANY directional class prob >= threshold:
python make_hit_matrix.py --threshold 0.5
```

We call a pair an interaction at `prob >= 0.5`; `0.95` is a stricter "confident"
cutoff used for high-precision hit lists.

The derivation of the per-screen `delta_prime` GI score and the optional GAM
correction are documented in [`docs/MODEL.md`](docs/MODEL.md); a worked
walk-through of the joint model lives in the `joint_model_demo` notebook under
[`notebooks/`](notebooks/).

## Reproduce on the example data

The repository ships a tiny, fully **real** example dataset: **15 published genes**
("Set A") across **both screens** (100 ng and 500 ng ATc) = 105 gene pairs
(+ 15 self-pairs). The set is deliberately *representative* — most pairs do not
interact, with a handful of clear, biologically sensible signals: three
**negative** (aggravating) redundant-isoenzyme synthetic lethals (`ndh`/`ndhA`
NADH dehydrogenases, `thyX`/`thyA` thymidylate synthases, `ponA1`/`ponA2`
class-A PBPs) and two **positive** (alleviating) pairs (`ctaE`/`ctaC` cytochrome
oxidase, `atpB`/`embB`), plus five inert filler genes. It contains real
sequencing counts (`example_data/counts/`, both screens), the real per-screen
guide-pair GI scores aggregated to the gene-pair `result_summary_long_df` inputs
(`example_data/gi_input/`), and golden expected outputs (`example_data/expected/`).

`run_example.sh` is the single, end-to-end **runbook** — a shell script that
calls each discrete step script in order. For **each** screen it runs counts →
log2FC → the per-guide-pair two-line model → GAM correction →
guide-pair→gene-pair aggregation → the per-screen mixture (the **single-screen**
results); then it runs the **joint** cross-screen model, merges to the 20-column
table, and builds the gene × gene matrices. To run your **own** data, copy the
script and edit the CONFIG block at the top (point each screen at your metadata
CSV, set the output names) — every step is a plain script you can also run by hand.

```bash
# maintainer-only: (re)build the example data from the raw sources
python make_example_data.py

# full example: both screens from counts, then the joint analysis
bash run_example.sh

# fast path: skip the heavy per-guide-pair fitting and run only the joint half
# from the shipped golden gene-pair tables (one joint fit over ~120 pairs)
FROM_GOLDEN=1 bash run_example.sh
```

The final step validates the shipped example (known Set A hits vs the golden
tables via `check_example.py`); delete that block when adapting the runbook.

## Scope & honesty

This public repository reproduces the **joint genetic-interaction probability
model** on a small toy dataset. A few caveats stated plainly:

- The per-screen **guide-pair GI scores are real** (from the published screens),
  but the public **guide -> gene-pair aggregation is approximate** relative to
  the full HPC pipeline (which layers on additional filtering, GAM correction,
  and hierarchical pooling). For that reason we **ship the golden per-screen
  inputs** in `example_data/gi_input/*.tsv` rather than asking you to reproduce them
  bit-for-bit from counts.
- Because MCMC sampling is stochastic, the toy checks use **tolerances**, not
  exact equality, and the shipped `example_data/expected/` values are golden
  checkpoints rather than a single uniquely-correct answer.
- The interactive per-pair web bundle used for the paper's browsable figures is
  **intentionally excluded** from this repository.

## Output Files

### Step 1 Output (Count Generation)
- Individual `.bam` and `.sorted.bam` files for each sample
- `.counts` files containing sgRNA read counts
- `.diagnostics` files with alignment statistics
- `merged_*_counts.txt` - merged count matrix across all samples
- `merged_*_diagnostics.txt` - merged diagnostic statistics

### Step 2 Output (Log2FC Calculation)
- Log2FC dataframe in tab-delimited format with columns:
  - `strain`: Bacterial strain
  - `experiment`: Experiment identifier
  - `G`: Generation/passage number
  - `ORF`: Gene/ORF identifier
  - `SEQ`: sgRNA sequence identifier
  - `ID`: Full sgRNA ID
  - `Y`: Log2 fold-change (+ATC/-ATC)
  - `exp_mean`: Mean count in experimental (+ATC) condition
  - `ctrl_mean`: Mean count in control (-ATC) condition
  - `GOOD`: Quality flag based on limit of detection

### Step 3 Output (Genetic Interaction Scoring)
- `model_data/`: JSON files for each guide pair (Stan model input)
- `samples/`: Stan model posterior samples for each guide pair
- `results/gi_scores.tsv`: Genetic interaction scores with columns:
  - `guide_pair`: Combined guide pair identifier
  - `guide1`, `guide2`: Individual guide names
  - `y25_double`: Y25 prediction for double mutant
  - `y25_single1`, `y25_single2`: Y25 predictions for single mutants
  - `y25_expected`: Sum of single mutant Y25s
  - `y25_delta`: Uncorrected genetic interaction score (Y25_double - Y25_expected)
  - `y25_std`: Standard deviation of Y25 prediction
  - `y25_q025`, `y25_q975`: 95% confidence interval bounds
- `results/gi_scores_corrected.tsv` (after Step 4): GAM-corrected GI scores with additional columns:
  - `y25_delta_corrected`: GAM-corrected genetic interaction score
  - `gam_prediction`: GAM model prediction (if using Python method)

## Testing and validation

`run_example.sh` (see [Reproduce on the example data](#reproduce-on-the-example-data))
doubles as the test harness: it runs the full pipeline — both single screens and
the joint model — on the shipped 15-gene example dataset, and its final step
(`check_example.py`) asserts that every known Set A interaction has the correct
**direction** (its expected class dominates the other interaction classes), and
separately reports which pairs also clear the 0.5 hit threshold. It prints, for
reference, how the toy's probabilities compare to the published (golden)
`example_data/expected/` values.

In a clean run all four known pairs are directionally correct; the three strong
signals (`ndh`/`ndhA`, `ponA1`/`ponA2`, `ctaE`/`ctaC`) clear 0.5, while the
weakest positive (`atpB`/`embB`) stays directionally correct but can fall
sub-threshold — the joint mixture is re-fit on only ~120 pairs, so its null is
estimated from far fewer points than the published fit and borderline calls are
sensitive to it.

**The toy probabilities are expected to differ from the published numbers**, so
the class calls — not the exact values — are the pass/fail criterion. The main
reasons:

- **Global-mixture re-fit (largest effect)** — the per-screen and joint mixtures
  are *global* models. On the example they are re-estimated from only ~120
  interaction-enriched pairs, whereas the published values were fit across all
  ~290k pairs, so the learned null (and hence the probabilities) differ. The
  dominant class is unaffected. (For example, `ctaE`/`ctaC` scores ~0.88 in the
  full fit but ~1.0 on the toy — same call, alleviating.)
- **Guide→gene aggregation** — the public `aggregate_guide_pairs.py` is a
  documented approximation of the internal HPC step (see
  [Scope & honesty](#scope--honesty)).
- **Stan sampling / GAM** — seeded MCMC still varies by CmdStan/compiler version,
  and Python (`pygam`) vs R (`mgcv`) GAM differ slightly.

## Pipeline Overview

This repository provides a three-step pipeline for processing genetic interaction screen data:

1. **Read Processing & Counting**: Convert FASTQ files to sgRNA count matrices
2. **Log2FC Calculation**: Calculate log2 fold-changes between +ATC/-ATC conditions across multiple time points
3. **Genetic Interaction Scoring**: Use Bayesian modeling to identify genetic interactions between gene pairs

The pipeline is designed for large-scale CRISPRi passaging experiments where:
- sgRNA libraries target genes of interest in M. tuberculosis and M. smegmatis
- Samples are collected at multiple time points during passaging
- Each time point has both +ATC (CRISPRi ON) and -ATC (CRISPRi OFF) conditions
- Log2FC values represent the fitness effect of gene knockdown over time
- Genetic interaction scores quantify non-additive fitness effects between gene pairs

The pipeline can handle datasets with millions of guide pairs through memory-efficient chunked processing and parallel computation. The final output provides:
- **Y25_delta**: Raw genetic interaction scores (Y25_double - Y25_expected)
- **Y25_delta_corrected** (optional): GAM-corrected scores that account for systematic biases

GAM correction helps remove systematic deviations that correlate with expected fitness values, providing more accurate genetic interaction measurements for network analysis.