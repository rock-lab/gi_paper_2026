[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22694611.svg)](https://doi.org/10.5281/zenodo.22694611)


# Genetic Interaction Screens in Mycobacterium tuberculosis and M. smegmatis

Repository containing the code and methods for genetic interaction screens analysis in M. tuberculosis and M. smegmatis.

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
- Prepares model data in JSON format for Stan (chunked to bound peak memory; one JSON per guide pair)
- Runs Bayesian two-line fitness models for each guide pair independently
- Calculates genetic interaction scores and confidence intervals from model posterior samples
- Designed for large-scale datasets (>2M guide pairs) with parallel processing capabilities

#### run_per_screen_mixture.py

Fits the per-screen 1D Normal-Uniform interaction mixture (`normal_uniform_mix.stan`, the model the individual screens were called with; a measurement-error variant is available via `--stan`) to a `result_summary_long_df` TSV, turning each per-screen GI score into a probability of interaction. See Step 5.

#### run_joint_model.py

Fits the joint cross-screen quadrant mixture (the primary `joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan` model) to two screens, classifying every gene pair as aggravating / alleviating / discordant / no-interaction. See Step 6.

#### merge_joint_results.py

Merges the per-screen and joint outputs into a single 20-column per-pair results table.

#### make_signed_prob_matrix.py / make_hit_matrix.py

Project the per-pair probabilities onto gene x gene matrices: a signed probability-of-interaction matrix (diverging, centered at 0) and a boolean hit matrix.

#### run_example.sh / make_example_data.py / check_example.py

`run_example.sh` is the **runbook**: a shell script that calls each discrete step script in order — for every screen it runs counts → single-screen results, then the joint analysis. Copy it and edit the CONFIG block to run your own data. `make_example_data.py` cuts the 15-gene example dataset from the source data (maintainer-only); `check_example.py` validates a finished example run against the golden tables. See the [Usage Examples](#usage-examples) below to run the shipped example data.

## Dependencies

The **canonical full-pipeline environment** is [`environment.yml`](environment.yml)
(conda) — it installs everything the runbook needs, including `pygam` for the GAM
correction and a prebuilt CmdStan toolchain:

```bash
conda env create -f environment.yml
conda activate gi_paper_2026
```

[`requirements.txt`](requirements.txt) is a **minimal pip install** (core packages
only, as minimum-version bounds, not exact pins). It does **not** include `pygam`,
so the GAM-correction step needs it installed separately (or R + `mgcv`):

```bash
pip install -r requirements.txt
pip install pygam            # GAM-correction step (or use R + mgcv instead)
# then, once, build the CmdStan toolchain that cmdstanpy uses for sampling:
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Core packages: `cmdstanpy` (Stan interface), `numpy`, `scipy`, `pandas`, `tqdm`.

Optional / step-specific:

- `subread` - only for the optional FASTQ -> counts Step 1 (install via your OS package manager, e.g. `sudo apt install subread`).
- `pysam` - BAM file handling in Step 1.
- `pygam` - Python GAM correction in Step 3 (alternative: R + `mgcv`); included in `environment.yml`, **not** in `requirements.txt`.
- `pyarrow` - optional, faster IO for very large guide-pair datasets.
- `matplotlib`, `seaborn` - only for the demo notebooks / plotting.

External:

- `Stan` / CmdStan - probabilistic programming backend, compiled by `cmdstanpy`; needed for **every** Stan sampling step (GI scoring, the per-screen mixture, and the joint model).
- `R` with `mgcv` - optional alternative to `pygam` for the GAM correction.

## Usage Examples

The commands below run the shipped **example dataset** (`example_data/`) end to
end. It has two screens, `exp1` (100 ng/mL ATc) and `exp2` (500 ng/mL ATc),
and begins from pooled count tables. **Step 1 (FASTQ → counts) does not apply
to the example**, and sample reads are not included due to size constraints. The code is included as a reference but may need to be adapted to your sequencing and experimental setup.

Run everything from the repo root; outputs go under `example_out/`.

`bash run_example.sh` runs the whole thing in one command; the steps below are
the same pipeline broken out so you can run or adapt each piece. Steps 2–5 are
**per screen** — run them once for `exp1` and once for `exp2` — then Step 6
combines the two screens.

### Step 1: Process FASTQ files and generate sgRNA counts

> **Not part of the example.** The example dataset begins from pooled count
> tables (Step 2 onward), so this FASTQ → counts stage is **not** run by the
> example. It is the lab's read-processing code for starting from your own raw
> reads, and requires the external `subread` aligner plus `pysam` (BAM handling).
> The `subread` command wrappers are unit-tested for argument construction and
> fail loudly on a nonzero exit code, but the stage is not exercised end to end
> by the example — validate its output on your own data.

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

The example ships a metadata file per screen
(`example_data/experiment_metadata_exp{1,2}.csv`) that already points at the
shipped count tables. Calculate log2FC for a screen (shown for `exp1`):

```bash
python logfc_tools.py \
    --metadata example_data/experiment_metadata_exp1.csv \
    --output example_out/exp1/logfc.txt \
    --normalize
```

For your own data, generate a metadata template first with
`python logfc_tools.py --template --metadata my_metadata.csv` — it has the
required columns `strain`, `experiment`, `condition`, `atc` (`plus`/`minus`),
`generations`, `replicate`, and `count_file_path`.

Key parameters:

- `--metadata`: Path to experiment metadata CSV file
- `--output`: Path to output log2FC dataframe
- `--normalize`: Apply negative control normalization
- `--lod_limit`: Limit of detection for filtering low counts (default: 20.0)
- `--summary_metric`: How to summarize replicates (mean, median, etc.)
- `--pseudo`: Pseudocount for log2FC calculation (default: 1.0)
- `--allow-incomplete`: Skip (with a warning) any timepoint missing its +ATC/-ATC condition instead of failing. Default: an incomplete experiment (missing condition/file, or duplicate metadata rows) is a hard error.

### Step 3: Calculate genetic interaction scores

Fit the per-guide-pair two-line models and compute GI scores (shown for `exp1`):

```bash
python gi_scoring.py \
    --logfc_data example_out/exp1/logfc.txt \
    --output_dir example_out/exp1/gi \
    --workers 4

# optional GAM correction (needs pygam or R + mgcv); writes gi_scores_corrected.tsv
python gi_scoring.py --output_dir example_out/exp1/gi --step 4
```

The example is small, so a single command is fine. For large real datasets
(>2M guide pairs) — **not needed for the example** — process in chunks:

```bash
# All chunked commands MUST share the same --output_dir so that later steps
# read the model_data/ and samples/ the earlier chunks wrote there.
#
# Step 1: Prepare model data in chunks
python gi_scoring.py --logfc_data logfc_results.txt --output_dir ./gi_analysis --step 1 --start 0 --end 500000 --workers 8
python gi_scoring.py --logfc_data logfc_results.txt --output_dir ./gi_analysis --step 1 --start 500000 --end 1000000 --workers 8
# ... continue for all chunks

# Step 2: Run Bayesian models (memory-intensive, use fewer workers)
python gi_scoring.py --output_dir ./gi_analysis --step 2 --start 0 --end 500000 --workers 4
python gi_scoring.py --output_dir ./gi_analysis --step 2 --start 500000 --end 1000000 --workers 4

# Step 3: Calculate Y25_delta (uncorrected GI scores)
python gi_scoring.py --output_dir ./gi_analysis --step 3

# gi_scoring internal step 4 (Optional): Apply GAM correction to GI scores
python gi_scoring.py --output_dir ./gi_analysis --step 4
```

The GAM correction (gi_scoring's internal `--step 4`) adjusts for systematic biases in the genetic interaction scores based on the expected fitness values. This step can use either Python (pygam) or R (mgcv) for the correction.

Key parameters:

- `--logfc_data`: Path to log2FC dataframe from Step 2
- `--output_dir`: Directory for all GI analysis outputs
- `--step`: Run specific step only (1: data prep, 2: modeling, 3: Y25_delta calculation, 4: GAM correction)
- `--start/--end`: Indices for chunked processing of large datasets
- `--workers`: Number of parallel workers (use fewer for Stan models)
- `--force`: Rebuild from a clean output tree (clears `model_data/`, `samples/`, `results/` before step 1)

> **Output-directory safety.** Step 1 writes a `run_manifest.json` fingerprinting
> the log2FC input. Re-running step 1 into an `--output_dir` whose manifest does
> not match the current input (a changed input, or a pre-existing directory with
> no manifest) **fails with an error** rather than silently reusing stale
> `model_data`/`samples`. Pass `--force` to rebuild cleanly, or point
> `--output_dir` at a fresh directory. (Chunked runs are fine: successive step-1
> chunks share the same input fingerprint, so no `--force` is needed between them.)

### Step 4: Aggregate guide-pairs to gene-pairs

Step 3 emits one row per *guide pair*; the per-screen mixture and the joint model
work at the *gene pair* level. Collapse guide-pairs to gene-pairs (median GI
score + a pooled SE) into a `result_summary_long_df` table:

```bash
python aggregate_guide_pairs.py \
    example_out/exp1/gi/gi_scores_corrected.tsv \
    example_out/exp1/result_summary_exp1.tsv
```

(If you ran the optional GAM step, pass `gi_scores_corrected.tsv` as shown;
otherwise pass `example_out/exp1/gi/gi_scores.tsv`.) This public aggregation is a
**simplified stand-in** for the paper's HPC procedure, so the example's
downstream probabilities are meant for *running* the pipeline, not for matching
the paper's published numbers.

### Step 5: Per-screen interaction probability

Steps 1-4 produce, for each gene pair, a per-screen GI score
(`delta_prime_median`) and its standard error (`sd_delta_prime_median`). To turn
each score into a **probability of interaction**, fit a 1D Normal-Uniform
mixture (`normal_uniform_mix.stan`, the model the individual screens were called
with): the bulk of non-interacting pairs is a Normal centered near zero, and
genuine interactions form the diffuse Uniform component.

```bash
python run_per_screen_mixture.py \
    example_out/exp1/result_summary_exp1.tsv \
    example_out/exp1/single_screen_exp1.tsv \
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
that the pair is an interaction). **This per-screen probability is not an input
to the joint Stan fit** — the joint model is fit on the raw per-screen GI scores
(`delta_prime_median`) with their SEs. `prob_interaction_median` is carried
through the merge step only, appearing in the final table as
`prob_interaction_median_exp1` / `prob_interaction_median_exp2` for reference
alongside the joint model's own class probabilities.

> A measurement-error variant, `univariate_normal_uniform_mix_me.stan` — the 1-D
> analog of the joint model, which additionally uses each pair's SE to inflate
> the null and smear the interaction component — is available via
> `--stan univariate_normal_uniform_mix_me.stan`.

**Now repeat Steps 2–5 for `exp2`** (swap `exp1` → `exp2` throughout); both
screens' `single_screen_exp{1,2}.tsv` feed Step 6. Shortcut: the example also
ships the pre-derived per-screen tables
`example_data/gi_input/result_summary_long_df_exp{1,2}_toy.tsv`, so you can skip
Steps 2–5 and run Step 6 directly from those.

### Step 6: Joint cross-screen quadrant model

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
    example_out/exp1/single_screen_exp1.tsv \
    example_out/exp2/single_screen_exp2.tsv \
    example_out/joint \
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
    --exp1-tsv example_out/exp1/single_screen_exp1.tsv \
    --exp2-tsv example_out/exp2/single_screen_exp2.tsv \
    --summary-tsv example_out/joint_quad_me_trunc_halfsmeared_w010_summary.tsv \
    --output example_out/merged.tsv
```

The merged TSV (`example_out/merged.tsv`) has 20 columns:

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
python make_signed_prob_matrix.py \
    --merged-tsv example_out/merged.tsv \
    --value-mode signed_maxmag \
    --output example_out/signed_prob_matrix.tsv

# boolean hit matrix; a pair is a hit if ANY directional class prob >= threshold:
python make_hit_matrix.py \
    --merged-tsv example_out/merged.tsv \
    --threshold 0.5 \
    --output example_out/hit_matrix_thr050.tsv
```

> Both matrix scripts default `--merged-tsv` to the shipped golden example table,
> so **always pass `--merged-tsv`** (and `--output`) to act on the file you just
> produced rather than the bundled example.

Finally, confirm the run against the known Set A interactions and the expected
output structure:

```bash
python check_example.py \
    --single1 example_out/exp1/single_screen_exp1.tsv \
    --single2 example_out/exp2/single_screen_exp2.tsv \
    --merged example_out/merged.tsv \
    --signed-matrix example_out/signed_prob_matrix.tsv \
    --hit-matrix example_out/hit_matrix_thr050.tsv
```

We call a pair an interaction at `prob >= 0.5`; `0.95` is a stricter "confident"
cutoff used for high-precision hit lists.

The derivation of the per-screen `delta_prime` GI score and the optional GAM
correction are documented in [`docs/MODEL.md`](docs/MODEL.md); a worked
walk-through of the joint model lives in the `joint_model_demo` notebook under
[`notebooks/`](notebooks/).

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
  - `generations`: Generation/passage number
  - `ID`: Full sgRNA ID
  - `orf`: Gene/ORF identifier
  - `seq`: sgRNA sequence identifier
  - `log2fc`: Log2 fold-change (+ATC/-ATC)
  - `exp_mean`: Mean count in experimental (+ATC) condition
  - `ctrl_mean`: Mean count in control (-ATC) condition
  - `good`: Quality flag based on limit of detection
  - For dual-guide (paired) constructs the two sides are also split out:
    `orf1`, `orf2`, `seq1`, `seq2`, `guide_name`, `guide_name1`, `guide_name2`

### Step 3 Output (Genetic Interaction Scoring)

Paths below are relative to `--output_dir`.

- `model_data/`: JSON files for each guide pair (Stan model input)
- `samples/`: Stan model posterior samples for each guide pair
- `gi_scores.tsv` (written at the `--output_dir` root): Genetic interaction scores with columns:
  - `guide_pair`: Combined guide pair identifier
  - `guide1`, `guide2`: Individual guide names
  - `y25_double`: Y25 prediction for double mutant
  - `y25_single1`, `y25_single2`: Y25 predictions for single mutants
  - `y25_expected`: Sum of single mutant Y25s
  - `y25_delta`: Uncorrected genetic interaction score (Y25_double - Y25_expected)
  - `y25_std`: Standard deviation of Y25 prediction
  - `y25_q025`, `y25_q975`: 95% confidence interval bounds
- `gi_scores_corrected.tsv` (at the `--output_dir` root, after Step 4): GAM-corrected GI scores with additional columns:
  - `y25_delta_corrected`: GAM-corrected genetic interaction score
  - `gam_prediction`: GAM model prediction (if using Python method)
