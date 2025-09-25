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
- Prepares model data in JSON format for Stan (memory-efficient chunked processing)
- Runs Bayesian two-line fitness models for each guide pair independently
- Calculates genetic interaction scores and confidence intervals from model posterior samples
- Designed for large-scale datasets (>2M guide pairs) with parallel processing capabilities

## Dependencies

The scripts require the following Python packages:
- `pysam` - for BAM file processing
- `numpy` - for numerical operations
- `scipy` - for statistical functions
- `pandas` - for data manipulation
- `matplotlib` - for plotting
- `seaborn` - for statistical plotting
- `tqdm` - for progress bars

Additional dependencies for genetic interaction scoring:
- `cmdstanpy` - Python interface for Stan (Bayesian modeling)
- `pyarrow` - for efficient data handling (optional, recommended for large datasets)

External dependencies:
- `subread` alignment package
- `Stan` - probabilistic programming language for Bayesian modeling

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

# Step 3: Calculate final GI scores
python gi_scoring.py --step 3 --output_dir ./gi_analysis
```

Key parameters:
- `--logfc_data`: Path to log2FC dataframe from Step 2
- `--output_dir`: Directory for all GI analysis outputs
- `--step`: Run specific step only (1: data prep, 2: modeling, 3: scoring)
- `--start/--end`: Indices for chunked processing of large datasets
- `--workers`: Number of parallel workers (use fewer for Stan models)
- `--force`: Overwrite existing files

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
- `results/gi_scores.tsv`: Final genetic interaction scores with columns:
  - `guide1`, `guide2`: Individual guide names
  - `guide_pair`: Combined guide pair identifier
  - `y25_mean`: Mean fitness prediction at generation 25
  - `y25_std`: Standard deviation of fitness prediction
  - `y25_q025`, `y25_q975`: 95% confidence interval bounds
  - `gi_score`: Genetic interaction score
  - `n_samples`: Number of posterior samples

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

The pipeline can handle datasets with millions of guide pairs through memory-efficient chunked processing and parallel computation. The final output provides genetic interaction scores with statistical confidence measures for comprehensive interaction network analysis.