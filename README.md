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

## Dependencies

The scripts require the following Python packages:
- `pysam` - for BAM file processing
- `numpy` - for numerical operations
- `scipy` - for statistical functions
- `pandas` - for data manipulation
- `matplotlib` - for plotting
- `seaborn` - for statistical plotting

External dependencies:
- `subread` alignment package

## Usage Example

To process FASTQ files and generate sgRNA counts:

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

## Output Files

The pipeline generates:
- Individual `.bam` and `.sorted.bam` files for each sample
- `.counts` files containing sgRNA read counts
- `.diagnostics` files with alignment statistics
- `merged_*_counts.txt` - merged count matrix across all samples
- `merged_*_diagnostics.txt` - merged diagnostic statistics

## Data Structure

The repository is organized for genetic interaction screen analysis, where:
- sgRNA libraries target genes of interest in M. tuberculosis and M. smegmatis
- FASTQ files contain amplified sgRNA sequences from different experimental conditions
- Count matrices enable downstream analysis of gene fitness and interactions

This pipeline provides the foundation for analyzing genetic interaction data from CRISPRi screens in mycobacterial species.