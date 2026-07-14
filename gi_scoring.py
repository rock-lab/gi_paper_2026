#!/usr/bin/env python3

"""
Genetic interaction scoring pipeline for CRISPRi screen data.

This module provides a streamlined implementation of the genetic interaction
scoring pipeline, designed to handle large-scale datasets (>2M guide pairs)
with memory-efficient chunked processing.

Key steps:
1. Prepare model data from log2FC dataframes (JSON format for Stan)
2. Run Bayesian modeling for guide pairs using Stan
3. Process and merge Stan results
4. Calculate genetic interaction scores and statistics

This implementation preserves the essential functionality while being
standalone and publishable.
"""

import sys
import os
import json
import glob
import subprocess
import tempfile
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# Optional dependencies for full functionality
try:
    from cmdstanpy import CmdStanModel
    HAS_CMDSTAN = True
except ImportError:
    HAS_CMDSTAN = False

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from pygam import GAM, s
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s')
logger.setLevel(logging.DEBUG)


# Module-level functions for multiprocessing
def process_guide_id_for_model(args):
    """
    Process a single guide ID to create model data JSON.
    Module-level function for multiprocessing compatibility.
    """
    guide_id, logfc_df, model_data_dir, force = args

    # Parse the guide ID to get ORF1 and ORF2 for folder structure
    # Format: ORF1:gene1_SEQ1_ORF2:gene2_SEQ2
    parts = guide_id.split('_')
    if len(parts) == 4:
        orf1 = parts[0].split(':')[0] if ':' in parts[0] else parts[0]
        orf2 = parts[2].split(':')[0] if ':' in parts[2] else parts[2]
    else:
        # Fallback for unexpected format
        orf1 = "unknown"
        orf2 = "unknown"

    # Create hierarchical directory structure: ORF1/ORF2/
    output_dir = model_data_dir / orf1 / orf2
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from guide ID
    safe_id = guide_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    json_path = output_dir / f"{safe_id}.json"

    if json_path.exists() and not force:
        return None

    # Get data for this guide pair
    pair_data = logfc_df[logfc_df['ID'] == guide_id].copy()

    if len(pair_data) < 3:  # Need minimum data points
        return None

    # Apply quality filters if available
    if 'good' in pair_data.columns:
        pair_data = pair_data[pair_data['good']]

    if len(pair_data) < 3:
        return None

    # Sort by generation
    pair_data = pair_data.sort_values('generations')

    # Create model data using raw log2fc values - ONLY NUMERIC VALUES FOR STAN
    model_data_dict = {
        'N': len(pair_data),
        'J': 1,  # Single guide pair
        'y': pair_data['log2fc'].tolist(),  # Raw log2fc values as input
        'x': pair_data['generations'].tolist(),
        'guides': [1] * len(pair_data)  # All data from same pair, using integer index
    }

    # Save JSON for Stan (numeric only)
    with open(json_path, 'w') as f:
        json.dump(model_data_dict, f, indent=2)

    # Save metadata TSV alongside JSON with additional information
    metadata_path = str(json_path).replace('.json', '.tsv')
    metadata_df = pair_data[['strain', 'experiment', 'generations', 'ID', 'log2fc']].copy()
    metadata_df['guide_id'] = guide_id
    metadata_df['guide_index'] = 1  # The index used in Stan
    metadata_df.to_csv(metadata_path, sep='\t', index=False)

    return str(json_path)


def run_stan_model_for_pair(args):
    """
    Run Stan model for a single guide pair.
    Module-level function for multiprocessing compatibility.
    """
    model, json_path, chains, iter_sampling = args

    try:
        # Load data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Determine output path with same folder structure
        # Replace model_data with samples in the path
        output_path = str(json_path).replace('/model_data/', '/samples/').replace('.json', '_samples.tsv')

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Run Stan model
        fit = model.sample(
            data=data,
            chains=chains,
            iter_sampling=iter_sampling,
            show_progress=False,
            show_console=False
            # refresh parameter removed - let Stan use default
        )

        # Save results
        draws_df = fit.draws_pd()
        draws_df.to_csv(output_path, sep='\t', index=False)

        return True

    except Exception as e:
        logger.debug(f"Stan model failed for {json_path}: {e}")
        return False


class GIScoring:
    """
    Genetic interaction scoring pipeline for large-scale CRISPRi data.

    Designed for memory-efficient processing of millions of guide pairs.
    """

    def __init__(self, max_guides_per_gene=2, output_dir="./gi_results",
                 stan_model_path=None, workers=None):
        """
        Initialize GI scoring pipeline.

        Args:
            max_guides_per_gene (int): Maximum guides to consider per gene
            output_dir (str): Base directory for all outputs
            stan_model_path (str): Path to Stan model file
            workers (int): Number of parallel workers
        """
        self.max_guides_per_gene = max_guides_per_gene
        self.output_dir = Path(output_dir)
        self.workers = workers if workers else min(cpu_count(), 8)

        # Create output structure
        self.model_data_dir = self.output_dir / "model_data"
        self.samples_dir = self.output_dir / "samples"
        self.results_dir = self.output_dir / "results"

        for dir_path in [self.model_data_dir, self.samples_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Lazy load Stan model - only initialize when needed
        self.stan_model_path = stan_model_path
        self._stan_model = None  # Will be loaded on first use

    @property
    def stan_model(self):
        """Lazy-load Stan model only when needed."""
        if self._stan_model is None and HAS_CMDSTAN:
            logger.info("Initializing Stan model...")
            if self.stan_model_path:
                self._stan_model = CmdStanModel(stan_file=self.stan_model_path)
            else:
                self._stan_model = self._create_default_stan_model()
        return self._stan_model

    def _create_default_stan_model(self):
        """Create default Stan model for genetic interactions."""
        stan_code = """
functions {
  vector get_twoline_mean(vector x, array[] int guides, vector alpha_l,
                         vector beta_l, vector gamma, vector beta_e) {
    int N = num_elements(x);
    vector[N] mu_ii;
    for (i in 1:N) {
      if (x[i] <= gamma[guides[i]]) {
        mu_ii[i] = alpha_l[guides[i]] + (beta_l[guides[i]] * x[i]);
      } else {
        mu_ii[i] = (alpha_l[guides[i]] + beta_l[guides[i]] * gamma[guides[i]]) +
                   (beta_e[guides[i]] * (x[i] - gamma[guides[i]]));
      }
    }
    return mu_ii;
  }
}

data {
  int<lower=0> N; // Number of data points
  int<lower=0> J; // Number of guides
  vector[N] y;    // logfc values
  vector[N] x;    // generations
  array[N] int guides; // guide index for each data point
}

parameters {
  real<lower=0.01, upper=100> nu_y;
  vector<lower=0.01, upper=100>[J] sigma;
  vector<lower=-10, upper=10>[J] alpha_l;
  vector<lower=-10, upper=10>[J] beta_e;
  vector<lower=-2, upper=2>[J] beta_l;
  vector<lower=0.01, upper=20>[J] gamma;
}

model {
  vector[N] mu_ii;

  // Priors
  alpha_l ~ normal(0, 1);
  beta_e ~ normal(-0.2, 0.5);
  beta_l ~ normal(0, 0.2);
  gamma ~ normal(4, 2);
  sigma ~ normal(0.5, 1);
  nu_y ~ normal(3, 1);

  mu_ii = get_twoline_mean(x, guides, alpha_l, beta_l, gamma, beta_e);
  y ~ student_t(nu_y, mu_ii, sigma[guides]);
}

generated quantities {
  // Y25: prediction for 25th generation
  real Y25 = mean((alpha_l + beta_l .* gamma) + (beta_e .* (25.0 - gamma)));
}
        """

        model_file = self.output_dir / "gi_twoline_model.stan"
        model_file.write_text(stan_code)
        return CmdStanModel(stan_file=str(model_file))

    @staticmethod
    def is_negative_control(guide_name: str) -> bool:
        """Check if guide is a negative control."""
        negative_terms = ["Negative", "NT", "Non-targeting", "NonTargeting"]
        return any(term in guide_name for term in negative_terms)

    def get_guide_pairs_from_logfc_data(self, logfc_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Extract all possible guide pairs from log2FC dataframe.

        Args:
            logfc_df: DataFrame with log2FC data including ORF1, ORF2, etc.

        Returns:
            List of (guide1, guide2) tuples
        """
        logger.info("Extracting guide pairs from log2FC data...")

        # Ensure we have the required columns
        required_cols = ['orf1', 'orf2', 'seq1', 'seq2', 'guide_name1', 'guide_name2']
        missing_cols = [col for col in required_cols if col not in logfc_df.columns]

        if missing_cols:
            # Try to create missing columns if possible
            if 'guide_name1' not in logfc_df.columns and all(col in logfc_df.columns for col in ['orf1', 'seq1']):
                logfc_df = self._add_guide_names(logfc_df)
            else:
                raise ValueError(f"Missing required columns: {missing_cols}")

        # Get unique guide pairs, including negative controls
        unique_guides = set()
        guide_pairs = set()

        # Extract all individual guides
        for _, row in logfc_df.iterrows():
            if pd.notna(row['guide_name1']):
                unique_guides.add(row['guide_name1'])
            if pd.notna(row['guide_name2']):
                unique_guides.add(row['guide_name2'])

        # Get existing pairs from data
        for _, row in logfc_df.iterrows():
            if pd.notna(row['guide_name1']) and pd.notna(row['guide_name2']):
                guide1, guide2 = row['guide_name1'], row['guide_name2']
                if not (self.is_negative_control(guide1) and self.is_negative_control(guide2)):
                    guide_pairs.add((guide1, guide2))

        # Add guide + negative control pairs for each non-negative guide
        for guide in unique_guides:
            if not self.is_negative_control(guide):
                guide_pairs.add((guide, "Negative"))
                guide_pairs.add(("Negative", guide))

        logger.info(f"Found {len(guide_pairs)} guide pairs to process")
        return list(guide_pairs)

    def _add_guide_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add guide_name columns if missing."""
        df = df.copy()

        # Create sequence to number mapping for each ORF
        seq_to_number = {}
        reference_df = df[df['generations'] == 0] if 'generations' in df.columns and (df['generations'] == 0).any() else df

        for orf in reference_df['orf1'].unique():
            if pd.isna(orf):
                continue
            orf_seqs = reference_df[reference_df['orf1'] == orf]['seq1'].unique()
            for i, seq in enumerate(sorted(orf_seqs), 1):
                if pd.notna(seq):
                    seq_to_number[seq] = i

        # Apply guide numbering
        df['guide_name1'] = df.apply(
            lambda row: f"{row['orf1']}-{seq_to_number.get(row['seq1'], '1')}"
            if pd.notna(row['orf1']) and pd.notna(row['seq1']) else row['orf1'], axis=1
        )

        df['guide_name2'] = df.apply(
            lambda row: f"{row['orf2']}-{seq_to_number.get(row['seq2'], '1')}"
            if pd.notna(row['orf2']) and pd.notna(row['seq2']) else row['orf2'], axis=1
        )

        return df

    def prepare_model_data_for_pair(self, logfc_df: pd.DataFrame, guide1: str, guide2: str) -> Optional[Dict]:
        """
        Prepare model data for a specific guide pair.

        Args:
            logfc_df: Full log2FC dataframe
            guide1: First guide name
            guide2: Second guide name

        Returns:
            Dictionary with data for Stan model, or None if insufficient data
        """
        # Filter data for this guide pair
        pair_data = logfc_df[
            ((logfc_df['guide_name1'] == guide1) & (logfc_df['guide_name2'] == guide2)) |
            ((logfc_df['guide_name1'] == guide2) & (logfc_df['guide_name2'] == guide1))
        ].copy()

        if len(pair_data) < 3:  # Need minimum data points
            return None

        # Apply quality filters
        if 'good' in pair_data.columns:
            pair_data = pair_data[pair_data['good']].copy()

        if len(pair_data) < 3:
            return None

        # Prepare data for Stan - always use log2fc values
        generations = pair_data['generations'].values

        # The Stan model fits log2fc trajectories over generations using a two-line model
        # It will predict log2fc at generation 25 for this guide pair
        logfc_values = pair_data['log2fc'].values

        # Create guide indices (all data is from same pair, so index=1 for all)
        guide_indices = np.ones(len(pair_data), dtype=int)

        model_data = {
            'N': len(pair_data),
            'J': 1,  # Single guide pair
            'y': logfc_values.tolist(),
            'x': generations.tolist(),
            'guides': guide_indices.tolist()
        }

        # Store metadata for later use
        if 'log2fc_nt1' in pair_data.columns and 'log2fc_nt2' in pair_data.columns:
            model_data['metadata'] = {
                'log2fc_nt1_mean': float(pair_data['log2fc_nt1'].mean()),
                'log2fc_nt2_mean': float(pair_data['log2fc_nt2'].mean()),
                'log2fc_mean': float(pair_data['log2fc'].mean())
            }

        return model_data

    def save_model_data_json(self, guide1: str, guide2: str, model_data: Dict,
                            output_path: str) -> None:
        """Save model data as JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)

    def run_stan_model_on_pair(self, json_path: str, output_path: str,
                              chains: int = 2, iter_sampling: int = 1000) -> bool:
        """
        Run Stan model on a guide pair.

        Args:
            json_path: Path to JSON data file
            output_path: Path to save results
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain

        Returns:
            True if successful, False otherwise
        """
        if not self.stan_model:
            logger.error("Stan model not available")
            return False

        try:
            # Load data
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Run Stan model
            fit = self.stan_model.sample(
                data=data,
                chains=chains,
                iter_sampling=iter_sampling,
                show_progress=False,
                show_console=False
            )

            # Save results
            draws_df = fit.draws_pd()
            draws_df.to_csv(output_path, sep='\t', index=False)

            return True

        except Exception as e:
            logger.warning(f"Stan model failed for {json_path}: {e}")
            return False

    def step1_prepare_model_data(self, logfc_df_path: str,
                               start_idx: int = 0, end_idx: int = -1,
                               force: bool = False) -> None:
        """
        Step 1: Prepare model data JSON files for all guide pairs.
        Creates one JSON file per unique guide pair ID in the dataframe.

        Args:
            logfc_df_path: Path to log2FC dataframe
            start_idx: Starting index for processing (for chunking)
            end_idx: Ending index for processing (-1 for all)
            force: Overwrite existing files
        """
        logger.info("Step 1: Preparing model data...")

        # Load log2FC data
        logger.info(f"Loading log2FC data from {logfc_df_path}")
        logfc_df = pd.read_csv(logfc_df_path, sep='\t', comment='#', low_memory=False)

        # Get all unique guide pair IDs from the dataframe
        # This includes both double and single mutants
        unique_ids = logfc_df['ID'].unique()

        if end_idx == -1:
            end_idx = len(unique_ids)

        unique_ids_chunk = unique_ids[start_idx:end_idx]
        logger.info(f"Processing {len(unique_ids_chunk)} guide pairs (indices {start_idx}:{end_idx})")

        # Prepare arguments for multiprocessing
        print(f"\nPreparing to process {len(unique_ids_chunk)} guide pairs...")
        args_list = [(guide_id, logfc_df, self.model_data_dir, force) for guide_id in unique_ids_chunk]

        # Process in parallel using module-level function
        print(f"Processing in parallel with {self.workers} workers...")
        results = process_map(
            process_guide_id_for_model,
            args_list,
            max_workers=self.workers,
            desc="Creating model JSON files",
            unit="pairs",
            chunksize=max(1, len(args_list) // (self.workers * 10))  # Dynamic chunk size
        )

        successful = [r for r in results if r is not None]
        logger.info(f"Successfully prepared model data for {len(successful)} guide pairs")

    def step2_run_stan_models(self, start_idx: int = 0, end_idx: int = -1,
                            force: bool = False, chains: int = 2,
                            iter_sampling: int = 1000) -> None:
        """
        Step 2: Run Stan models on all prepared JSON files.

        Args:
            start_idx: Starting index for processing
            end_idx: Ending index for processing (-1 for all)
            force: Overwrite existing results
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain
        """
        logger.info("Step 2: Running Stan models...")

        if not self.stan_model:
            logger.error("Stan model not available. Install CmdStanPy and Stan.")
            return

        # Get JSON files from all subdirectories
        json_files = list(self.model_data_dir.rglob("*.json"))
        json_files.sort()

        if end_idx == -1:
            end_idx = len(json_files)

        json_files = json_files[start_idx:end_idx]
        logger.info(f"Processing {len(json_files)} model data files")

        # First, filter to files that need processing
        print("\nChecking which files need processing...")
        files_to_process = []
        for json_path in tqdm(json_files, desc="Checking existing samples", unit="files"):
            # Determine output path with same folder structure
            output_path = str(json_path).replace('/model_data/', '/samples/').replace('.json', '_samples.tsv')
            if not os.path.exists(output_path) or force:
                files_to_process.append(json_path)

        logger.info(f"Found {len(files_to_process)} files to process ({len(json_files) - len(files_to_process)} already done)")

        if len(files_to_process) == 0:
            logger.info("All models already run, nothing to do")
            return

        # Prepare arguments for parallel processing
        # Pass the compiled model as part of the arguments
        args_list = [(self.stan_model, str(json_path), chains, iter_sampling)
                     for json_path in files_to_process]

        # Process in parallel with progress bar
        # You can limit workers if memory is an issue, but use all available by default
        stan_workers = self.workers
        print(f"\nRunning {len(args_list)} Stan models in parallel with {stan_workers} workers...")
        print(f"This may take a while. Each model runs MCMC sampling...")

        results = process_map(
            run_stan_model_for_pair,
            args_list,
            max_workers=stan_workers,
            desc="Fitting Stan models",
            unit="models",
            chunksize=max(1, len(args_list) // (stan_workers * 10))
        )

        successful = sum(results)
        logger.info(f"Successfully ran Stan models on {successful} guide pairs")

    def calculate_y25_delta_scores(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Y25_delta (genetic interaction scores) from Y25 predictions.

        Y25_delta = Y25(double mutant) - [Y25(single mutant 1) + Y25(single mutant 2)]

        This is the true genetic interaction score.

        Args:
            results_df: DataFrame with Y25 predictions for all guide pairs

        Returns:
            DataFrame with Y25_delta scores added
        """
        # Build lookup dictionaries for single mutants, POSITION-MATCHED to the
        # double (matching the paper convention in
        # indiv_tools.get_all_possible_guide_y25_label_combinations): for a double
        # "A(left)_B(right)", gene A's single is A on the LEFT (A_Negative) and
        # gene B's single is B on the RIGHT (Negative_B) — each gene's single is
        # taken in the same construct position it occupies in the double.
        #
        # A targeting guide appears in many single constructs per position
        # (different non-targeting partners), so we ACCUMULATE its Y25 across all
        # of them and average — deterministic and using every replicate, rather
        # than keeping one arbitrary construct.
        double_mutants = []
        single_mutants_accum = {}  # {(gene, seq, side): [y25_values]}; side in {'left','right'}

        print("Identifying double and single mutants...")
        for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Classifying mutants", unit="pairs", leave=False):
            guide_id = row['guide_id']
            parts = guide_id.split('_')

            if len(parts) == 4:  # Format: ORF1:gene1_SEQ1_ORF2:gene2_SEQ2
                orf1_gene1 = parts[0]
                seq1 = parts[1]
                orf2_gene2 = parts[2]
                seq2 = parts[3]

                # Classify, tracking which side the targeting guide sits on.
                if 'Negative' in orf1_gene1:
                    # Negative_X_ORF2:gene2_SEQ2 -> targeting guide is on the RIGHT
                    single_mutants_accum.setdefault((orf2_gene2, seq2, 'right'), []).append(row['y25_mean'])
                elif 'Negative' in orf2_gene2:
                    # ORF1:gene1_SEQ1_Negative_X -> targeting guide is on the LEFT
                    single_mutants_accum.setdefault((orf1_gene1, seq1, 'left'), []).append(row['y25_mean'])
                else:
                    # Double mutant
                    double_mutants.append(row)

        # Reduce each guide's per-position single-mutant Y25 measurements to their mean.
        single_mutants_by_gene = {k: float(np.mean(v)) for k, v in single_mutants_accum.items()}

        logger.info(f"Found {len(double_mutants)} double mutants and {len(single_mutants_by_gene)} unique single mutants")

        # Calculate Y25_delta for each double mutant
        y25_delta_results = []

        print(f"Calculating Y25_delta for {len(double_mutants)} double mutants...")
        for row in tqdm(double_mutants, desc="Computing Y25_delta", unit="pairs", leave=False):
            guide_id = row['guide_id']
            parts = guide_id.split('_')

            # Extract components
            orf1_gene1 = parts[0]
            seq1 = parts[1]
            orf2_gene2 = parts[2]
            seq2 = parts[3]

            # Look up each gene's single mutant in the SAME position it occupies
            # in this double: left gene -> its LEFT single, right gene -> its RIGHT.
            y25_double = row['y25_mean']
            y25_single1 = single_mutants_by_gene.get((orf1_gene1, seq1, 'left'), np.nan)
            y25_single2 = single_mutants_by_gene.get((orf2_gene2, seq2, 'right'), np.nan)

            if not np.isnan(y25_single1) and not np.isnan(y25_single2):
                # Calculate Y25_delta (the genetic interaction score)
                y25_delta = y25_double - (y25_single1 + y25_single2)

                # Calculate expected (sum of single mutants)
                y25_expected = y25_single1 + y25_single2

                y25_delta_results.append({
                    'guide_pair': guide_id,
                    'guide1': f"{orf1_gene1}_{seq1}",
                    'guide2': f"{orf2_gene2}_{seq2}",
                    'y25_double': y25_double,
                    'y25_single1': y25_single1,
                    'y25_single2': y25_single2,
                    'y25_expected': y25_expected,  # Sum of single mutants
                    'y25_delta': y25_delta,  # This is the uncorrected GI score
                    'y25_std': row['y25_std'],
                    'y25_q025': row['y25_q025'],
                    'y25_q975': row['y25_q975']
                })

        logger.info(f"Calculated Y25_delta for {len(y25_delta_results)} double mutant pairs")
        return pd.DataFrame(y25_delta_results)

    def step3_calculate_gi_scores(self, output_path: str = None) -> pd.DataFrame:
        """
        Step 3: Calculate genetic interaction scores from Stan results.

        Args:
            output_path: Path to save results

        Returns:
            DataFrame with GI scores
        """
        logger.info("Step 3: Calculating genetic interaction scores...")

        # Get all sample files from all subdirectories
        sample_files = list(self.samples_dir.rglob("*_samples.tsv"))

        if len(sample_files) == 0:
            logger.error("No sample files found. Run steps 1 and 2 first.")
            return pd.DataFrame()

        logger.info(f"Processing {len(sample_files)} sample files")

        results = []

        print(f"\nExtracting Y25 predictions from {len(sample_files)} Stan samples...")
        for sample_file in tqdm(sample_files, desc="Extracting Y25 predictions", unit="files"):
            try:
                # Load corresponding metadata to get the original guide ID
                # The TSV file should be in the same relative path but in model_data
                metadata_path = str(sample_file).replace('/samples/', '/model_data/').replace('_samples.tsv', '.tsv')

                if os.path.exists(metadata_path):
                    metadata_df = pd.read_csv(metadata_path, sep='\t', nrows=1)
                    guide_id = metadata_df['guide_id'].iloc[0] if 'guide_id' in metadata_df.columns else metadata_df['ID'].iloc[0]
                else:
                    # Try to reconstruct guide ID from filename
                    guide_id = sample_file.stem.replace('_samples', '')

                # Load samples
                samples_df = pd.read_csv(sample_file, sep='\t')

                if 'Y25' in samples_df.columns:
                    # Calculate statistics for Y25 (predicted fitness at generation 25)
                    y25_mean = samples_df['Y25'].mean()
                    y25_std = samples_df['Y25'].std()
                    y25_q025 = samples_df['Y25'].quantile(0.025)
                    y25_q975 = samples_df['Y25'].quantile(0.975)

                    # Y25 is the predicted log2fc at generation 25 for this guide pair
                    # This is NOT the final GI score - that requires comparing with single mutants

                    results.append({
                        'guide_id': guide_id,
                        'y25_mean': y25_mean,   # Predicted log2fc at generation 25
                        'y25_std': y25_std,     # Standard deviation
                        'y25_q025': y25_q025,   # 2.5% quantile (lower CI bound)
                        'y25_q975': y25_q975,   # 97.5% quantile (upper CI bound)
                        'n_samples': len(samples_df),
                        'sample_file': str(sample_file)  # Keep track of source file
                    })

            except Exception as e:
                logger.warning(f"Error processing {sample_file}: {e}")
                continue

        # Create results dataframe
        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            logger.info(f"Extracted Y25 predictions for {len(results_df)} guide pairs")

            # Calculate Y25_delta (GI scores) by comparing double and single mutants
            print(f"\nCalculating Y25_delta scores for {len(results_df)} guide pairs...")
            gi_scores_df = self.calculate_y25_delta_scores(results_df)

            if output_path and len(gi_scores_df) > 0:
                gi_scores_df.to_csv(output_path, sep='\t', index=False)
                logger.info(f"Saved GI scores to {output_path}")

                # Print summary statistics
                print(f"\nGenetic Interaction Score Summary:")
                print(f"Total guide pairs with GI scores: {len(gi_scores_df)}")
                print(f"Mean Y25_delta (GI score): {gi_scores_df['y25_delta'].mean():.3f}")
                print(f"Std Y25_delta: {gi_scores_df['y25_delta'].std():.3f}")
                print(f"Min Y25_delta: {gi_scores_df['y25_delta'].min():.3f}")
                print(f"Max Y25_delta: {gi_scores_df['y25_delta'].max():.3f}")

                return gi_scores_df
        else:
            logger.error("No results generated")

        return results_df

    def apply_gam_correction_python(self, gi_scores_df: pd.DataFrame, n_splines: int = 20) -> pd.DataFrame:
        """
        Apply GAM correction to Y25_delta scores using Python (pygam).

        The GAM models the relationship between expected fitness (sum of singles)
        and the deviation from expectation (Y25_delta).

        Args:
            gi_scores_df: DataFrame with Y25_delta scores
            n_splines: Number of splines for GAM (default: 20)

        Returns:
            DataFrame with corrected Y25_delta scores
        """
        if not HAS_PYGAM:
            logger.warning("pygam not available. Install with: pip install pygam")
            logger.info("Returning uncorrected scores")
            return gi_scores_df

        logger.info("Applying GAM correction using pygam...")

        df = gi_scores_df.copy()

        # Remove any rows with NaN values
        df_clean = df.dropna(subset=['y25_expected', 'y25_delta'])

        if len(df_clean) < 10:
            logger.warning("Not enough data points for GAM correction")
            return gi_scores_df

        try:
            # Fit GAM: y25_delta ~ s(y25_expected)
            X = df_clean[['y25_expected']].values
            y = df_clean['y25_delta'].values

            # Create and fit GAM model
            gam = GAM(s(0, n_splines=n_splines))
            gam.gridsearch(X, y, progress=False)

            # Get predictions
            y_pred = gam.predict(X)

            # Calculate corrected Y25_delta (residuals)
            y25_delta_corrected = y - y_pred

            # Add corrected values back to dataframe
            df.loc[df_clean.index, 'y25_delta_corrected'] = y25_delta_corrected
            df.loc[df_clean.index, 'gam_prediction'] = y_pred

            # Calculate GAM statistics
            gam_r2 = gam.statistics_['pseudo_r2']['explained_deviance']
            logger.info(f"GAM correction applied. Pseudo R²: {gam_r2:.4f}")

        except Exception as e:
            logger.error(f"GAM correction failed: {e}")
            logger.info("Returning uncorrected scores")
            return gi_scores_df

        return df

    def apply_gam_correction_r(self, gi_scores_df: pd.DataFrame,
                              r_script_path: str = None,
                              sp: float = None, k: int = 20) -> pd.DataFrame:
        """
        Apply GAM correction using R script (requires R and mgcv package).

        Args:
            gi_scores_df: DataFrame with Y25_delta scores
            r_script_path: Path to gam_correction_standalone.R script
            sp: Smoothing parameter (None for automatic)
            k: Basis dimension (default: 20)

        Returns:
            DataFrame with corrected Y25_delta scores
        """
        import tempfile
        import subprocess

        logger.info("Applying GAM correction using R/mgcv...")

        # If no R script path provided, check for it in the current directory
        if r_script_path is None:
            r_script_path = "gam_correction_standalone.R"

        if not os.path.exists(r_script_path):
            logger.warning(f"R script not found at {r_script_path}")
            logger.info("Returning uncorrected scores")
            return gi_scores_df

        # Prepare data for R
        df_for_r = gi_scores_df[['guide_pair', 'y25_expected', 'y25_double']].copy()
        df_for_r.columns = ['guide_pair', 'expected', 'y25']  # R script expects these column names

        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as input_file:
            df_for_r.to_csv(input_file.name, sep='\t', index=False)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as output_file:
            output_path = output_file.name

        try:
            # Prepare command
            cmd = ["Rscript", r_script_path, input_path, output_path]
            if sp is not None:
                cmd.append(str(sp))
            else:
                cmd.append("NULL")
            cmd.append(str(k))
            cmd.append("ML")

            # Run R script
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"R script failed: {result.stderr}")
                return gi_scores_df

            # Read corrected data
            corrected_df = pd.read_csv(output_path, sep='\t')

            # Merge corrected values back
            gi_scores_df = gi_scores_df.merge(
                corrected_df[['guide_pair', 'y_corrected']],
                on='guide_pair',
                how='left'
            )
            gi_scores_df.rename(columns={'y_corrected': 'y25_delta_corrected'}, inplace=True)

            logger.info("GAM correction applied successfully using R")

        except Exception as e:
            logger.error(f"GAM correction failed: {e}")
            return gi_scores_df

        finally:
            # Clean up temporary files
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)

        return gi_scores_df

    def step4_apply_gam_correction(self, gi_scores_path: str, output_path: str = None,
                                  method: str = 'python', **kwargs) -> pd.DataFrame:
        """
        Step 4: Apply GAM correction to genetic interaction scores.

        Args:
            gi_scores_path: Path to GI scores from step 3
            output_path: Path to save corrected scores
            method: 'python' (uses pygam) or 'r' (uses R script)
            **kwargs: Additional arguments for GAM correction

        Returns:
            DataFrame with corrected GI scores
        """
        logger.info("Step 4: Applying GAM correction...")

        # Load GI scores
        gi_scores_df = pd.read_csv(gi_scores_path, sep='\t')

        # Apply GAM correction
        if method == 'python':
            corrected_df = self.apply_gam_correction_python(
                gi_scores_df,
                n_splines=kwargs.get('n_splines', 20)
            )
        elif method == 'r':
            corrected_df = self.apply_gam_correction_r(
                gi_scores_df,
                r_script_path=kwargs.get('r_script_path'),
                sp=kwargs.get('sp'),
                k=kwargs.get('k', 20)
            )
        else:
            logger.error(f"Unknown GAM correction method: {method}")
            return gi_scores_df

        # Save corrected scores
        if output_path:
            corrected_df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Corrected GI scores saved to {output_path}")

        # Print summary
        if 'y25_delta_corrected' in corrected_df.columns:
            print(f"\nCorrected GI Score Summary:")
            print(f"Mean Y25_delta (corrected): {corrected_df['y25_delta_corrected'].mean():.3f}")
            print(f"Std Y25_delta (corrected): {corrected_df['y25_delta_corrected'].std():.3f}")
            print(f"Min Y25_delta (corrected): {corrected_df['y25_delta_corrected'].min():.3f}")
            print(f"Max Y25_delta (corrected): {corrected_df['y25_delta_corrected'].max():.3f}")

        return corrected_df


def main():
    """Command line interface for GI scoring pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Genetic interaction scoring pipeline")
    parser.add_argument("--logfc_data", help="Path to log2FC dataframe (required for step 1)")
    parser.add_argument("--output_dir", default="./gi_results", help="Output directory")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run specific step only")
    parser.add_argument("--start", type=int, default=0, help="Start index for chunked processing")
    parser.add_argument("--end", type=int, default=-1, help="End index for chunked processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--stan_model", help="Path to custom Stan model file")
    parser.add_argument("--gam_method", choices=['python', 'r'], default='python',
                       help="GAM correction method: 'python' (pygam) or 'r' (R/mgcv)")
    parser.add_argument("--r_script", help="Path to GAM correction R script (default: gam_correction_standalone.R)")

    args = parser.parse_args()

    # Validate that logfc_data is provided when needed
    if args.step in [None, 1] and not args.logfc_data:
        parser.error("--logfc_data is required for step 1 or when running all steps")

    # Initialize GI scoring pipeline
    gi_scorer = GIScoring(
        output_dir=args.output_dir,
        stan_model_path=args.stan_model,
        workers=args.workers
    )

    # Run requested steps
    if args.step is None or args.step == 1:
        gi_scorer.step1_prepare_model_data(
            args.logfc_data,
            start_idx=args.start,
            end_idx=args.end,
            force=args.force
        )

    if args.step is None or args.step == 2:
        gi_scorer.step2_run_stan_models(
            start_idx=args.start,
            end_idx=args.end,
            force=args.force
        )

    if args.step is None or args.step == 3:
        output_path = os.path.join(args.output_dir, "gi_scores.tsv")
        gi_scorer.step3_calculate_gi_scores(output_path)

    if args.step == 4:
        # GAM correction step (optional)
        gi_scores_path = os.path.join(args.output_dir, "gi_scores.tsv")
        corrected_path = os.path.join(args.output_dir, "gi_scores_corrected.tsv")

        # Use specified GAM correction method
        gi_scorer.step4_apply_gam_correction(
            gi_scores_path,
            output_path=corrected_path,
            method=args.gam_method,
            r_script_path=args.r_script
        )


if __name__ == "__main__":
    main()