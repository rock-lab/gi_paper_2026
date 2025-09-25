#!/usr/bin/env python3

"""
Bare-bones tools for calculating log2 fold-changes from sgRNA count data
in genetic interaction experiments.

This module provides minimal functionality to:
1. Calculate log2FC between +ATC and -ATC conditions
2. Process multiple time points from passaging experiments
3. Handle basic quality control and normalization
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s')
logger.setLevel(logging.DEBUG)


def get_summary_stat_function(name="mean"):
    """
    Get summary statistic function by name.

    Args:
        name (str): Name of summary statistic ('mean', 'median', 'max', 'min')

    Returns:
        function: Function that computes summary statistic along axis 1
    """
    if name.lower() in ["mean", "average"]:
        return lambda data: np.mean(data, axis=1)
    elif name.lower() in ["median"]:
        return lambda data: np.median(data, axis=1)
    elif name.lower() in ["max"]:
        return lambda data: np.max(data, axis=1)
    elif name.lower() in ["min"]:
        return lambda data: np.min(data, axis=1)
    else:
        logger.warning(f"Could not find summary function '{name}'. Defaulting to mean.")
        return lambda data: np.mean(data, axis=1)


def log2fc(A, B, pseudo=1.0, summary_metric="mean"):
    """
    Calculate log2 fold change between two conditions.

    Args:
        A (array): Control condition counts (rows=sgRNAs, cols=replicates)
        B (array): Experimental condition counts (rows=sgRNAs, cols=replicates)
        pseudo (float): Pseudocount to add before log transformation
        summary_metric (str): How to summarize across replicates

    Returns:
        array: log2(B/A) fold changes for each sgRNA
    """
    if isinstance(summary_metric, str):
        F = get_summary_stat_function(summary_metric)
    else:
        F = summary_metric

    top = F(B) + pseudo
    bottom = F(A) + pseudo
    fc = top / bottom
    logfc = np.log2(fc.astype('float64'))
    return logfc


def load_count_file(filepath):
    """
    Load sgRNA count file.

    Args:
        filepath (str): Path to count file (tab-separated)

    Returns:
        pd.DataFrame: DataFrame with sgRNA counts
    """
    try:
        df = pd.read_csv(filepath, sep='\t', comment='#', low_memory=False)
        return df
    except Exception as e:
        logger.error(f"Error loading count file {filepath}: {e}")
        raise


def get_count_matrix_from_dataframe(df, id_col="Id"):
    """
    Extract count matrix from dataframe.

    Args:
        df (pd.DataFrame): Count dataframe
        id_col (str): Column name containing sgRNA IDs

    Returns:
        tuple: (sgRNA_ids, count_matrix) where count_matrix has shape (n_sgrnas, n_samples)
    """
    sgrna_ids = df[id_col].values
    count_cols = [col for col in df.columns if col != id_col]
    count_matrix = df[count_cols].values.astype(float)
    return sgrna_ids, count_matrix


def create_experiment_metadata_template():
    """
    Create a template for experiment metadata CSV file.

    Returns:
        pd.DataFrame: Template dataframe showing required columns
    """
    template = pd.DataFrame({
        'strain': ['H37Rv', 'H37Rv', 'H37Rv', 'H37Rv'],
        'experiment': ['gi_exp1', 'gi_exp1', 'gi_exp1', 'gi_exp1'],
        'condition': ['sample1_G0', 'sample1_G10', 'sample1_G20', 'sample1_G30'],
        'atc': ['minus', 'minus', 'plus', 'plus'],
        'generations': [0, 10, 20, 30],
        'replicate': [1, 1, 1, 1],
        'count_file_path': [
            'counts/sample1_G0_minus_ATC.counts',
            'counts/sample1_G10_minus_ATC.counts',
            'counts/sample1_G20_plus_ATC.counts',
            'counts/sample1_G30_plus_ATC.counts'
        ]
    })
    return template


def get_logfc_dataframe_from_metadata(metadata_path, output_path=None, summary_metric="mean",
                                     pseudo=1.0, lod_limit=20.0):
    """
    Calculate log2FC from experiment metadata file.

    Args:
        metadata_path (str): Path to CSV file with experiment metadata
        output_path (str): Path to save output dataframe
        summary_metric (str): How to summarize replicates
        pseudo (float): Pseudocount for log2FC calculation
        lod_limit (float): Limit of detection for filtering low counts

    Returns:
        pd.DataFrame: Long-format dataframe with log2FC values
    """
    logger.info(f"Loading experiment metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path, comment='#')

    # Validate required columns
    required_cols = ['strain', 'experiment', 'condition', 'atc', 'generations', 'count_file_path']
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")

    results = []

    # Group by experiment and generations to pair +ATC/-ATC conditions
    for (strain, experiment), exp_group in metadata.groupby(['strain', 'experiment']):
        logger.info(f"Processing {strain} {experiment}")

        for generations, gen_group in exp_group.groupby('generations'):
            # Get +ATC and -ATC conditions for this generation
            plus_atc = gen_group[gen_group['atc'] == 'plus']
            minus_atc = gen_group[gen_group['atc'] == 'minus']

            if len(plus_atc) == 0 or len(minus_atc) == 0:
                logger.warning(f"Missing +ATC or -ATC condition for {strain} {experiment} G{generations}")
                continue

            # Load count files
            plus_count_files = plus_atc['count_file_path'].tolist()
            minus_count_files = minus_atc['count_file_path'].tolist()

            # Load and merge count data
            plus_data = []
            minus_data = []
            sgrna_ids = None

            for count_file in plus_count_files:
                if not os.path.exists(count_file):
                    logger.error(f"Count file not found: {count_file}")
                    continue
                df = load_count_file(count_file)
                ids, counts = get_count_matrix_from_dataframe(df)
                if sgrna_ids is None:
                    sgrna_ids = ids
                plus_data.append(counts.flatten() if counts.ndim == 2 and counts.shape[1] == 1 else counts)

            for count_file in minus_count_files:
                if not os.path.exists(count_file):
                    logger.error(f"Count file not found: {count_file}")
                    continue
                df = load_count_file(count_file)
                ids, counts = get_count_matrix_from_dataframe(df)
                minus_data.append(counts.flatten() if counts.ndim == 2 and counts.shape[1] == 1 else counts)

            if not plus_data or not minus_data:
                logger.warning(f"No valid count data for {strain} {experiment} G{generations}")
                continue

            # Convert to arrays
            plus_matrix = np.column_stack(plus_data) if len(plus_data) > 1 else plus_data[0].reshape(-1, 1)
            minus_matrix = np.column_stack(minus_data) if len(minus_data) > 1 else minus_data[0].reshape(-1, 1)

            # Apply LOD filtering to experimental condition
            plus_summary = get_summary_stat_function(summary_metric)(plus_matrix)
            good_detection = plus_summary >= lod_limit

            # Calculate log2FC
            logfc_values = log2fc(minus_matrix, plus_matrix, pseudo=pseudo, summary_metric=summary_metric)

            # Create results for this generation
            for i, sgrna_id in enumerate(sgrna_ids):
                # Parse sgRNA information from ID
                if '_' in sgrna_id:
                    parts = sgrna_id.split('_')
                    orf = parts[0] if len(parts) > 0 else sgrna_id
                    seq = parts[-1] if len(parts) > 1 else ""
                else:
                    orf = sgrna_id
                    seq = ""

                result_row = {
                    'strain': strain,
                    'experiment': experiment,
                    'G': generations,
                    'ORF': orf,
                    'SEQ': seq,
                    'ID': sgrna_id,
                    'Y': logfc_values[i],
                    'exp_mean': plus_summary[i],
                    'ctrl_mean': get_summary_stat_function(summary_metric)(minus_matrix)[i],
                    'GOOD': good_detection[i]
                }
                results.append(result_row)

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    if output_path:
        logger.info(f"Saving results to {output_path}")
        result_df.to_csv(output_path, sep='\t', index=False)

    return result_df


def normalize_negative_controls(df, negative_orf_names=["Negative", "NT", "NonTargeting"]):
    """
    Normalize log2FC values using negative control sgRNAs.

    Args:
        df (pd.DataFrame): DataFrame with log2FC values
        negative_orf_names (list): ORF names that represent negative controls

    Returns:
        pd.DataFrame: DataFrame with normalized Y values
    """
    logger.info("Normalizing using negative controls")
    df_norm = df.copy()

    for (strain, experiment, generation), group in df.groupby(['strain', 'experiment', 'G']):
        # Find negative controls
        is_negative = group['ORF'].isin(negative_orf_names)

        if not is_negative.any():
            logger.warning(f"No negative controls found for {strain} {experiment} G{generation}")
            continue

        # Calculate negative control median
        negative_median = np.median(group.loc[is_negative, 'Y'])

        # Subtract from all values in this group
        group_indices = group.index
        df_norm.loc[group_indices, 'Y'] = group['Y'] - negative_median

        logger.debug(f"Applied normalization: {strain} {experiment} G{generation}, offset={negative_median:.3f}")

    return df_norm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate log2FC from sgRNA count data")
    parser.add_argument("--metadata", required=True, help="Path to experiment metadata CSV file")
    parser.add_argument("--output", required=True, help="Path to output log2FC file")
    parser.add_argument("--template", action="store_true", help="Create metadata template file")
    parser.add_argument("--summary_metric", default="mean", help="Summary metric for replicates")
    parser.add_argument("--pseudo", type=float, default=1.0, help="Pseudocount for log2FC")
    parser.add_argument("--lod_limit", type=float, default=20.0, help="Limit of detection")
    parser.add_argument("--normalize", action="store_true", help="Normalize using negative controls")

    args = parser.parse_args()

    if args.template:
        template = create_experiment_metadata_template()
        template_path = args.metadata.replace('.csv', '_template.csv')
        template.to_csv(template_path, index=False)
        logger.info(f"Created metadata template: {template_path}")
        sys.exit(0)

    # Calculate log2FC
    df = get_logfc_dataframe_from_metadata(
        args.metadata,
        summary_metric=args.summary_metric,
        pseudo=args.pseudo,
        lod_limit=args.lod_limit
    )

    # Apply normalization if requested
    if args.normalize:
        df = normalize_negative_controls(df)

    # Save results
    df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Saved {len(df)} log2FC measurements to {args.output}")