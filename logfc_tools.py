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
from tqdm import tqdm

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


def get_count_matrix_from_dataframe(df):
    """
    Extract count matrix from dataframe.

    First column is assumed to be IDs, all other columns are count replicates.

    Args:
        df (pd.DataFrame): Count dataframe

    Returns:
        tuple: (sgRNA_ids, count_matrix) where count_matrix has shape (n_sgrnas, n_replicates)
    """
    # First column is IDs, regardless of name
    sgrna_ids = df.iloc[:, 0].values

    # All other columns are counts
    count_matrix = df.iloc[:, 1:].values.astype(float)

    logger.debug(f"Loaded {len(sgrna_ids)} sgRNAs with {count_matrix.shape[1]} replicate(s)")
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
    exp_groups = list(metadata.groupby(['strain', 'experiment']))
    print(f"\nProcessing {len(exp_groups)} experiment(s)...")
    for (strain, experiment), exp_group in tqdm(exp_groups, desc="Experiments", unit="exp"):
        logger.debug(f"Processing {strain} {experiment}")

        gen_groups = list(exp_group.groupby('generations'))
        for generations, gen_group in tqdm(gen_groups, desc=f"  Generations for {experiment}", unit="gen", leave=False):
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
            plus_ids_list = []
            minus_ids_list = []
            sgrna_ids = None

            # Build sequence to guide number mapping
            seq_to_guide_num = {}

            # Load plus condition files
            for count_file in tqdm(plus_count_files, desc=f"    Loading +ATC files (G{generations})", unit="file", leave=False):
                # Expand tilde in path
                count_file = os.path.expanduser(count_file)
                if not os.path.exists(count_file):
                    logger.error(f"Count file not found: {count_file}")
                    continue
                df = load_count_file(count_file)
                ids, counts = get_count_matrix_from_dataframe(df)
                plus_ids_list.append(ids)
                # Keep as matrix for now, will average later
                plus_data.append(counts)

            # Load minus condition files
            for count_file in tqdm(minus_count_files, desc=f"    Loading -ATC files (G{generations})", unit="file", leave=False):
                # Expand tilde in path
                count_file = os.path.expanduser(count_file)
                if not os.path.exists(count_file):
                    logger.error(f"Count file not found: {count_file}")
                    continue
                df = load_count_file(count_file)
                ids, counts = get_count_matrix_from_dataframe(df)
                minus_ids_list.append(ids)
                # Keep as matrix for now, will average later
                minus_data.append(counts)

            if not plus_data or not minus_data:
                logger.warning(f"No valid count data for {strain} {experiment} G{generations}")
                continue

            # Align all count files (plus AND minus) to a single canonical row
            # order BEFORE stacking. The log2FC step pairs +ATc and -ATc rows
            # POSITIONALLY, so every file must contribute rows for the same
            # sgRNAs in the same order. We therefore reindex each file BY ID
            # (never by positional masks) to the common set of sgRNAs.
            if plus_ids_list:
                sgrna_ids = plus_ids_list[0]
                all_ids = plus_ids_list + minus_ids_list
                identical = all(
                    len(ids) == len(sgrna_ids) and np.array_equal(ids, sgrna_ids)
                    for ids in all_ids
                )
                if not identical:
                    # Intersection of IDs present in EVERY plus and minus file
                    # (np.intersect1d returns a sorted, unique array).
                    common_ids = sgrna_ids
                    for ids in all_ids[1:]:
                        common_ids = np.intersect1d(common_ids, ids)

                    if len(common_ids) < 100:  # need a reasonable number of guides
                        logger.error(
                            f"Too few common sgRNAs ({len(common_ids)}) across the count "
                            f"files for {strain} {experiment} G{generations}; skipping")
                        continue
                    logger.warning(
                        f"Count files for {strain} {experiment} G{generations} do not "
                        f"share identical sgRNA sets/order; realigning all files to "
                        f"{len(common_ids)} common sgRNAs")

                    def _reindex_to_common(file_ids, matrix):
                        # Select this file's rows for common_ids, IN common_ids order.
                        pos = {sid: k for k, sid in enumerate(file_ids)}
                        sel = np.fromiter((pos[sid] for sid in common_ids),
                                          dtype=int, count=len(common_ids))
                        return matrix[sel]

                    plus_data = [_reindex_to_common(plus_ids_list[j], plus_data[j])
                                 for j in range(len(plus_data))]
                    minus_data = [_reindex_to_common(minus_ids_list[j], minus_data[j])
                                  for j in range(len(minus_data))]
                    sgrna_ids = common_ids

            # Convert to arrays for log2FC calculation
            try:
                # Concatenate all replicates (both within files and across files)
                # Each element in plus_data/minus_data is a matrix of shape (n_sgrnas, n_replicates_in_file)
                plus_matrix = np.hstack(plus_data) if len(plus_data) > 0 else np.array([[]])
                minus_matrix = np.hstack(minus_data) if len(minus_data) > 0 else np.array([[]])

                logger.debug(f"Plus matrix shape: {plus_matrix.shape} (sgRNAs x replicates)")
                logger.debug(f"Minus matrix shape: {minus_matrix.shape} (sgRNAs x replicates)")

                # Check that we have the same number of sgRNAs
                if plus_matrix.shape[0] != minus_matrix.shape[0]:
                    logger.error(f"Mismatch in sgRNA count: plus has {plus_matrix.shape[0]}, minus has {minus_matrix.shape[0]}")
                    continue

            except Exception as e:
                logger.error(f"Error creating count matrices: {e}")
                logger.info(f"  Plus data shapes: {[p.shape for p in plus_data]}")
                logger.info(f"  Minus data shapes: {[m.shape for m in minus_data]}")
                continue

            # Apply LOD filtering to experimental condition. The +ATc and -ATc
            # summaries are each computed ONCE per generation here (not once per
            # sgRNA row below), so this stays O(n) at the >2M-guide scale.
            plus_summary = get_summary_stat_function(summary_metric)(plus_matrix)
            minus_summary = get_summary_stat_function(summary_metric)(minus_matrix)
            good_detection = plus_summary >= lod_limit

            # First pass: build sequence to guide number mapping
            # We need to scan all IDs first to assign consistent numbering
            orf_seq_to_num = {}
            for sgrna_id in sgrna_ids:
                parts = sgrna_id.split('_')
                if len(parts) == 4:
                    # First guide
                    if parts[0] == "Negative":
                        orf1 = "Negative"
                    elif ':' in parts[0]:
                        orf1 = parts[0].split(':')[0]
                    else:
                        orf1 = parts[0]
                    seq1 = parts[1]

                    # Second guide
                    if parts[2] == "Negative":
                        orf2 = "Negative"
                    elif ':' in parts[2]:
                        orf2 = parts[2].split(':')[0]
                    else:
                        orf2 = parts[2]
                    seq2 = parts[3]

                    # Track unique sequences for each ORF
                    if (orf1, seq1) not in orf_seq_to_num:
                        if orf1 not in seq_to_guide_num:
                            seq_to_guide_num[orf1] = {}
                        if seq1 not in seq_to_guide_num[orf1]:
                            seq_to_guide_num[orf1][seq1] = len(seq_to_guide_num[orf1]) + 1
                        orf_seq_to_num[(orf1, seq1)] = seq_to_guide_num[orf1][seq1]

                    if (orf2, seq2) not in orf_seq_to_num:
                        if orf2 not in seq_to_guide_num:
                            seq_to_guide_num[orf2] = {}
                        if seq2 not in seq_to_guide_num[orf2]:
                            seq_to_guide_num[orf2][seq2] = len(seq_to_guide_num[orf2]) + 1
                        orf_seq_to_num[(orf2, seq2)] = seq_to_guide_num[orf2][seq2]

            # Calculate log2FC
            logfc_values = log2fc(minus_matrix, plus_matrix, pseudo=pseudo, summary_metric=summary_metric)

            # Create results for this generation
            for i, sgrna_id in enumerate(tqdm(sgrna_ids, desc=f"    Processing sgRNAs (G{generations})", unit="sgRNA", leave=False)):
                # Parse sgRNA information from ID
                # Format expected: RVBD0001:dnaA_GATGACGATTTGCTTG_RVBD0002:dnaN_ACCCGGGCGCCAAGTGCTCAGC
                # Or for negatives: Negative_GATGACGATTTGCTTG_RVBD0002:dnaN_ACCCGGGCGCCAAGTGCTCAGC
                parts = sgrna_id.split('_')

                # Check if this is paired guide data (should have 4 parts)
                if len(parts) == 4:
                    # Parse first guide
                    if parts[0] == "Negative":
                        orf1 = "Negative"
                        gene1 = ""
                    elif ':' in parts[0]:
                        orf1 = parts[0].split(':')[0]  # RVBD0001
                        gene1 = parts[0].split(':')[1]  # dnaA
                    else:
                        orf1 = parts[0]
                        gene1 = ""
                    seq1 = parts[1]  # SEQUENCE1

                    # Parse second guide
                    if parts[2] == "Negative":
                        orf2 = "Negative"
                        gene2 = ""
                    elif ':' in parts[2]:
                        orf2 = parts[2].split(':')[0]  # RVBD0002
                        gene2 = parts[2].split(':')[1]  # dnaN
                    else:
                        orf2 = parts[2]
                        gene2 = ""
                    seq2 = parts[3]  # SEQUENCE2

                    # Get guide numbers based on sequences
                    num1 = orf_seq_to_num.get((orf1, seq1), 1)
                    num2 = orf_seq_to_num.get((orf2, seq2), 1)

                    # Create combined ORF and SEQ for compatibility
                    orf = f"{orf1}_{orf2}"
                    seq = f"{seq1}_{seq2}"

                    # Create guide names with numbering (e.g., RVBD0001-1, Negative-3)
                    guide_name1 = f"{orf1}-{num1}"
                    guide_name2 = f"{orf2}-{num2}"

                    result_row = {
                        'strain': strain,
                        'experiment': experiment,
                        'generations': generations,
                        'ID': sgrna_id,
                        'orf': orf,
                        'seq': seq,
                        'orf1': orf1,
                        'orf2': orf2,
                        'seq1': seq1,
                        'seq2': seq2,
                        'guide_name': f"{guide_name1}_{guide_name2}",
                        'guide_name1': guide_name1,
                        'guide_name2': guide_name2,
                        'log2fc': logfc_values[i],
                        'exp_mean': plus_summary[i],
                        'ctrl_mean': minus_summary[i],
                        'good': good_detection[i]
                    }
                else:
                    # Fallback for unexpected formats
                    logger.warning(f"Unexpected ID format: {sgrna_id}")
                    result_row = {
                        'strain': strain,
                        'experiment': experiment,
                        'generations': generations,
                        'ID': sgrna_id,
                        'orf': sgrna_id,
                        'seq': "",
                        'log2fc': logfc_values[i],
                        'exp_mean': plus_summary[i],
                        'ctrl_mean': minus_summary[i],
                        'good': good_detection[i]
                    }

                results.append(result_row)

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    if output_path:
        logger.info(f"Saving results to {output_path}")
        result_df.to_csv(output_path, sep='\t', index=False)

    return result_df


def add_single_mutant_fitness(df, negative_orf_names=["Negative", "NT", "NonTargeting"]):
    """
    Add log2fc_nt1 and log2fc_nt2 columns with single mutant fitness values.

    These represent the fitness of guide1+negative and guide2+negative pairs,
    essential for calculating genetic interaction scores.

    Args:
        df (pd.DataFrame): DataFrame with log2FC values
        negative_orf_names (list): ORF names that represent negative controls

    Returns:
        pd.DataFrame: DataFrame with log2fc_nt1 and log2fc_nt2 columns added
    """
    logger.info("Adding single mutant fitness values (log2fc_nt1, log2fc_nt2)")
    df = df.copy()

    # Only process if we have paired guide columns
    if not all(col in df.columns for col in ['orf1', 'orf2', 'seq1', 'seq2']):
        logger.warning("Paired guide columns not found, skipping single mutant fitness calculation")
        return df

    # Calculate log2fc_nt1 (guide1 + negative control)
    log2fc_nt1_values = {}
    log2fc_nt2_values = {}

    groups = list(df.groupby(['strain', 'experiment', 'generations']))
    for (strain, experiment, generation), group in tqdm(groups, desc="Finding single mutant fitness", unit="group", leave=False):
        # For each unique guide1, find its fitness with negative control
        for seq1 in group['seq1'].unique():
            if pd.isna(seq1):
                continue
            # Find where guide1 is paired with negative control
            mask = (group['seq1'] == seq1) & group['orf2'].isin(negative_orf_names)
            if mask.any():
                log2fc_nt1_values[(strain, experiment, generation, seq1)] = group.loc[mask, 'log2fc'].median()

        # For each unique guide2, find its fitness with negative control
        for seq2 in group['seq2'].unique():
            if pd.isna(seq2):
                continue
            # Find where guide2 is paired with negative control
            mask = (group['seq2'] == seq2) & group['orf1'].isin(negative_orf_names)
            if mask.any():
                log2fc_nt2_values[(strain, experiment, generation, seq2)] = group.loc[mask, 'log2fc'].median()

    # Apply values to dataframe
    df['log2fc_nt1'] = df.apply(
        lambda row: log2fc_nt1_values.get(
            (row['strain'], row['experiment'], row['generations'], row.get('seq1', None)),
            np.nan
        ) if 'seq1' in row else np.nan,
        axis=1
    )

    df['log2fc_nt2'] = df.apply(
        lambda row: log2fc_nt2_values.get(
            (row['strain'], row['experiment'], row['generations'], row.get('seq2', None)),
            np.nan
        ) if 'seq2' in row else np.nan,
        axis=1
    )

    # Calculate log2fc_delta (deviation from additive expectation - input for GI modeling)
    # Note: This is NOT the final GI score - it's the input for Stan modeling
    if 'log2fc_nt1' in df.columns and 'log2fc_nt2' in df.columns:
        df['log2fc_delta'] = df['log2fc'] - (df['log2fc_nt1'] + df['log2fc_nt2'])
        logger.info(f"Calculated log2fc_delta for {df['log2fc_delta'].notna().sum()} measurements")

    return df


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

    groups = list(df.groupby(['strain', 'experiment', 'generations']))
    for (strain, experiment, generation), group in tqdm(groups, desc="Normalizing by negative controls", unit="group", leave=False):
        # Find double negative controls (BOTH guides are negative/non-targeting)
        # Check if we have the guide columns to identify double negatives
        if 'orf1' in group.columns and 'orf2' in group.columns:
            # Both guides must be negative controls
            is_double_negative = (
                group['orf1'].isin(negative_orf_names) &
                group['orf2'].isin(negative_orf_names)
            )
        elif 'orf' in group.columns:
            # Fallback: check the combined orf column for patterns like "Negative_Negative"
            is_double_negative = group['orf'].apply(
                lambda x: all(part in negative_orf_names for part in str(x).split('_'))
                if pd.notna(x) else False
            )
        else:
            logger.warning(f"Cannot identify negative controls - missing orf columns")
            continue

        if not is_double_negative.any():
            logger.warning(f"No double negative controls found for {strain} {experiment} generation {generation}")
            continue

        # Calculate the median log2FC of double negative controls
        negative_median = np.median(group.loc[is_double_negative, 'log2fc'])

        # Subtract from all values in this group to normalize
        group_indices = group.index
        df_norm.loc[group_indices, 'log2fc'] = group['log2fc'] - negative_median

        logger.debug(f"Applied normalization: {strain} {experiment} generation {generation}, "
                    f"double negatives={is_double_negative.sum()}, offset={negative_median:.3f}")

    return df_norm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate log2FC from sgRNA count data")
    parser.add_argument("--metadata", required=True, help="Path to experiment metadata CSV file")
    parser.add_argument("--output", help="Path to output log2FC file (required unless --template)")
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

    if not args.output:
        parser.error("--output is required (except with --template)")

    # Calculate log2FC
    df = get_logfc_dataframe_from_metadata(
        args.metadata,
        summary_metric=args.summary_metric,
        pseudo=args.pseudo,
        lod_limit=args.lod_limit
    )

    # Check if we have any data
    if len(df) == 0:
        logger.error("No data to process - check that count files exist and are readable")
        sys.exit(1)

    # Apply normalization if requested
    if args.normalize:
        df = normalize_negative_controls(df)

    # Add single mutant fitness values for GI calculation (if paired guide data)
    if 'orf1' in df.columns and 'orf2' in df.columns:
        df = add_single_mutant_fitness(df)

    # Save results
    df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Saved {len(df)} log2FC measurements to {args.output}")
    if 'log2fc_delta' in df.columns:
        logger.info(f"Includes log2fc_delta (GI scores) for {df['log2fc_delta'].notna().sum()} measurements")