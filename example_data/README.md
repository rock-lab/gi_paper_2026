# Toy dataset

A small, self-contained slice of the real *M. tuberculosis* (H37Rv) dual-CRISPRi
genetic-interaction (GI) screens, cut down to **15 published genes** so readers
can reproduce the joint GI-probability model end to end without the full
~2.2M-construct library.

Everything here is REAL data (real barcode counts and real per-screen GI
scores), row-filtered to the 15 genes below. It is regenerated from the raw HPC
sources by [`../make_example_data.py`](../make_example_data.py) (a maintainer-only
script — readers do not need to run it).

## The 15 genes ("Set A")

The set is deliberately *representative*: most pairs do not interact, with a few
clear, biologically sensible signals — five signal pairs plus five inert filler
genes.

| role | ORF ids | genes | interaction |
|------|---------|-------|-------------|
| negative | RVBD0392c / RVBD1854c | ndhA / ndh | aggravating (redundant NADH dehydrogenases) |
| negative | RVBD2754c / RVBD2764c | thyX / thyA | aggravating (redundant thymidylate synthases) |
| negative | RVBD0050 / RVBD3682 | ponA1 / ponA2 | aggravating (redundant class-A PBPs) |
| positive | RVBD2193 / RVBD2200c | ctaE / ctaC | alleviating (cytochrome-c oxidase) |
| positive | RVBD1304 / RVBD3795 | atpB / embB | alleviating |
| filler | RVBD1025, RVBD1822, RVBD3261, RVBD2447c, RVBD2794c | RVBD1025, pgsA2, fbiA, folC, pptT | ~none (inert background) |

These 15 genes give **105 unordered ORF pairs** (+ 15 self-pairs). A pair is
identified by `orf_pair`, the canonical sorted `"orf1_orf2"` string. One weak
background edge (thyA–ponA1, P≈0.62) also rides along, illustrating a marginal
call.

## Two experiments

| tag  | experiment          | Cas9 induction | timepoint label | count dir        |
|------|---------------------|----------------|-----------------|------------------|
| exp1 | `tb_lowcas9_100ng`  | 100 ng/mL ATc  | passage `P{n}`  | `counts/exp1/`   |
| exp2 | `tb_lowcas9_500ng`  | 500 ng/mL ATc  | generation `G{n}` | `counts/exp2/` |

For each timepoint there is a `+ATc` (`plus`, Cas9 ON) and a `-ATc` (`minus`,
uninduced control) pooled count table.

## File inventory

```
example_data/
├── README.md                              this file
├── experiment_metadata_exp1.csv           exp1 timepoint -> generations, ATc, count file
├── experiment_metadata_exp2.csv           exp2 timepoint -> generations, ATc, count file
├── guide_name_map.tsv                     single-sgRNA library map (toy genes + Negative)
├── gi_input/
│   ├── result_summary_long_df_exp1_toy.tsv   PRE-DERIVED GOLDEN per-screen GI scores (exp1)
│   └── result_summary_long_df_exp2_toy.tsv   PRE-DERIVED GOLDEN per-screen GI scores (exp2)
├── counts/
│   ├── exp1/counts_P{0,0_5,1,1_5,2,3,4,5,6}_{plus,minus}.txt   pooled barcode counts
│   └── exp2/counts_G{0,2,4,6,8,10,12,14,16,18,20}_{plus,minus}.txt
└── expected/
    └── merged_quad_me_trunc_halfsmeared_toy.tsv   GOLDEN joint-model output (primary model)
```

### `gi_input/*.tsv` are PRE-DERIVED GOLDEN values

The two `gi_input/result_summary_long_df_exp{1,2}_toy.tsv` files are the
**input to the joint model**, not something you compute in this repo. Each row
is one ORF pair in one screen, and carries that screen's GI score
(`delta_prime_median`) and its standard error (`sd_delta_prime_median`), which
were produced upstream by the per-guide-pair two-line fitness model. They are
checkpointed here so you can run the joint model directly. Columns:

```
(unnamed index)  orf1  orf2  orf_pair  delta_prime_median  sd_delta_prime_median
                 prob_interaction_median  correlation  name1  name2
```

The leading unnamed column is a row index (the joint runner and the merge script
both read these with `index_col=0` and merge on `orf_pair`).

### `expected/merged_quad_me_trunc_halfsmeared_toy.tsv` is the GOLDEN output

This is the published joint-model result for the **primary model**
(`quad_me_trunc_halfsmeared`, winsorize percentile 0.10), sliced to the 120
example pairs (105 gene pairs + 15 self-pairs). Use it to check that your re-run
reproduces the paper. Its 20 columns are:

```
orf1 orf2 name1 name2
gi_score_exp1 se_exp1 gi_score_overlaps_zero_exp1
gi_score_exp2 se_exp2 gi_score_overlaps_zero_exp2
correlation_exp1 correlation_exp2
prob_interaction_median_exp1 prob_interaction_median_exp2
prob_interaction prob_aggravating prob_alleviating
prob_discordant prob_no_interaction prob_concordant
```

An interaction is "called" at `prob_interaction >= 0.5` (a stricter, "confident"
cutoff of `0.95` is also reported in the paper).

## Construct-id grammar

Each row of a count table is one dual-guide **construct**, identified in the `Id`
column. A construct id is `LEFT_RIGHT`, where each side is either

* `ORF:name_SEQ` — e.g. `RVBD0392c:ndhA_GATGACGATTTGCTTG`, or
* `Negative_SEQ` — a non-targeting (NT) control, e.g. `Negative_GGTGGCC...`

so a full id has exactly three underscores, e.g.

```
RVBD0392c:ndhA_GATGACGATTTGCTTG_RVBD1854c:ndh_GGTAGCAACTGCCGAAACT
RVBD0392c:ndhA_GATGACGATTTGCTTG_Negative_GGTGGCCTGTCTGCACGGTGGGG
```

The ORF of a side is the text before the first `:` (`Negative` has no `:` and
maps to itself). A construct is kept in the toy set iff **both** sides target a
toy gene or a Negative control. Count tables have an `Id` column plus one or more
integer count columns (`lane_x`, `lane_y`, `lane`, ...); the exp1 `P0` files also
carried a stray `test_counts_*` column, which is dropped during subsetting.

## Passage / generation mapping

**exp1 (100 ng/mL ATc)** — passages sampled over ~32 generations:

| passage | P0 | P0.5 | P1  | P1.5 | P2   | P3   | P4   | P5   | P6   |
|---------|----|------|-----|------|------|------|------|------|------|
| gens    | 0  | 2.6  | 5.4 | 8.1  | 10.8 | 16.2 | 21.6 | 27.1 | 32.5 |

**exp2 (500 ng/mL ATc)** — sampled at `G0, G2, ..., G20`.

> **Note (exp2 generations):** the `generations` column in
> `experiment_metadata_exp2.csv` is taken from the `G{n}` timepoint label in each
> file name (i.e. `G8` → 8 generations), because the source records these
> timepoints only as `G{n}` labels rather than as an explicit generations field.
> This nominal mapping is what the example uses; treat it as an approximate
> generation count rather than an exact one.

## Provenance

All files are regenerated by `../make_example_data.py`, which row-filters the real
HPC sources (per-screen `result_summary_long_df` TSVs with SE, the pooled
`counts_*` tables, the golden `merged_quad_me_trunc_halfsmeared_w010.tsv`, and
the `H37Rv_single_sgrna_library.tsv` guide map) down to the 15 genes above.
