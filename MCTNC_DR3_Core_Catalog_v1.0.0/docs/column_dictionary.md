# Column Dictionary

This document defines the most important columns in the M-CTNC Gaia DR3 core-catalog release. Some diagnostic files contain additional run-specific columns; see the local README files in each diagnostic directory for package-specific details.

## Common identifiers

| Column | Meaning |
|---|---|
| `Cluster` | Cluster identifier used in cluster-level summary products. |
| `cluster` | Cluster identifier used in benchmark, diagnostic, ablation, and baseline products. |
| `Cluster_Name` | Cluster identifier used in source-level release tables. |
| `source_id` | Gaia DR3 source identifier. Read this column as a string to avoid precision loss in spreadsheet software. |
| `mode` | Operating mode, usually `CORE_BENCHMARK` or `HALO_EXPLORATION`. |
| `Extraction_Mode` | Source-level operating-mode label in auxiliary source catalogs. |

## Source-level astrometric and photometric columns

These columns occur in the principal source catalog and auxiliary source catalogs.

| Column | Meaning |
|---|---|
| `ra` | Gaia DR3 right ascension, in degrees. |
| `dec` | Gaia DR3 declination, in degrees. |
| `parallax` | Gaia DR3 parallax, in milliarcseconds. |
| `pmra` | Gaia DR3 proper motion in right ascension, in milliarcseconds per year. |
| `pmdec` | Gaia DR3 proper motion in declination, in milliarcseconds per year. |
| `phot_g_mean_mag` | Gaia DR3 $G$-band mean magnitude. Retained for validation and downstream reuse; not used in M-CTNC membership inference. |
| `phot_bp_mean_mag` | Gaia DR3 $G_{\rm BP}$ mean magnitude. Present in auxiliary source catalogs where available. |
| `phot_rp_mean_mag` | Gaia DR3 $G_{\rm RP}$ mean magnitude. Present in auxiliary source catalogs where available. |

## Cluster-level release summary columns

These columns occur in `data/ApJS_Table1_Cluster_Properties.csv`.

| Column | Meaning |
|---|---|
| `N_Cone` | Number of Gaia DR3 sources in the local field cone. |
| `N_True_Lit` | Number of literature-derived benchmark members in the local field. |
| `N_Pred_MCTNC` | Number of released M-CTNC members. |
| `F1_Score` | Cluster-level $F_1$ score against the adopted benchmark layer. |
| `R50_Core_deg` | Characteristic core radius proxy in degrees. |
| `R90_Halo_deg` | Characteristic halo/outer radius proxy in degrees. |
| `Mean_Plx_mas` | Mean parallax of the released member set, in milliarcseconds. |
| `Exec_Time_s` | Execution time in seconds. |

## Benchmark-summary columns

These columns occur in `data/ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv` and `data/ApJS_TableA_Full_Benchmark_HALO_EXPLORATION.csv`.

| Column | Meaning |
|---|---|
| `tier` | Benchmark-performance tier identifier. |
| `tier_label` | Human-readable benchmark tier label. |
| `n_cone` | Number of sources in the local field cone. |
| `n_true_in_cone` | Number of benchmark reference members in the local field cone. |
| `n_pred` | Number of predicted/released members. |
| `precision` | Cluster-level precision against the benchmark layer. |
| `recall` | Cluster-level recall against the benchmark layer. |
| `f1` | Cluster-level $F_1$ score. |
| `contam` | Contamination fraction, defined as $1-\mathrm{precision}$. |
| `runtime_s` | Runtime in seconds. |
| `center_mode` | Selected center-handling mode. |
| `center_shift_arcmin` | Center shift in arcminutes. |
| `objective` | Final arbitration objective value. |

## Anomaly-audit columns

These columns occur in `data/ApJS_TableB_Anomaly_Audit_CORE_BENCHMARK.csv` and `data/ApJS_TableB_Anomaly_Audit_HALO_EXPLORATION.csv`.

| Column | Meaning |
|---|---|
| `n_catalog_only` | Number of benchmark/reference members not recovered by the release. |
| `n_pred_only` | Number of released candidates not present in the benchmark/reference layer. |
| `label_suspect_flag` | Flag for suspected reference-label issues. |
| `fatal_disruption_flag` | Flag for severe or pathological benchmark disagreement. |
| `faint_flag` | Flag indicating a faint-member related discrepancy. |
| `halo_flag` | Flag indicating a halo/boundary-related discrepancy. |
| `audit_r_median_deg` | Median angular diagnostic radius in degrees. |
| `audit_pm_disp_masyr` | Proper-motion dispersion diagnostic in milliarcseconds per year. |
| `audit_plx_mad_mas` | Parallax median absolute deviation diagnostic in milliarcseconds. |
| `audit_reasons` | Text attribution for the anomaly audit. |
| `diagnostic_png_path` | Path to the associated diagnostic figure, when available. |

## Photometric-independence audit columns

These columns occur in `diagnostics/photometric_independence/`.

| Column | Meaning |
|---|---|
| `n_truth` | Number of benchmark/reference sources used in the photometric comparison. |
| `n_release` | Number of released sources used in the photometric comparison. |
| `n_tp` | Number of true-positive sources relative to the benchmark layer. |
| `n_fn` | Number of benchmark members not recovered by the release. |
| `n_fp` | Number of released sources not present in the benchmark layer. |
| `ks_stat` | Kolmogorov--Smirnov statistic comparing released and benchmark $G$ distributions. |
| `ks_pvalue` | Kolmogorov--Smirnov $p$-value. |
| `g_median_truth` | Median $G$ magnitude of the benchmark/reference layer. |
| `g_median_release` | Median $G$ magnitude of the released member set. |
| `delta_g_median` | Difference in median $G$ magnitude between release and benchmark. |
| `g_q90_truth` | 90th-percentile/faint-end $G$ statistic for the benchmark/reference layer. |
| `g_q90_release` | 90th-percentile/faint-end $G$ statistic for the released member set. |
| `delta_g_q90` | Difference in faint-end $G_{90}$ between release and benchmark. |
| `quantile_bin` | Cluster-relative apparent-magnitude quantile bin. |
| `recall` | Recall within the cluster-relative magnitude bin. |

## Robustness and reproducibility columns

These columns occur in `diagnostics/robustness_reproducibility/`.

| Column | Meaning |
|---|---|
| `parameter` | Perturbed parameter family. |
| `n_runs` | Number of perturbation runs included in the family summary. |
| `median_abs_delta_f1` | Median absolute $F_1$ response under perturbation. |
| `p90_abs_delta_f1` | 90th percentile of absolute $F_1$ response. |
| `mean_abs_delta_f1` | Mean absolute $F_1$ response. |
| `frac_le_0.01` | Fraction of systems with $|\Delta F_1|\leq0.01$. |
| `frac_le_0.03` | Fraction of systems with $|\Delta F_1|\leq0.03$. |
| `frac_le_0.05` | Fraction of systems with $|\Delta F_1|\leq0.05$. |
| `center_effect` | Center-refinement effect category: `improved`, `stable`, or `degraded`. |
| `delta_f1` | Change in $F_1$ after the audited operation. |
| `abs_delta_f1` | Absolute value of `delta_f1`. |

## Ablation columns

These columns occur in `diagnostics/ablation/tables/`.

| Column | Meaning |
|---|---|
| `variant` | Ablation variant name. |
| `mean_f1` | Mean $F_1$ score across the 359-cluster benchmark population. |
| `median_f1` | Median $F_1$ score. |
| `p90_f1` | 90th-percentile $F_1$ statistic. |
| `mean_precision` | Mean precision. |
| `mean_recall` | Mean recall. |
| `mean_contam` | Mean contamination fraction. |
| `success_frac_f1_ge_0_9` | Fraction of systems satisfying $F_1\geq0.9$. |
| `tie_n` | Number of systems treated as tied relative to the full-core comparator. In `ablation_delta_vs_fullcore.csv`, ties use $|\Delta F_1|\leq0.01$. |
| `full_core_better_n` | Number of systems where the canonical full-core comparator is materially better. |
| `ablation_better_n` | Number of systems where the ablated variant is materially better. |

## Baseline-comparison columns

These columns occur in `diagnostics/baseline/tables/`.

| Column | Meaning |
|---|---|
| `method` | Method name: `M-CTNC`, `hdbscan`, `heuristic_cut`, or `gmm_bic`. |
| `mean_f1` | Mean $F_1$ score across the 359-cluster benchmark population. |
| `median_f1` | Median $F_1$ score. |
| `mean_precision` | Mean precision. |
| `mean_recall` | Mean recall. |
| `mean_contam` | Mean contamination fraction. |
| `median_runtime_s` | Median runtime in seconds. |
| `success_frac_f1_ge_0_9` | Fraction of systems satisfying $F_1\geq0.9$. |

## Notes on reading CSV files

When using Python, read Gaia source identifiers as strings:

```python
import pandas as pd

cat = pd.read_csv(
    "data/ApJS_Table2_MCTNC_Master_Catalog.csv",
    dtype={"source_id": str}
)
```

Spreadsheet software may display long `source_id` values in scientific notation. Avoid resaving the catalog from spreadsheet software unless `source_id` is explicitly treated as text.
