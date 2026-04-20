# Manuscript Table Mapping

This file maps the manuscript tables and major validation components to files in the archived release.

## Main data products

| Manuscript item | Release file |
|---|---|
| Table 1: cluster-level release summary | `data/ApJS_Table1_Cluster_Properties.csv` |
| Table 2: principal source catalog | `data/ApJS_Table2_MCTNC_Master_Catalog.csv` |
| Table A: full benchmark summary, CORE mode | `data/ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv` |
| Table A: full benchmark summary, HALO mode | `data/ApJS_TableA_Full_Benchmark_HALO_EXPLORATION.csv` |
| Table B: anomaly audit, CORE mode | `data/ApJS_TableB_Anomaly_Audit_CORE_BENCHMARK.csv` |
| Table B: anomaly audit, HALO mode | `data/ApJS_TableB_Anomaly_Audit_HALO_EXPLORATION.csv` |
| Table C: auxiliary source catalog, CORE mode | `data/ApJS_TableC_Master_Catalog_CORE_BENCHMARK.csv` |
| Table C: auxiliary source catalog, HALO mode | `data/ApJS_TableC_Master_Catalog_HALO_EXPLORATION.csv` |

## Validation and diagnostic products

| Manuscript component | Release files |
|---|---|
| Photometric-independence population audit | `diagnostics/photometric_independence/population_photometric_independence_summary.csv`; `diagnostics/photometric_independence/population_photometric_independence_per_cluster.csv` |
| Photometric outlier disclosure | `diagnostics/photometric_independence/population_photometric_outlier_clusters.csv` |
| Magnitude-quantile recall audit | `diagnostics/photometric_independence/population_relative_quantile_recall.csv`; `diagnostics/photometric_independence/population_relative_quantile_recall_summary.csv` |
| Appendix C representative CMD cases | `diagnostics/photometric_independence/representative_cluster_selection.csv` |
| Production-consistent rerun and overview metrics | `diagnostics/robustness_reproducibility/Table05_overview_metrics.csv` |
| Perturbation robustness ranking | `diagnostics/robustness_reproducibility/Table01_population_robustness_ranking_polished.csv` |
| Center-refinement summary | `diagnostics/robustness_reproducibility/Table02_center_refinement_population_summary_polished.csv`; `diagnostics/robustness_reproducibility/Table03_center_refinement_material_cases_only.csv` |
| Rerun mismatch cases | `diagnostics/robustness_reproducibility/Table04_exact_mismatch_cases_only.csv` |
| Canonical ablation summary | `diagnostics/ablation/tables/ablation_overall_summary.csv`; `diagnostics/ablation/tables/ablation_cluster_results.csv` |
| Ablation trade-off and ranking | `diagnostics/ablation/tables/ablation_delta_vs_fullcore.csv`; `diagnostics/ablation/tables/ablation_rank_table.csv`; `diagnostics/ablation/tables/ablation_tier_stratified_summary.csv` |
| Shared-preprocessing baseline comparison | `diagnostics/baseline/tables/baseline_overall_summary.csv`; `diagnostics/baseline/tables/baseline_cluster_results.csv` |
| Baseline fairness audit | `diagnostics/baseline/tables/fairness_audit_table.csv`; `diagnostics/baseline/tables/mctnc_import_audit_table.csv` |
| M-CTNC vs baseline win/loss analysis | `diagnostics/baseline/tables/mctnc_vs_baseline_winloss.csv`; `diagnostics/baseline/tables/top40_mctnc_minus_best_baseline_cases.csv` |

## Figures

The manuscript figures are generated from the release products and diagnostic tables. If final figure files are included in the repository, place them under:

```text
figures/
```

or under a diagnostic-specific subdirectory such as:

```text
figures/appendix/
```

## Spectroscopic Appendix D

The spectroscopic cross-check in Appendix D is treated as a coverage-limited auxiliary check. Its raw diagnostic products are not included in this release package. The public release focuses on the principal catalog products and primary validation diagnostics.
