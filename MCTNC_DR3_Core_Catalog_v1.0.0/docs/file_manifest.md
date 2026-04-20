# File Manifest

This document describes the files in the `MCTNC_DR3_Core_Catalog_v1.0.0` release.

## Top-level files

| File | Description |
|---|---|
| `README.md` | Main repository overview, recommended entry points, and citation guidance. |
| `LICENSE` | License statement for data products and source code. |
| `CITATION.cff` | Citation metadata for GitHub/Zenodo. The Zenodo DOI should be added after release archival. |
| `VERSION` | Release version tag. For the manuscript release, use `v1.0.0`. |
| `requirements.txt` | Python package dependencies used by the analysis and diagnostic scripts. |
| `.gitignore` | Files excluded from the GitHub repository. |

## `data/`

The `data/` directory contains the formal release products. These are the principal files that support the catalog release.

| File | Description |
|---|---|
| `ApJS_Table1_Cluster_Properties.csv` | Cluster-level summary for the 359 benchmark open clusters. |
| `ApJS_Table2_MCTNC_Master_Catalog.csv` | Principal `CORE_BENCHMARK` astrometric core-member source catalog. This is the default scientific product. |
| `ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv` | Cluster-level benchmark summary for the principal `CORE_BENCHMARK` operating mode. |
| `ApJS_TableA_Full_Benchmark_HALO_EXPLORATION.csv` | Cluster-level benchmark summary for the auxiliary `HALO_EXPLORATION` operating mode. |
| `ApJS_TableB_Anomaly_Audit_CORE_BENCHMARK.csv` | Anomaly-audit layer for non-ideal `CORE_BENCHMARK` cases. |
| `ApJS_TableB_Anomaly_Audit_HALO_EXPLORATION.csv` | Anomaly-audit layer for non-ideal `HALO_EXPLORATION` cases. |
| `ApJS_TableC_Master_Catalog_CORE_BENCHMARK.csv` | Auxiliary source-level table for the `CORE_BENCHMARK` mode, including mode labels and expanded photometry. |
| `ApJS_TableC_Master_Catalog_HALO_EXPLORATION.csv` | Auxiliary source-level table for the `HALO_EXPLORATION` mode, including mode labels and expanded photometry. |

## `diagnostics/photometric_independence/`

This directory contains the Step 10 photometric-independence audit. Gaia photometry is not used in the M-CTNC clustering space or final arbitration; these products are post hoc validation outputs.

| File | Description |
|---|---|
| `appendixC_prewrite_summary.txt` | Human-readable summary of the photometric audit and Appendix C writing basis. |
| `population_photometric_independence_per_cluster.csv` | Cluster-level KS, median-$G$, faint-end $G_{90}$, TP/FN/FP diagnostics for 359 systems. |
| `population_photometric_independence_summary.csv` | One-row population-level summary of the photometric audit. |
| `population_photometric_outlier_clusters.csv` | KS-significant photometric outlier clusters. |
| `population_relative_quantile_recall.csv` | Cluster-relative recall in five apparent-magnitude quantile bins for each cluster. |
| `population_relative_quantile_recall_summary.csv` | Population summary of the quantile-recall audit. |
| `representative_cluster_selection.csv` | Representative CMD cases selected for Appendix C diagnostic panels. |
| `step10_run.log` | Run log for the Step 10 photometric audit. |
| `step10_run_metadata.json` | Metadata for the Step 10 audit, including overlap counts and representative-cluster roles. |

## `diagnostics/robustness_reproducibility/`

This directory contains production-consistency, perturbation, and center-refinement diagnostics.

| File | Description |
|---|---|
| `APJS_Evidence_Package_POLISHED.xlsx` | Workbook containing polished evidence tables for reproducibility and robustness diagnostics. |
| `Table01_population_robustness_ranking_polished.csv` | Population-level parameter-family robustness ranking. |
| `Table02_center_refinement_population_summary_polished.csv` | Center-refinement population summary: improved, stable, degraded. |
| `Table03_center_refinement_material_cases_only.csv` | Material center-refinement cases with $|\Delta F_1|>0.01$. |
| `Table04_exact_mismatch_cases_only.csv` | Appendix-level exact-rerun or center-mode mismatch cases. |
| `Table05_overview_metrics.csv` | Overview KPI table for rerun, robustness, and center-refinement diagnostics. |

## `diagnostics/ablation/`

This directory contains the canonical `CORE_BENCHMARK` ablation suite.

| File or directory | Description |
|---|---|
| `README_ablation.txt` | Local README for the ablation package. |
| `mctnc_ablation_package.xlsx` | Workbook containing consolidated ablation diagnostics. |
| `tables/` | Consolidated ablation tables. These are the preferred tables for analysis. |
| `cache/` | Per-variant intermediate exports. These support provenance checks but are not primary tables. |

Recommended `diagnostics/ablation/tables/` contents:

| File | Description |
|---|---|
| `ablation_cluster_results.csv` | Cluster-level results for the full-core comparator and ablated variants. |
| `ablation_overall_summary.csv` | Overall performance summary by ablation variant. |
| `ablation_comparison_matrix.csv` | Comparison matrix among ablation variants. |
| `ablation_delta_vs_fullcore.csv` | Variant-level differences relative to the canonical full-core comparator. `tie_n` uses a material tolerance of $|\Delta F_1|\leq0.01$. |
| `ablation_design_table.csv` | Design description for each ablation variant. |
| `ablation_rank_table.csv` | Ranked ablation summary. |
| `ablation_tier_stratified_summary.csv` | Ablation behavior stratified by benchmark tier. |
| `ablation_import_audit_table.csv` | Import audit for the canonical full-core comparator. |
| `ablation_fullcore_source_audit.csv` | Source audit for the imported full-core reference. |

Recommended `diagnostics/ablation/cache/` contents:

| File | Description |
|---|---|
| `cache_fixed_single_k_cluster_results.csv` | Per-cluster intermediate results for the fixed single-$k$ variant. |
| `cache_no_anchor_prior_cluster_results.csv` | Per-cluster intermediate results for the no-anchor-prior variant. |
| `cache_no_candidate_protocol_cluster_results.csv` | Per-cluster intermediate results for the no-candidate-protocol variant. |
| `cache_no_center_refinement_cluster_results.csv` | Per-cluster intermediate results for the no-center-refinement variant. |
| `cache_no_objective_regularization_cluster_results.csv` | Per-cluster intermediate results for the no-objective-regularization variant. |
| `cache_no_quality_cut_cluster_results.csv` | Per-cluster intermediate results for the no-quality-cut variant. |
| `cache_no_support_gate_cluster_results.csv` | Per-cluster intermediate results for the no-support-gate variant. |

Empty ablation placeholder files such as `ablation_case_selection.csv` and `ablation_fullcore_vs_internal_audit.csv` are not required in the public release.

## `diagnostics/baseline/`

This directory contains the shared-preprocessing external-baseline suite.

| File or directory | Description |
|---|---|
| `README_part5.txt` | Local README for the baseline package. |
| `part5_baseline_package.xlsx` | Workbook containing consolidated baseline diagnostics. |
| `tables/` | Consolidated baseline comparison tables. These are the preferred tables for analysis. |
| `cache/` | Per-method intermediate exports. These support provenance checks but are not primary tables. |

Recommended `diagnostics/baseline/tables/` contents:

| File | Description |
|---|---|
| `baseline_cluster_results.csv` | Cluster-level results for baseline methods. |
| `baseline_overall_summary.csv` | Overall method-level summary for M-CTNC and external baselines. |
| `comparison_matrix.csv` | Cluster-level comparison matrix. |
| `fairness_audit_table.csv` | Shared-preprocessing fairness audit. |
| `mctnc_import_audit_table.csv` | Audit of imported M-CTNC benchmark reference used for comparison. |
| `mctnc_vs_baseline_winloss.csv` | Win/loss comparison between M-CTNC and baselines. |
| `method_rank_table.csv` | Ranked method-level summary. |
| `representative_case_selection.csv` | Representative baseline-comparison cases. |
| `tier_stratified_summary.csv` | Baseline comparison stratified by benchmark tier. |
| `top40_mctnc_minus_best_baseline_cases.csv` | Top cases by M-CTNC advantage over the strongest baseline. |

Recommended `diagnostics/baseline/cache/` contents:

| File | Description |
|---|---|
| `cache_gmm_bic_cluster_results.csv` | Per-cluster intermediate results for the GMM+BIC baseline. |
| `cache_hdbscan_cluster_results.csv` | Per-cluster intermediate results for HDBSCAN. |
| `cache_heuristic_cut_cluster_results.csv` | Per-cluster intermediate results for the heuristic-cut baseline. |

## `reference/`

| File | Description |
|---|---|
| `ocfinder_table1.csv` | Literature/reference input table used in benchmark construction. The exact provenance should be stated in the main `README.md`. |
| `ocfinder_table2.csv` | Literature/reference input table used in benchmark construction. The exact provenance should be stated in the main `README.md`. |

## `scripts/`

| File | Description |
|---|---|
| `MCTNC.py` | Main M-CTNC production workflow. |
| `MCTNC-ablation.py` | Canonical ablation-suite workflow. |
| `MCTNC-baseline.py` | Shared-preprocessing baseline-comparison workflow. |
| `MCTNC-sensitivity.py` | Sensitivity / robustness workflow. |
| `MCTNC-sensitivity-1.py` | Evidence-packaging / reproducibility diagnostic workflow using relative-path discovery. |
| `MCTNC-sensitivity-2.py` | Polished evidence figure/table package workflow. |

## `docs/`

| File | Description |
|---|---|
| `file_manifest.md` | This file. |
| `column_dictionary.md` | Column descriptions for the primary data and diagnostic products. |
| `usage_guide.md` | Practical examples for reading and using the released catalogs. |
| `reproduction_notes.md` | Notes on scripts, provenance, and reproducibility boundaries. |
| `manuscript_table_mapping.md` | Mapping between manuscript tables and archived files. |
