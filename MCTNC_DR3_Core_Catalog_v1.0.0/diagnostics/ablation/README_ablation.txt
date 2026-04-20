Unified Part-Six ablation suite for M-CTNC
=========================================
Version: mctnc_ablation_v1_2_corebenchmark_canonical_release
Variants: full_core, no_quality_cut, no_candidate_protocol, no_anchor_prior, fixed_single_k, no_support_gate, no_center_refinement, no_objective_regularization
Imported M-CTNC reference: D:\HuaweiMoveData\Users\王浩\Desktop\课题组\DUCT-Clust\ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv
Benchmark policy: canonical CORE_BENCHMARK ablation suite
Primary full-core policy: use imported CORE_BENCHMARK reference as the canonical full-core comparator whenever available
Reference rejection policy: halo/exploration-style result files are excluded from canonical import
Canonical full-core source: imported_core_benchmark_reference
Internal full-core diagnostic enabled: 0
M-CTNC mode filter policy: explicit_core_mode_filter
M-CTNC source-name policy: non_halo_filename
M-CTNC duplicate policy: none

Key exported tables:
  tables/ablation_cluster_results.csv
  tables/ablation_overall_summary.csv
  tables/ablation_comparison_matrix.csv
  tables/ablation_delta_vs_fullcore.csv
  tables/ablation_design_table.csv
  tables/ablation_rank_table.csv
  tables/ablation_tier_stratified_summary.csv
  tables/ablation_import_audit_table.csv
  tables/ablation_fullcore_source_audit.csv
  tables/ablation_fullcore_vs_internal_audit.csv
  tables/ablation_case_selection.csv

Key exported figures:
  FigA00_ablation_design_matrix.png
  FigA01_ablation_distributions.png
  FigA02_ablation_overall_summary.png
  FigA03_ablation_delta_vs_fullcore.png
  FigA04_ablation_tier_stratified_comparison.png
  FigA05_ablation_contam_recall_frontier.png
  FigA06_ablation_runtime_accuracy_tradeoff.png
  FigA07_ablation_top_case_deltas.png
  FigA08_ablation_representative_cases.png