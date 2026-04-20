Unified Part-Five baseline suite for M-CTNC
===========================================
Version: part5_baselines_v1_6_final_coremode_closure_corecsvfix
Clusters processed: 359
Methods: gmm_bic, heuristic_cut, hdbscan
M-CTNC comparison file: D:\HuaweiMoveData\Users\王浩\Desktop\课题组\DUCT-Clust\ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv
Benchmark policy: shared-preprocessing baseline suite
M-CTNC import policy: prefer core-mode rows over halo/exploration rows whenever mode metadata are available
Heuristic center refinement enabled: False
HDBSCAN center refinement enabled: True
M-CTNC mode filter policy: explicit_core_mode_filter
M-CTNC duplicate policy: none

Key exported tables:
  tables/baseline_cluster_results.csv
  tables/baseline_overall_summary.csv
  tables/comparison_matrix.csv
  tables/mctnc_vs_baseline_winloss.csv
  tables/fairness_audit_table.csv
  tables/mctnc_import_audit_table.csv
  tables/representative_case_selection.csv

Key exported figures:
  Fig00_fairness_audit_matrix.png
  Fig01_baseline_distributions.png
  Fig02_overall_summary.png
  Fig03_tier_stratified_comparison.png
  Fig04_contam_recall_frontier.png
  Fig05_runtime_accuracy_tradeoff.png
  Fig06_mctnc_winloss_vs_baselines.png
  Fig07_top_case_deltas.png
  Fig08_representative_baseline_cases.png
