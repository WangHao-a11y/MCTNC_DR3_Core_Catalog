# Usage Guide

This guide gives practical examples for reading and using the M-CTNC Gaia DR3 core-catalog release.

## 1. Which file should I use?

For most scientific use, start with the principal source catalog:

```text
data/ApJS_Table2_MCTNC_Master_Catalog.csv
```

This file contains the conservative `CORE_BENCHMARK` astrometric core members. It is the default released member list.

Use the auxiliary source catalogs only if you need to compare operating modes:

```text
data/ApJS_TableC_Master_Catalog_CORE_BENCHMARK.csv
data/ApJS_TableC_Master_Catalog_HALO_EXPLORATION.csv
```

The `HALO_EXPLORATION` layer is not a replacement for the principal core catalog. It is an auxiliary boundary-sensitive extraction.

## 2. Read the principal source catalog

```python
import pandas as pd

cat = pd.read_csv(
    "data/ApJS_Table2_MCTNC_Master_Catalog.csv",
    dtype={"source_id": str}
)

print(cat.head())
print(cat.shape)
```

Always read `source_id` as a string to avoid precision loss.

## 3. Select one cluster

```python
cluster_name = "UBC1001"

members = cat.loc[cat["Cluster_Name"] == cluster_name].copy()
print(members.head())
print(f"{cluster_name}: {len(members)} released core members")
```

## 4. Save a cluster-specific member list

```python
members.to_csv("UBC1001_core_members.csv", index=False)
```

## 5. Join the source catalog with the cluster-level summary

```python
summary = pd.read_csv("data/ApJS_Table1_Cluster_Properties.csv")

merged = cat.merge(
    summary,
    left_on="Cluster_Name",
    right_on="Cluster",
    how="left"
)
```

## 6. Inspect benchmark behavior

The cluster-level benchmark summary is available for the two operating modes:

```python
core_bench = pd.read_csv("data/ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv")
halo_bench = pd.read_csv("data/ApJS_TableA_Full_Benchmark_HALO_EXPLORATION.csv")

print(core_bench[["cluster", "precision", "recall", "f1", "contam"]].describe())
print(halo_bench[["cluster", "precision", "recall", "f1", "contam"]].describe())
```

## 7. Compare CORE_BENCHMARK and HALO_EXPLORATION source layers

```python
core = pd.read_csv(
    "data/ApJS_TableC_Master_Catalog_CORE_BENCHMARK.csv",
    dtype={"source_id": str}
)

halo = pd.read_csv(
    "data/ApJS_TableC_Master_Catalog_HALO_EXPLORATION.csv",
    dtype={"source_id": str}
)

core_pairs = set(zip(core["Cluster_Name"], core["source_id"]))
halo_pairs = set(zip(halo["Cluster_Name"], halo["source_id"]))

shared = core_pairs & halo_pairs
core_only = core_pairs - halo_pairs
halo_only = halo_pairs - core_pairs

print("shared:", len(shared))
print("core only:", len(core_only))
print("halo only:", len(halo_only))
```

The two source layers are close but not strictly nested because they use different support thresholds and final arbitration behavior.

## 8. Inspect anomaly-audit cases

```python
audit = pd.read_csv("data/ApJS_TableB_Anomaly_Audit_CORE_BENCHMARK.csv")

cols = [
    "cluster",
    "tier",
    "n_catalog_only",
    "n_pred_only",
    "faint_flag",
    "halo_flag",
    "audit_reasons"
]

print(audit[cols].head())
```

The anomaly-audit table is not a second membership definition. It records how to interpret non-ideal benchmark cases.

## 9. Use photometric-independence diagnostics

```python
phot = pd.read_csv("diagnostics/photometric_independence/population_photometric_independence_per_cluster.csv")

outliers = pd.read_csv("diagnostics/photometric_independence/population_photometric_outlier_clusters.csv")

print(phot[["Cluster", "ks_pvalue", "delta_g_median", "delta_g_q90"]].head())
print(outliers)
```

These diagnostics use Gaia photometry only after membership inference has been completed.

## 10. Use robustness and reproducibility diagnostics

```python
rob = pd.read_csv("diagnostics/robustness_reproducibility/Table01_population_robustness_ranking_polished.csv")
print(rob[["parameter", "median_abs_delta_f1", "p90_abs_delta_f1", "frac_le_0.05_num"]])
```

## 11. Use ablation diagnostics

```python
abl = pd.read_csv("diagnostics/ablation/tables/ablation_overall_summary.csv")
print(abl[["variant", "mean_f1", "mean_contam", "success_frac_f1_ge_0_9"]])
```

## 12. Use baseline-comparison diagnostics

```python
base = pd.read_csv("diagnostics/baseline/tables/baseline_overall_summary.csv")
print(base[["method", "mean_f1", "mean_precision", "mean_recall", "mean_contam"]])
```

## 13. Recommended workflow

For a new analysis:

1. Start with `data/ApJS_Table2_MCTNC_Master_Catalog.csv`.
2. Use `data/ApJS_Table1_Cluster_Properties.csv` for cluster-level context.
3. Check `data/ApJS_TableB_Anomaly_Audit_CORE_BENCHMARK.csv` for flagged systems.
4. Use `diagnostics/photometric_independence/` only if photometric-selection behavior is relevant.
5. Use `diagnostics/ablation/`, `diagnostics/baseline/`, and `diagnostics/robustness_reproducibility/` for methodological validation and manuscript reproduction.
