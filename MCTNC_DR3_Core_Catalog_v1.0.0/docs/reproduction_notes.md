# Reproduction Notes

This release provides the frozen data products and diagnostic tables associated with the M-CTNC Gaia DR3 core-catalog manuscript.

## Authoritative release products

The authoritative products for version `v1.0.0` are the CSV files in `data/`.

The principal scientific product is:

```text
data/ApJS_Table2_MCTNC_Master_Catalog.csv
```

The diagnostic tables in `diagnostics/` support the validation and audit analyses reported in the manuscript. They are not replacements for the principal source catalog.

## Scripts

The `scripts/` directory contains the workflows used to construct and audit the release:

| Script | Role |
|---|---|
| `MCTNC.py` | Main M-CTNC production workflow. |
| `MCTNC-ablation.py` | Canonical ablation-suite workflow. |
| `MCTNC-baseline.py` | Shared-preprocessing external-baseline workflow. |
| `MCTNC-sensitivity.py` | Sensitivity / robustness workflow. |
| `MCTNC-sensitivity-1.py` | Evidence-packaging workflow with relative-path discovery. |
| `MCTNC-sensitivity-2.py` | Polished evidence table/figure workflow. |

The scripts are provided for transparency. Some scripts may require local path adjustments or command-line arguments before reuse, depending on the user's data layout and Python environment.

## Relative paths

The public release should use relative paths. In the archived repository, the canonical M-CTNC benchmark reference is:

```text
data/ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv
```

If any diagnostic audit table preserves an original internal Windows path, interpret it as run provenance from the internal environment. The corresponding public file is located under `data/` or `diagnostics/`.

## Reproducing the principal catalog

The frozen catalog in `data/` is the authoritative submitted product. Re-running M-CTNC may require access to the same Gaia DR3 local-field construction and reference inputs used by the manuscript workflow. The scripts document the analysis logic, but the released CSV products should be used as the stable versioned catalog.

## Reproducing diagnostic summaries

The consolidated diagnostic tables are intended to make the manuscript's validation results auditable without requiring a full re-run of every branch.

Examples:

- `diagnostics/photometric_independence/population_photometric_independence_summary.csv` supports the photometric-independence population summary.
- `diagnostics/robustness_reproducibility/Table01_population_robustness_ranking_polished.csv` supports the perturbation-sensitivity ranking.
- `diagnostics/ablation/tables/ablation_overall_summary.csv` supports the canonical ablation summary.
- `diagnostics/baseline/tables/baseline_overall_summary.csv` supports the shared-preprocessing external-baseline comparison.

## Notes on ablation ties

In `diagnostics/ablation/tables/ablation_delta_vs_fullcore.csv`, `tie_n` is defined using a material tolerance:

```text
|Delta F1| <= 0.01
```

This tolerance is used to avoid treating numerically tiny changes as material wins or losses.

## Spectroscopic cross-check

The manuscript includes a coverage-limited spectroscopic cross-check in Appendix D. The raw spectroscopic diagnostics are not part of the principal public release package because the cross-check is auxiliary and coverage-limited. The released data package focuses on the principal catalog products and the primary validation diagnostics for photometric independence, robustness, reproducibility, ablation, and baseline comparison.

## Software environment

The minimum recommended environment is Python 3.9 or newer with the packages listed in `requirements.txt`.
