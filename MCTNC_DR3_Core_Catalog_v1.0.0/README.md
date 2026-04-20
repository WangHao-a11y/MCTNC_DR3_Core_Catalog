# MCTNC_DR3_Core_Catalog

Gaia DR3 core-membership catalog for 359 benchmark open clusters produced with Multiscale Common Tightest Neighbors Consensus (M-CTNC).

This repository contains the public data products and diagnostic tables associated with the manuscript:

**A Gaia DR3 Core-Membership Catalog for 359 Open Clusters from M-CTNC**

## Authors

- Hao Wang
- Luosheng Wen

Correspondence: Luosheng Wen, `wls@cqu.edu.cn`

## Version

`v1.0.0`

## Principal data product

The default scientific product is:

```text
data/ApJS_Table2_MCTNC_Master_Catalog.csv
```

This file contains the conservative `CORE_BENCHMARK` astrometric core-member source catalog for 359 benchmark open clusters.

The catalog is designed as a low-contamination, astrometry-defined core sample. M-CTNC uses Gaia DR3 sky position, proper motion, and parallax for membership inference. Gaia photometry is retained for post hoc validation and downstream reuse, but it is not used in candidate construction, graph formation, support counting, final arbitration, or release definition.

## Operating modes

Two operating modes are provided:

- `CORE_BENCHMARK`: the principal conservative astrometric core release.
- `HALO_EXPLORATION`: an auxiliary halo-oriented extraction for boundary-sensitive inspection.

The `HALO_EXPLORATION` source layer is not a replacement for the principal core catalog and should not be interpreted as a strict nested superset of the `CORE_BENCHMARK` catalog.

## Repository structure

```text
MCTNC_DR3_Core_Catalog_v1.0.0/
|
├── README.md
├── LICENSE
├── CITATION.cff
├── VERSION
├── requirements.txt
├── .gitignore
|
├── data/
├── diagnostics/
│   ├── photometric_independence/
│   ├── robustness_reproducibility/
│   ├── ablation/
│   └── baseline/
├── reference/
├── scripts/
└── docs/
```

## `data/`

The `data/` directory contains the formal release products:

```text
data/ApJS_Table1_Cluster_Properties.csv
data/ApJS_Table2_MCTNC_Master_Catalog.csv
data/ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv
data/ApJS_TableA_Full_Benchmark_HALO_EXPLORATION.csv
data/ApJS_TableB_Anomaly_Audit_CORE_BENCHMARK.csv
data/ApJS_TableB_Anomaly_Audit_HALO_EXPLORATION.csv
data/ApJS_TableC_Master_Catalog_CORE_BENCHMARK.csv
data/ApJS_TableC_Master_Catalog_HALO_EXPLORATION.csv
```

Use `ApJS_Table2_MCTNC_Master_Catalog.csv` for the principal released member list.

## `diagnostics/`

The `diagnostics/` directory contains validation and audit products.

- `photometric_independence/`: population-level Gaia photometric audit, apparent-magnitude distribution tests, relative magnitude-bin recall, and Appendix C representative-case metadata.
- `robustness_reproducibility/`: production-consistent rerun, bounded perturbation, and center-refinement evidence products.
- `ablation/`: canonical `CORE_BENCHMARK` ablation suite.
- `baseline/`: shared-preprocessing external comparison against GMM+BIC, heuristic cut, and HDBSCAN baselines.

These diagnostic files support the validation results reported in the manuscript. They are not alternative membership catalogs.

## `reference/`

The `reference/` directory contains the reference input tables used in benchmark construction:

```text
reference/ocfinder_table1.csv
reference/ocfinder_table2.csv
```

These files correspond to the UBC-based open-cluster reference layer used by the manuscript and aligned with the Castro-Ginard et al. Gaia open-cluster sample. They are included to document the benchmark provenance and should not be interpreted as M-CTNC release products.

## `scripts/`

The `scripts/` directory contains the analysis workflows used to construct and audit the release:

```text
scripts/MCTNC.py
scripts/MCTNC-ablation.py
scripts/MCTNC-baseline.py
scripts/MCTNC-sensitivity.py
scripts/MCTNC-sensitivity-1.py
scripts/MCTNC-sensitivity-2.py
```

The CSV products in `data/` are the authoritative frozen release products for version `v1.0.0`. The scripts are provided for transparency and reproducibility support.

## Documentation

Additional documentation is available in `docs/`:

```text
docs/file_manifest.md
docs/column_dictionary.md
docs/usage_guide.md
docs/reproduction_notes.md
docs/manuscript_table_mapping.md
```

Start with `docs/usage_guide.md` for examples of reading the source catalog and diagnostic tables.

## Reading the catalog

Use Python and read `source_id` as a string:

```python
import pandas as pd

cat = pd.read_csv(
    "data/ApJS_Table2_MCTNC_Master_Catalog.csv",
    dtype={"source_id": str}
)

members = cat[cat["Cluster_Name"] == "UBC1001"]
print(members.head())
```

Avoid resaving the catalog from spreadsheet software unless `source_id` is explicitly treated as text.

## Citation

If you use this catalog or diagnostic package, cite the associated manuscript:

Wang, H., & Wen, L. A Gaia DR3 Core-Membership Catalog for 359 Open Clusters from M-CTNC.

If you use a versioned archive of this repository, also cite the persistent identifier associated with that archive.

## License

Data products are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Source code is released under the MIT License.

See `LICENSE` for details.
