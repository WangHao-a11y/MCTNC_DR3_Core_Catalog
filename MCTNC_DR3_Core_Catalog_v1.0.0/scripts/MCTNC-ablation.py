from __future__ import annotations

import argparse
import math
import re
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")


# ============================================================================
# Unified configuration
# ============================================================================
PIPELINE_NAME = "MCTNC_ABLATION_SUITE"
VERSION = "mctnc_ablation_v1_2_corebenchmark_canonical_release"
CACHE_PIPELINE_SIGNATURE = "MCTNC_ABLATION_RELEASE_v1_2_COREBENCHMARK_CANONICAL"
DEFAULT_CANONICAL_MCTNC_CORE_REFERENCE = r"D:\HuaweiMoveData\Users\王浩\Desktop\课题组\DUCT-Clust\ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv"
DEFAULT_CANONICAL_MCTNC_CORE_FILENAME = "ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv"

# Canonical full-core policy
USE_IMPORTED_CORE_AS_CANONICAL_FULLCORE = True
REJECT_HALO_EXPLORATION_REFERENCES = True

# Quality control
DEFAULT_RUWE_MAX = 1.6

# Shared astrometric normalization
W_POS = 1.00
W_PLX = 1.00
W_PM = 1.00
FLOOR_PLX = 0.10
FLOOR_PM = 0.20

# Candidate protocol
CAP_MIN, CAP_MAX = 300, 1200
CAND_SCORE_RATIO, CAND_POS_RATIO, CAND_KIN_RATIO = 0.60, 0.25, 0.15

# Center refinement
CENTER_REFINE_TOP_R = 600
CENTER_REFINE_K = 24
CENTER_SHIFT_BLEND = 0.70
CENTER_SHIFT_LIMIT_ARCMIN = 20.0

# Core engine parameters
FULL_K_SET = [12, 18, 24, 32, 48]
FIXED_K = 24
SUPPORT_TAU_SET = [0.45, 0.55, 0.65, 0.75]
TOPFRACTION_SET = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
SEED_TOP_SCORE = 24
ANCHOR_TOPK = 20
MIN_MEMBERS = 5
TOP_CASES = 12
FIG_DPI = 220
RANK_TIE_TOL = 1e-10
DELTA_TIE_TOL = 0.01

TIER_ORDER = [
    "Perfect Match",
    "Tier 1 (Near-perfect)",
    "Tier 2 (Conservative Core)",
    "Tier 3 (Topological Over-expansion)",
    "Tier 4 (Borderline)",
]

MCTNC_REQUIRE_CORE_MODE = True


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass(frozen=True)
class AblationSpec:
    name: str
    title: str
    description: str
    quality_cut: bool = True
    candidate_protocol: bool = True
    anchor_prior: bool = True
    adaptive_k: bool = True
    support_gate: bool = True
    objective_regularization: bool = True
    center_refinement: bool = True


@dataclass
class RunResult:
    cluster: str
    variant: str
    n_cone: int
    n_raw_cone: int
    n_quality_rejected: int
    n_true_in_cone: int
    n_pred: int
    precision: float
    recall: float
    f1: float
    contam: float
    runtime_s: float
    center_mode: str
    center_shift_arcmin: float
    objective: float
    tag: str
    chosen_k: float = np.nan
    chosen_tau: float = np.nan
    mean_support: float = np.nan
    median_radius: float = np.nan


# ============================================================================
# Variant design
# ============================================================================
ABLATION_SPECS: List[AblationSpec] = [
    AblationSpec(
        name="full_core",
        title="Full core model",
        description="Full production-aligned core configuration.",
    ),
    AblationSpec(
        name="no_quality_cut",
        title="Without quality cut",
        description="Removes RUWE-based upstream quality control.",
        quality_cut=False,
    ),
    AblationSpec(
        name="no_candidate_protocol",
        title="Without candidate protocol",
        description="Runs the core objective on the full cone rather than the capped candidate subset.",
        candidate_protocol=False,
    ),
    AblationSpec(
        name="no_anchor_prior",
        title="Without anchor prior",
        description="Builds seeds from score ranking alone, without the anchor track.",
        anchor_prior=False,
    ),
    AblationSpec(
        name="fixed_single_k",
        title="Without adaptive k-set",
        description="Replaces the adaptive k-set with a single fixed neighborhood scale.",
        adaptive_k=False,
    ),
    AblationSpec(
        name="no_support_gate",
        title="Without support gate",
        description="Removes the explicit support-threshold gate and relies on ranked support slices.",
        support_gate=False,
    ),
    AblationSpec(
        name="no_center_refinement",
        title="Without center refinement",
        description="Locks the solution to the catalog center only.",
        center_refinement=False,
    ),
    AblationSpec(
        name="no_objective_regularization",
        title="Without objective regularization",
        description="Chooses the solution by mean support alone, without size/cohesion regularization.",
        objective_regularization=False,
    ),
]
ABLATION_ORDER = [spec.name for spec in ABLATION_SPECS]
AB_SPEC_MAP = {spec.name: spec for spec in ABLATION_SPECS}


# ============================================================================
# Utilities
# ============================================================================
def _now() -> float:
    return time.perf_counter()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def get_column_safely(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols_lower = [str(c).lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols_lower:
            return df.columns[cols_lower.index(cand.lower())]
    return None


def find_first_col_strict(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    res = get_column_safely(df, candidates)
    if res is None:
        raise KeyError(f"Missing required column among: {list(candidates)}")
    return res


def robust_tree_query(tree: cKDTree, X: np.ndarray, k: int):
    try:
        return tree.query(X, k=k, workers=-1)
    except TypeError:
        return tree.query(X, k=k)


def wrap_ra_coordinate(ra_deg: np.ndarray, ra0_deg: float) -> np.ndarray:
    return (ra_deg - ra0_deg + 180.0) % 360.0 - 180.0


def compute_tangent_plane(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = wrap_ra_coordinate(ra_deg, ra0_deg) * math.cos(math.radians(dec0_deg))
    y = dec_deg - dec0_deg
    return x.astype(np.float32), y.astype(np.float32)


def robust_mad_scale(x: np.ndarray, floor: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float(floor)
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    if not np.isfinite(s):
        return float(floor)
    return float(max(s, floor))


def infer_cluster_name(path: Path) -> str:
    s = path.stem.strip()
    if s.lower().startswith("gaia_cone_"):
        s = s[len("gaia_cone_"):]
    m = re.match(r"([A-Za-z]+\d+)", s)
    return m.group(1) if m else s


def discover_cone_files(base_dir: Path, data_dir: Path, user_cone_dir: Optional[Path] = None) -> List[Path]:
    candidates: List[Path] = []
    if user_cone_dir is not None:
        candidates.append(user_cone_dir)
    candidates.extend([data_dir, base_dir, base_dir / "cones", base_dir / "cone", base_dir / "data-gaia", base_dir / "gaia"])

    dirs: List[Path] = []
    seen = set()
    for d in candidates:
        try:
            d = d.resolve()
        except Exception:
            pass
        if d.exists() and d.is_dir() and str(d) not in seen:
            seen.add(str(d))
            dirs.append(d)

    files: List[Path] = []
    for d in dirs:
        files.extend(list(d.glob("gaia_cone_*.csv")))
        files.extend(list(d.glob("gaia_cone_*.CSV")))
    if not files:
        for d in dirs:
            files.extend([
                p for p in d.rglob("*")
                if p.is_file() and p.name.lower().startswith("gaia_cone_") and p.name.lower().endswith(".csv")
            ])
    return sorted({p.resolve() for p in files})


def read_cone_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    rename_map = {}
    lower = {c.lower(): c for c in df.columns}
    aliases = {
        "source_id": ["source_id", "gaiaedr3", "SOURCE_ID"],
        "ra": ["ra", "RA_ICRS", "RA"],
        "dec": ["dec", "DE_ICRS", "DE"],
        "parallax": ["parallax", "plx"],
        "pmra": ["pmra", "pmRA"],
        "pmdec": ["pmdec", "pmDE"],
        "parallax_error": ["parallax_error", "e_plx"],
        "pmra_error": ["pmra_error", "e_pmra"],
        "pmdec_error": ["pmdec_error", "e_pmdec"],
    }
    for std, alts in aliases.items():
        for alt in alts:
            if alt.lower() in lower:
                rename_map[lower[alt.lower()]] = std
                break
    return df.rename(columns=rename_map)


def load_benchmark_tables(data_dir: Path) -> Tuple[Dict[str, pd.Series], Dict[str, np.ndarray]]:
    t1_path = data_dir / "ocfinder_table1.csv"
    t2_path = data_dir / "ocfinder_table2.csv"
    if not t1_path.exists() or not t2_path.exists():
        raise FileNotFoundError("ocfinder_table1.csv / ocfinder_table2.csv not found in data_dir.")

    table1 = pd.read_csv(t1_path)
    table2 = pd.read_csv(t2_path)

    c1_cl = find_first_col_strict(table1, ["Cluster", "cluster", "Name", "name"])
    t1 = table1.copy()
    t1["cluster"] = t1[c1_cl].astype(str).str.strip()

    for c, alts in [
        ("ra0", ["RA_ICRS", "ra", "RA"]),
        ("dec0", ["DE_ICRS", "dec", "DE"]),
        ("radius", ["r_deg", "radius", "r", "RADIUS"]),
        ("plx0", ["plx", "parallax", "plx0"]),
        ("pmra0", ["pmra", "pmRA", "pmra0"]),
        ("pmdec0", ["pmdec", "pmDE", "pmdec0"]),
    ]:
        col = get_column_safely(t1, alts)
        t1[c] = pd.to_numeric(t1[col], errors="coerce") if col else np.nan

    t1_idx = {str(r["cluster"]): r for _, r in t1.iterrows()}

    c2_cl = find_first_col_strict(table2, ["Cluster", "cluster", "Name", "name"])
    c2_sid = get_column_safely(table2, ["source_id", "GaiaEDR3", "gaiaedr3", "SOURCE_ID"])
    if c2_sid is None:
        raise KeyError("Table 2 missing source_id / GaiaEDR3.")

    t2 = table2.copy()
    t2["cluster"] = t2[c2_cl].astype(str).str.strip()
    sid2 = pd.to_numeric(t2[c2_sid], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    true_map = {str(cl): np.unique(sid2[g.index][sid2[g.index] > 0]) for cl, g in t2.groupby("cluster")}
    return t1_idx, true_map


# ============================================================================
# Optional imported M-CTNC reference
# ============================================================================
def is_halo_exploration_name(path: Path) -> bool:
    name = path.name.lower()
    return ("halo" in name) or ("explor" in name)


def resolve_preferred_mctnc_core_reference(base_dir: Path, data_dir: Path, user_path: Optional[Path] = None) -> Optional[Path]:
    if user_path is not None:
        return user_path.resolve()

    preferred = [
        Path(DEFAULT_CANONICAL_MCTNC_CORE_REFERENCE),
        base_dir / DEFAULT_CANONICAL_MCTNC_CORE_FILENAME,
        data_dir / DEFAULT_CANONICAL_MCTNC_CORE_FILENAME,
    ]
    for fp in preferred:
        try:
            if fp.exists() and fp.is_file():
                return fp.resolve()
        except Exception:
            continue
    return None


def locate_mctnc_candidates(data_dir: Path, user_path: Optional[Path] = None) -> List[Path]:
    cands: List[Path] = []
    if user_path is not None and user_path.exists() and user_path.is_file():
        cands.append(user_path.resolve())

    patterns = [
        "*mctnc*core*benchmark*.csv",
        "*MCTNC*core*benchmark*.csv",
        "*mctnc*core*.csv",
        "*MCTNC*core*.csv",
        "*results*core*.csv",
        "*benchmark*core*.csv",
        "*M-CTNC*.csv",
        "*mctnc*.csv",
    ]
    for pat in patterns:
        cands.extend([p.resolve() for p in data_dir.glob(pat) if p.is_file()])
        cands.extend([p.resolve() for p in data_dir.rglob(pat) if p.is_file()])

    uniq = []
    seen = set()
    for p in cands:
        if REJECT_HALO_EXPLORATION_REFERENCES and is_halo_exploration_name(p):
            continue
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)

    def sort_key(p: Path):
        name = p.name.lower()
        score = 0
        if "mctnc" in name or "m-ctnc" in name:
            score += 10
        if "core" in name:
            score += 8
        if "benchmark" in name:
            score += 6
        if "release" in name or "final" in name:
            score += 2
        if "halo" in name or "explor" in name:
            score -= 100
        return (-score, len(str(p)), str(p))

    uniq.sort(key=sort_key)
    return uniq


def normalize_mctnc_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty M-CTNC dataframe.")

    if "cluster" not in df.columns:
        c = get_column_safely(df, ["Cluster", "cluster", "name", "Name"])
        if c is None:
            raise KeyError("M-CTNC results csv lacks cluster column.")
        df["cluster"] = df[c].astype(str).str.strip()

    rename = {}
    for std, alts in {
        "precision": ["precision", "prec", "P"],
        "recall": ["recall", "rec", "R"],
        "f1": ["f1", "F1"],
        "runtime_s": ["runtime_s", "time_s", "runtime"],
        "contam": ["contam", "contamination"],
        "tier": ["tier", "performance_tier", "Tier"],
        "mctnc_mode": [
            "mctnc_mode", "mode", "run_mode", "profile", "variant", "membership_mode",
            "pipeline_mode", "result_mode", "solution_mode",
        ],
    }.items():
        col = get_column_safely(df, alts)
        if col and col != std:
            rename[col] = std
    df = df.rename(columns=rename)

    needed = ["cluster", "precision", "recall", "f1"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise KeyError(f"M-CTNC results csv missing columns: {miss}")

    if "runtime_s" not in df.columns:
        df["runtime_s"] = np.nan
    if "contam" not in df.columns:
        df["contam"] = 1.0 - pd.to_numeric(df["precision"], errors="coerce")

    keep = ["cluster", "precision", "recall", "f1", "runtime_s", "contam"]
    if "tier" in df.columns:
        keep.append("tier")
    if "mctnc_mode" in df.columns:
        keep.append("mctnc_mode")

    out = df[keep].copy()
    out["cluster"] = out["cluster"].astype(str).str.strip()
    for c in ["precision", "recall", "f1", "runtime_s", "contam"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "mctnc_mode" in out.columns:
        out["mctnc_mode"] = out["mctnc_mode"].astype(str).str.strip()
    out = out.dropna(subset=["cluster", "precision", "recall", "f1"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("M-CTNC dataframe becomes empty after cleaning.")
    return out


def filter_mctnc_core_mode(df: pd.DataFrame, source_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = df.copy()
    source_name = str(source_path.name) if source_path is not None else ""
    meta: Dict[str, Any] = {
        "mctnc_source": str(source_path) if source_path is not None else "",
        "mctnc_source_name": source_name,
        "source_name_policy": "non_halo_filename" if source_path is not None and not is_halo_exploration_name(source_path) else "unknown",
        "mctnc_mode_column": "mctnc_mode" if "mctnc_mode" in work.columns else "",
        "core_mode_required": int(MCTNC_REQUIRE_CORE_MODE),
        "raw_rows": int(len(work)),
        "kept_rows": int(len(work)),
        "raw_cluster_rows": int(work["cluster"].nunique()) if "cluster" in work.columns else 0,
        "kept_cluster_rows": int(work["cluster"].nunique()) if "cluster" in work.columns else 0,
        "core_rows_detected": 0,
        "halo_rows_detected": 0,
        "filter_policy": "no_mode_column",
        "duplicate_policy": "none",
        "assumed_core_benchmark": 0,
    }

    if source_path is not None and REJECT_HALO_EXPLORATION_REFERENCES and is_halo_exploration_name(source_path):
        raise ValueError(f"Rejected halo/exploration-style M-CTNC reference: {source_path}")

    if "mctnc_mode" in work.columns:
        mode = work["mctnc_mode"].astype(str).str.strip().str.lower()
        core_mask = mode.str.contains(r"(^|[^a-z])(core|locked[_\- ]?core|core[_\- ]?mode|core[_\- ]?only|core_benchmark)([^a-z]|$)", regex=True, na=False)
        halo_mask = mode.str.contains(r"halo|explor", regex=True, na=False)
        meta["core_rows_detected"] = int(core_mask.sum())
        meta["halo_rows_detected"] = int(halo_mask.sum())
        if MCTNC_REQUIRE_CORE_MODE and int(core_mask.sum()) > 0:
            work = work.loc[core_mask].copy()
            meta["filter_policy"] = "explicit_core_mode_filter"
            meta["assumed_core_benchmark"] = 1
        elif int(core_mask.sum()) == 0 and int((~halo_mask).sum()) > 0 and int(halo_mask.sum()) > 0:
            work = work.loc[~halo_mask].copy()
            meta["filter_policy"] = "excluded_halo_exploration_rows"
        else:
            meta["filter_policy"] = "mode_column_present_but_no_core_subset"
    else:
        meta["assumed_core_benchmark"] = 1 if (source_path is not None and not is_halo_exploration_name(source_path)) else 0
        meta["filter_policy"] = "no_mode_column_nonhalo_filename_assumed_core" if meta["assumed_core_benchmark"] else "no_mode_column"

    if work["cluster"].duplicated().any():
        tmp = work.copy()
        if "mctnc_mode" in tmp.columns:
            mode = tmp["mctnc_mode"].astype(str).str.strip().str.lower()
            tmp["_mode_priority"] = 0
            tmp.loc[mode.str.fullmatch(r"core"), "_mode_priority"] = 4
            tmp.loc[mode.str.contains(r"locked[_\- ]?core|core[_\- ]?mode|core[_\- ]?only|core_benchmark", regex=True, na=False), "_mode_priority"] = 3
            tmp.loc[mode.str.contains(r"core", regex=True, na=False), "_mode_priority"] = 2
            tmp.loc[mode.str.contains(r"halo|explor", regex=True, na=False), "_mode_priority"] = -2
        else:
            tmp["_mode_priority"] = 0
        tmp["_runtime_key"] = pd.to_numeric(tmp["runtime_s"], errors="coerce").fillna(np.inf)
        tmp["_row_order"] = np.arange(len(tmp))
        tmp = (
            tmp.sort_values(["cluster", "_mode_priority", "_runtime_key", "_row_order"], ascending=[True, False, True, True])
            .drop_duplicates("cluster", keep="first")
            .drop(columns=["_mode_priority", "_runtime_key", "_row_order"])
            .reset_index(drop=True)
        )
        work = tmp
        meta["duplicate_policy"] = "cluster_first_after_core_priority_then_runtime"

    meta["kept_rows"] = int(len(work))
    meta["kept_cluster_rows"] = int(work["cluster"].nunique())
    return work, meta


def try_read_mctnc_csv(fp: Path) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    try:
        if (not fp.exists()) or (not fp.is_file()) or fp.stat().st_size == 0:
            return None
        if REJECT_HALO_EXPLORATION_REFERENCES and is_halo_exploration_name(fp):
            return None
        df = pd.read_csv(fp)
        df = normalize_mctnc_frame(df)
        df, meta = filter_mctnc_core_mode(df, fp)
        return df, meta
    except (EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError, KeyError):
        return None
    except Exception:
        return None


def read_mctnc_results(data_dir: Path, user_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Optional[Path], Dict[str, Any]]:
    candidates = locate_mctnc_candidates(data_dir, user_path)
    for fp in candidates:
        payload = try_read_mctnc_csv(fp)
        if payload is not None:
            df, meta = payload
            if df is not None and not df.empty:
                return df, fp, meta
    return None, None, {
        "mctnc_source": "",
        "mctnc_source_name": "",
        "source_name_policy": "not_found",
        "mctnc_mode_column": "",
        "core_mode_required": int(MCTNC_REQUIRE_CORE_MODE),
        "raw_rows": 0,
        "kept_rows": 0,
        "raw_cluster_rows": 0,
        "kept_cluster_rows": 0,
        "core_rows_detected": 0,
        "halo_rows_detected": 0,
        "filter_policy": "not_found",
        "duplicate_policy": "none",
        "assumed_core_benchmark": 0,
    }


def build_canonical_fullcore_from_import(mctnc_ref: pd.DataFrame) -> pd.DataFrame:
    cols = ["cluster", "precision", "recall", "f1", "runtime_s", "contam"]
    g = mctnc_ref[cols].copy()
    g["variant"] = "full_core"
    g["n_cone"] = np.nan
    g["n_raw_cone"] = np.nan
    g["n_quality_rejected"] = np.nan
    g["n_true_in_cone"] = np.nan
    g["n_pred"] = np.nan
    g["center_mode"] = "imported_core_reference"
    g["center_shift_arcmin"] = 0.0
    g["objective"] = np.nan
    g["tag"] = "IMPORTED_CORE_BENCHMARK_REFERENCE"
    g["chosen_k"] = np.nan
    g["chosen_tau"] = np.nan
    g["mean_support"] = np.nan
    g["median_radius"] = np.nan
    ordered = [
        "cluster", "variant", "n_cone", "n_raw_cone", "n_quality_rejected", "n_true_in_cone", "n_pred",
        "precision", "recall", "f1", "contam", "runtime_s", "center_mode", "center_shift_arcmin",
        "objective", "tag", "chosen_k", "chosen_tau", "mean_support", "median_radius",
    ]
    return g[ordered].sort_values("cluster").reset_index(drop=True)


def build_fullcore_source_audit(mctnc_ref_path: Optional[Path], mctnc_meta: Dict[str, Any], canonical_source: str, internal_diag_enabled: bool) -> pd.DataFrame:
    return pd.DataFrame([{
        "canonical_full_core_source": canonical_source,
        "use_imported_core_as_canonical_fullcore": int(USE_IMPORTED_CORE_AS_CANONICAL_FULLCORE),
        "imported_reference_found": int(mctnc_ref_path is not None),
        "mctnc_source": str(mctnc_ref_path) if mctnc_ref_path is not None else "",
        "mctnc_filter_policy": mctnc_meta.get("filter_policy", "not_found"),
        "mctnc_source_name_policy": mctnc_meta.get("source_name_policy", "not_found"),
        "mctnc_mode_column": mctnc_meta.get("mctnc_mode_column", ""),
        "mctnc_duplicate_policy": mctnc_meta.get("duplicate_policy", "none"),
        "assumed_core_benchmark": int(mctnc_meta.get("assumed_core_benchmark", 0)),
        "internal_full_core_diagnostic_enabled": int(internal_diag_enabled),
    }])


def compare_internal_fullcore_to_canonical(comp_internal: pd.DataFrame) -> pd.DataFrame:
    if "full_core_internal_f1" not in comp_internal.columns or "full_core_f1" not in comp_internal.columns:
        return pd.DataFrame()
    d = pd.to_numeric(comp_internal["full_core_internal_f1"], errors="coerce") - pd.to_numeric(comp_internal["full_core_f1"], errors="coerce")
    p = pd.to_numeric(comp_internal["full_core_internal_precision"], errors="coerce") - pd.to_numeric(comp_internal["full_core_precision"], errors="coerce")
    r = pd.to_numeric(comp_internal["full_core_internal_recall"], errors="coerce") - pd.to_numeric(comp_internal["full_core_recall"], errors="coerce")
    return pd.DataFrame([{
        "n_clusters": int(len(comp_internal)),
        "exact_f1_match_n": int((np.abs(d) <= 1e-12).sum()),
        "median_abs_delta_f1": float(np.nanmedian(np.abs(d))),
        "p90_abs_delta_f1": float(np.nanquantile(np.abs(d), 0.90)),
        "mean_delta_f1_internal_minus_canonical": float(np.nanmean(d)),
        "mean_delta_precision_internal_minus_canonical": float(np.nanmean(p)),
        "mean_delta_recall_internal_minus_canonical": float(np.nanmean(r)),
    }])

# ============================================================================
# Core preprocessing
# ============================================================================
def apply_quality_cuts(df: pd.DataFrame, enable: bool, ruwe_max: float) -> Tuple[pd.DataFrame, int]:
    if (not enable) or ("ruwe" not in df.columns):
        return df.reset_index(drop=True).copy(), 0
    ruwe = pd.to_numeric(df["ruwe"], errors="coerce").to_numpy(np.float64)
    keep = np.isfinite(ruwe) & (ruwe <= float(ruwe_max))
    n_rej = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True).copy(), n_rej


def preprocess_astrometry(
    df: pd.DataFrame,
    ra0: float,
    dec0: float,
    rdeg: float,
    plx0: Optional[float],
    pmra0: Optional[float],
    pmdec0: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(np.float64)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(np.float64)
    x_deg, y_deg = compute_tangent_plane(ra, dec, ra0, dec0)
    sig_pos = max(0.10, float(rdeg) if np.isfinite(rdeg) and rdeg > 0 else 1.0)

    plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(np.float32)
    pmra = pd.to_numeric(df["pmra"], errors="coerce").to_numpy(np.float32)
    pmdec = pd.to_numeric(df["pmdec"], errors="coerce").to_numpy(np.float32)

    plx0 = float(np.nanmedian(plx)) if plx0 is None or not np.isfinite(plx0) else float(plx0)
    pmra0 = float(np.nanmedian(pmra)) if pmra0 is None or not np.isfinite(pmra0) else float(pmra0)
    pmdec0 = float(np.nanmedian(pmdec)) if pmdec0 is None or not np.isfinite(pmdec0) else float(pmdec0)

    dplx = (plx - plx0).astype(np.float32, copy=False)
    dpmra = (pmra - pmra0).astype(np.float32, copy=False)
    dpmdec = (pmdec - pmdec0).astype(np.float32, copy=False)

    core_idx = np.argsort(np.hypot(x_deg, y_deg))[: min(600, max(80, int(0.10 * len(df))))]
    sig_plx_intr = robust_mad_scale(dplx[core_idx], floor=FLOOR_PLX)
    sig_pm_intr = robust_mad_scale(np.hypot(dpmra[core_idx], dpmdec[core_idx]), floor=FLOOR_PM)

    if "parallax_error" in df.columns:
        plx_err = pd.to_numeric(df["parallax_error"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        plx_err = np.zeros(len(df), dtype=np.float32)

    if "pmra_error" in df.columns:
        pmra_err = pd.to_numeric(df["pmra_error"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        pmra_err = np.zeros(len(df), dtype=np.float32)

    if "pmdec_error" in df.columns:
        pmdec_err = pd.to_numeric(df["pmdec_error"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        pmdec_err = np.zeros(len(df), dtype=np.float32)

    sig_plx = np.sqrt(sig_plx_intr ** 2 + np.maximum(plx_err, 0.0) ** 2).astype(np.float32)
    sig_pm = np.sqrt(sig_pm_intr ** 2 + 0.5 * (np.maximum(pmra_err, 0.0) ** 2 + np.maximum(pmdec_err, 0.0) ** 2)).astype(np.float32)

    zx = (x_deg / sig_pos).astype(np.float32, copy=False)
    zy = (y_deg / sig_pos).astype(np.float32, copy=False)
    zplx = (dplx / sig_plx).astype(np.float32, copy=False)
    zpmra = (dpmra / sig_pm).astype(np.float32, copy=False)
    zpmdec = (dpmdec / sig_pm).astype(np.float32, copy=False)

    pos2 = (zx * zx + zy * zy).astype(np.float32, copy=False)
    kin2 = (zplx * zplx + zpmra * zpmra + zpmdec * zpmdec).astype(np.float32, copy=False)
    score = (W_POS * pos2 + W_PLX * (zplx * zplx) + W_PM * (zpmra * zpmra + zpmdec * zpmdec)).astype(np.float32)
    X = np.column_stack([zx, zy, zplx, zpmra, zpmdec]).astype(np.float32, copy=False)

    aux = {
        "sig_pos_deg": float(sig_pos),
        "sig_plx_intr": float(sig_plx_intr),
        "sig_pm_intr": float(sig_pm_intr),
    }
    return score, X, pos2, kin2, aux


def sample_candidate_subset(score: np.ndarray, pos2: np.ndarray, kin2: np.ndarray) -> np.ndarray:
    n_total = len(score)
    cap = int(min(max(CAP_MIN, int(round(0.18 * n_total))), CAP_MAX, n_total))
    n_score = max(20, min(int(round(cap * CAND_SCORE_RATIO)), cap))
    n_pos = max(15, min(int(round(cap * CAND_POS_RATIO)), cap))
    n_kin = max(10, min(cap - n_score - n_pos, cap))
    cand = np.unique(
        np.concatenate([
            np.argsort(score)[:n_score],
            np.argsort(pos2)[:n_pos],
            np.argsort(kin2)[:n_kin],
        ]).astype(np.int32)
    )
    if len(cand) < cap:
        extra = np.argsort(score)[n_score:n_score + (cap - len(cand))].astype(np.int32)
        cand = np.unique(np.concatenate([cand, extra]))
    return cand.astype(np.int32, copy=False)


def refine_center_if_needed(df_cone: pd.DataFrame, ra0: float, dec0: float, enable: bool) -> List[Tuple[str, float, float, float]]:
    centers = [("center0", float(ra0), float(dec0), 0.0)]
    if not enable:
        return centers

    ra = pd.to_numeric(df_cone["ra"], errors="coerce").to_numpy(np.float64)
    dec = pd.to_numeric(df_cone["dec"], errors="coerce").to_numpy(np.float64)
    x0, y0 = compute_tangent_plane(ra, dec, ra0, dec0)
    top_idx = np.argsort(np.hypot(x0, y0))[: min(CENTER_REFINE_TOP_R, len(ra))]
    if len(top_idx) < max(32, CENTER_REFINE_K + 5):
        return centers

    P = np.column_stack([x0[top_idx], y0[top_idx]]).astype(np.float32, copy=False)
    tree = cKDTree(P)
    dist, _ = robust_tree_query(tree, P, k=min(CENTER_REFINE_K + 1, len(top_idx)))
    peak_idx = top_idx[int(np.argmin(dist[:, -1]))]
    ra_peak = float(ra[peak_idx])
    dec_peak = float(dec[peak_idx])
    dx, dy = compute_tangent_plane(np.array([ra_peak]), np.array([dec_peak]), ra0, dec0)
    shift_arcmin = float(np.hypot(dx[0], dy[0]) * 60.0)
    if shift_arcmin <= 0.0:
        return centers
    shift_arcmin = min(shift_arcmin, CENTER_SHIFT_LIMIT_ARCMIN)
    ra1 = float((1.0 - CENTER_SHIFT_BLEND) * ra0 + CENTER_SHIFT_BLEND * ra_peak)
    dec1 = float((1.0 - CENTER_SHIFT_BLEND) * dec0 + CENTER_SHIFT_BLEND * dec_peak)
    centers.append(("center1_refined", ra1, dec1, shift_arcmin))
    return centers


# ============================================================================
# Core engine
# ============================================================================
def robust_center_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) < 3:
        return np.nanmedian(X, axis=0).astype(np.float32), np.array([0.16, 0.16, 0.26, 0.30, 0.30], dtype=np.float32)
    center = np.nanmedian(X, axis=0).astype(np.float32)
    floor_vec = np.array([0.16, 0.16, 0.26, 0.30, 0.30], dtype=np.float32)
    scale = np.array([max(robust_mad_scale(X[:, i], float(floor_vec[i])), float(floor_vec[i])) for i in range(X.shape[1])], dtype=np.float32)
    return center, scale


def minmax_clip(x: np.ndarray, p_hi: float = 0.98) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = np.nanmin(finite)
    hi = np.nanquantile(finite, p_hi)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    z = (x - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def build_seed_indices(sc: np.ndarray, Xc: np.ndarray, anchor_prior: bool) -> np.ndarray:
    top_score = np.argsort(sc)[: min(SEED_TOP_SCORE, len(sc))].astype(np.int32)
    if not anchor_prior:
        return top_score
    kin_proxy = Xc[:, 2] ** 2 + Xc[:, 3] ** 2 + Xc[:, 4] ** 2
    top_anchor = np.argsort(kin_proxy)[: min(ANCHOR_TOPK, len(sc))].astype(np.int32)
    seed = np.unique(np.concatenate([top_score, top_anchor])).astype(np.int32)
    if len(seed) < min(MIN_MEMBERS, len(sc)):
        seed = top_score[: min(max(MIN_MEMBERS, 12), len(sc))]
    return seed


def compute_support_components(Xc: np.ndarray, sc: np.ndarray, seed_idx: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center, scale = robust_center_scale(Xc[seed_idx])
    z = (Xc - center[None, :]) / scale[None, :]
    radius = np.sqrt(np.sum(z * z, axis=1))
    affinity = np.exp(-0.5 * (radius ** 2 / Xc.shape[1])).astype(np.float32)

    tree = cKDTree(Xc)
    k_use = min(max(k + 1, 2), len(Xc))
    dist, _ = robust_tree_query(tree, Xc, k=k_use)
    kth = dist[:, -1] if getattr(dist, "ndim", 1) > 1 else dist
    density = 1.0 / np.maximum(kth, 1e-6)
    density = minmax_clip(density)

    score_bonus = 1.0 - minmax_clip(sc)
    support = (0.50 * affinity + 0.30 * density + 0.20 * score_bonus).astype(np.float32)
    return support, radius.astype(np.float32), affinity.astype(np.float32), density.astype(np.float32)


def proposal_masks_from_support(support: np.ndarray, support_gate: bool) -> List[Tuple[np.ndarray, float, str]]:
    masks: List[Tuple[np.ndarray, float, str]] = []
    if support_gate:
        for tau in SUPPORT_TAU_SET:
            m = support >= float(tau)
            masks.append((m.astype(bool, copy=False), float(tau), f"tau={tau:.2f}"))
        return masks

    order = np.argsort(support)[::-1]
    n = len(order)
    for frac in TOPFRACTION_SET:
        keep = max(MIN_MEMBERS, int(round(frac * n)))
        mask = np.zeros(n, dtype=bool)
        mask[order[:keep]] = True
        masks.append((mask, float(frac), f"topfrac={frac:.2f}"))
    return masks


def evaluate_mask(mask: np.ndarray, support: np.ndarray, score: np.ndarray, radius: np.ndarray, regularized: bool) -> float:
    idx = np.flatnonzero(mask)
    if idx.size < MIN_MEMBERS:
        return -1e9
    mean_support = float(np.mean(support[idx]))
    if not regularized:
        return mean_support
    size_term = math.log1p(idx.size) / math.log1p(len(mask))
    cohesion_term = float(np.mean(score[idx]))
    radius_term = float(np.median(radius[idx]))
    objective = mean_support + 0.16 * size_term - 0.10 * cohesion_term - 0.08 * radius_term
    return float(objective)


def solve_mctnc_core_variant(Xc: np.ndarray, sc: np.ndarray, spec: AblationSpec) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    seed_idx = build_seed_indices(sc, Xc, spec.anchor_prior)
    k_grid = FULL_K_SET if spec.adaptive_k else [FIXED_K]

    best_obj = -1e9
    best_mask = np.zeros(len(Xc), dtype=bool)
    best_pack: Dict[str, Any] = {
        "objective": -1e9,
        "tag": "EMPTY",
        "chosen_k": np.nan,
        "chosen_tau": np.nan,
        "mean_support": np.nan,
        "median_radius": np.nan,
    }
    best_support = np.zeros(len(Xc), dtype=np.float32)
    best_radius = np.full(len(Xc), np.nan, dtype=np.float32)

    for k in k_grid:
        support, radius, _, _ = compute_support_components(Xc, sc, seed_idx, int(k))
        for mask, tau_value, tau_tag in proposal_masks_from_support(support, spec.support_gate):
            obj = evaluate_mask(mask, support, sc, radius, spec.objective_regularization)
            if obj > best_obj:
                best_obj = obj
                best_mask = mask.copy()
                best_support = support.copy()
                best_radius = radius.copy()
                best_pack = {
                    "objective": float(obj),
                    "tag": f"CORE|k={int(k)}|{tau_tag}",
                    "chosen_k": float(k),
                    "chosen_tau": float(tau_value),
                    "mean_support": float(np.mean(support[np.flatnonzero(mask)])) if int(mask.sum()) >= MIN_MEMBERS else np.nan,
                    "median_radius": float(np.median(radius[np.flatnonzero(mask)])) if int(mask.sum()) >= MIN_MEMBERS else np.nan,
                }

    if int(best_mask.sum()) < MIN_MEMBERS:
        seed_keep = np.zeros(len(Xc), dtype=bool)
        seed_keep[seed_idx[: min(len(seed_idx), max(MIN_MEMBERS, 12))]] = True
        best_mask = seed_keep
        best_support, best_radius, _, _ = compute_support_components(Xc, sc, seed_idx, FIXED_K)
        best_pack = {
            "objective": -1e6,
            "tag": "FALLBACK_SEED_SET",
            "chosen_k": float(FIXED_K),
            "chosen_tau": np.nan,
            "mean_support": float(np.mean(best_support[np.flatnonzero(best_mask)])),
            "median_radius": float(np.median(best_radius[np.flatnonzero(best_mask)])),
        }
    return best_mask.astype(bool, copy=False), best_pack, best_support, best_radius


# ============================================================================
# Single-cluster orchestration
# ============================================================================
def metric_from_sets(pred_set: set, true_set: set) -> Tuple[float, float, float]:
    tp = sum(x in true_set for x in pred_set)
    fp = len(pred_set) - tp
    fn = len(true_set) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def process_single_cluster(
    spec: AblationSpec,
    cluster: str,
    df_cone_raw: pd.DataFrame,
    row1: pd.Series,
    true_ids: np.ndarray,
) -> Tuple[RunResult, Dict[str, Any]]:
    t0 = _now()
    df_cone, n_rej = apply_quality_cuts(df_cone_raw, spec.quality_cut, DEFAULT_RUWE_MAX)

    ra0 = _safe_float(row1.get("ra0", row1.get("RA_ICRS", row1.get("ra", np.nan))))
    dec0 = _safe_float(row1.get("dec0", row1.get("DE_ICRS", row1.get("dec", np.nan))))
    rdeg = _safe_float(row1.get("radius", row1.get("r_deg", row1.get("r", np.nan))))
    plx0 = _safe_float(row1.get("plx0", row1.get("plx", row1.get("parallax", np.nan))))
    pmra0 = _safe_float(row1.get("pmra0", row1.get("pmra", np.nan)))
    pmdec0 = _safe_float(row1.get("pmdec0", row1.get("pmdec", np.nan)))

    if not np.isfinite(ra0):
        ra0 = float(np.nanmedian(pd.to_numeric(df_cone["ra"], errors="coerce")))
    if not np.isfinite(dec0):
        dec0 = float(np.nanmedian(pd.to_numeric(df_cone["dec"], errors="coerce")))
    if not np.isfinite(rdeg) or rdeg <= 0:
        rdeg = 1.0
    if not np.isfinite(plx0):
        plx0 = None
    if not np.isfinite(pmra0):
        pmra0 = None
    if not np.isfinite(pmdec0):
        pmdec0 = None

    true_set = set(int(x) for x in np.asarray(true_ids, dtype=np.int64))

    if len(df_cone) == 0:
        empty = RunResult(
            cluster=cluster,
            variant=spec.name,
            n_cone=0,
            n_raw_cone=int(len(df_cone_raw)),
            n_quality_rejected=int(n_rej),
            n_true_in_cone=0,
            n_pred=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            contam=1.0,
            runtime_s=float(_now() - t0),
            center_mode="center0",
            center_shift_arcmin=0.0,
            objective=-1e9,
            tag="EMPTY_AFTER_PREPROC",
        )
        cache = {
            "cluster": cluster,
            "variant": spec.name,
            "pred_ids": np.array([], dtype=np.int64),
            "true_ids": np.asarray(sorted(true_set), dtype=np.int64),
            "center_mode": "center0",
            "center_shift_arcmin": 0.0,
            "cone_df": df_cone.copy(),
            "pred_mask": np.array([], dtype=bool),
            "true_mask": np.array([], dtype=bool),
            "tag": "EMPTY_AFTER_PREPROC",
        }
        return empty, cache

    sid = pd.to_numeric(df_cone["source_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    true_mask = np.array([int(s) in true_set for s in sid], dtype=bool)

    best_res: Optional[RunResult] = None
    best_cache: Optional[Dict[str, Any]] = None
    best_obj = -1e9

    for center_mode, cra, cdec, cshift in refine_center_if_needed(df_cone, ra0, dec0, spec.center_refinement):
        score, X, pos2, kin2, _ = preprocess_astrometry(df_cone, cra, cdec, rdeg, plx0, pmra0, pmdec0)
        cand_idx = sample_candidate_subset(score, pos2, kin2) if spec.candidate_protocol else np.arange(len(X), dtype=np.int32)
        Xc = X[cand_idx]
        sc = score[cand_idx]

        pred_local, pack, support_local, radius_local = solve_mctnc_core_variant(Xc, sc, spec)
        pred_global = np.zeros(len(df_cone), dtype=bool)
        pred_global[cand_idx[pred_local]] = True
        pred_ids = sid[pred_global]

        pred_set = set(int(x) for x in pred_ids if int(x) > 0)
        n_true = int(true_mask.sum())
        precision, recall, f1 = metric_from_sets(pred_set, true_set)
        contam = float(1.0 - precision) if np.isfinite(precision) else 1.0
        objective = float(pack["objective"])

        res = RunResult(
            cluster=cluster,
            variant=spec.name,
            n_cone=int(len(df_cone)),
            n_raw_cone=int(len(df_cone_raw)),
            n_quality_rejected=int(n_rej),
            n_true_in_cone=int(n_true),
            n_pred=int(pred_global.sum()),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            contam=float(contam),
            runtime_s=float(_now() - t0),
            center_mode=center_mode,
            center_shift_arcmin=float(cshift),
            objective=objective,
            tag=str(pack["tag"]),
            chosen_k=float(pack.get("chosen_k", np.nan)),
            chosen_tau=float(pack.get("chosen_tau", np.nan)),
            mean_support=float(pack.get("mean_support", np.nan)),
            median_radius=float(pack.get("median_radius", np.nan)),
        )

        cache = {
            "cluster": cluster,
            "variant": spec.name,
            "pred_ids": np.asarray(sorted(pred_set), dtype=np.int64),
            "true_ids": np.asarray(sorted(true_set), dtype=np.int64),
            "support": support_local,
            "center_mode": center_mode,
            "center_shift_arcmin": float(cshift),
            "cone_df": df_cone.copy(),
            "pred_mask": pred_global.copy(),
            "true_mask": true_mask.copy(),
            "tag": str(pack["tag"]),
        }

        if objective > best_obj:
            best_obj = objective
            best_res = res
            best_cache = cache

    assert best_res is not None and best_cache is not None
    best_res.runtime_s = float(_now() - t0)
    return best_res, best_cache


# ============================================================================
# Cache helpers
# ============================================================================
def runs_to_frame(runs: List[RunResult]) -> pd.DataFrame:
    base_cols = list(RunResult.__dataclass_fields__.keys())
    if not runs:
        return pd.DataFrame(columns=base_cols + ["cache_pipeline_signature"])
    df = pd.DataFrame([asdict(r) for r in runs])
    for col in base_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[base_cols].copy()
    df["cache_pipeline_signature"] = CACHE_PIPELINE_SIGNATURE
    return df.sort_values(["variant", "cluster"]).reset_index(drop=True)


def try_load_variant_cache(fp: Path, variant: str, expected_clusters: List[str]) -> Optional[pd.DataFrame]:
    try:
        if not fp.exists() or fp.stat().st_size == 0:
            return None
        df = pd.read_csv(fp)
        if df.empty or "variant" not in df.columns or "cluster" not in df.columns:
            return None
        if "cache_pipeline_signature" in df.columns:
            sig_ok = df["cache_pipeline_signature"].astype(str).eq(CACHE_PIPELINE_SIGNATURE).all()
            if not sig_ok:
                return None
        else:
            return None
        df = df.loc[df["variant"].astype(str) == variant].copy()
        if sorted(df["cluster"].astype(str).tolist()) != sorted(expected_clusters):
            return None
        return df.reset_index(drop=True)
    except Exception:
        return None


def find_reusable_cache(data_dir: Path, current_cache: Path, variant: str, expected_clusters: List[str]) -> Optional[Tuple[pd.DataFrame, Path]]:
    local_df = try_load_variant_cache(current_cache, variant, expected_clusters)
    if local_df is not None:
        return local_df, current_cache
    pattern = f"mctnc_ablation_package_*/tables/cache_{variant}_cluster_results.csv"
    candidates = sorted(
        [p for p in data_dir.glob(pattern) if p.resolve() != current_cache.resolve()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for fp in candidates:
        df = try_load_variant_cache(fp, variant, expected_clusters)
        if df is not None:
            return df, fp
    return None


# ============================================================================
# Summary builders
# ============================================================================
def build_overall_summary(ab_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for variant, g in ab_df.groupby("variant"):
        parts.append(
            {
                "variant": variant,
                "n_clusters": int(len(g)),
                "mean_f1": float(pd.to_numeric(g["f1"], errors="coerce").mean()),
                "median_f1": float(pd.to_numeric(g["f1"], errors="coerce").median()),
                "p90_f1": float(pd.to_numeric(g["f1"], errors="coerce").quantile(0.90)),
                "mean_precision": float(pd.to_numeric(g["precision"], errors="coerce").mean()),
                "mean_recall": float(pd.to_numeric(g["recall"], errors="coerce").mean()),
                "mean_contam": float(pd.to_numeric(g["contam"], errors="coerce").mean()),
                "median_runtime_s": float(pd.to_numeric(g["runtime_s"], errors="coerce").median()),
                "success_frac_f1_ge_0_9": float((pd.to_numeric(g["f1"], errors="coerce") >= 0.90).mean()),
            }
        )
    return pd.DataFrame(parts).sort_values("mean_f1", ascending=False).reset_index(drop=True)


def add_tier_column(comp_df: pd.DataFrame, canonical_fullcore_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = comp_df.copy()
    if canonical_fullcore_df is not None and "tier" in canonical_fullcore_df.columns:
        tier_map = canonical_fullcore_df.set_index("cluster")["tier"].to_dict()
        out["tier"] = out["cluster"].map(tier_map)
        return out
    if "full_core_f1" in out.columns:
        f = pd.to_numeric(out["full_core_f1"], errors="coerce").to_numpy(float)
    else:
        f_cols = [c for c in out.columns if c.endswith("_f1")]
        f = np.nanmax(np.column_stack([pd.to_numeric(out[c], errors="coerce").to_numpy(float) for c in f_cols]), axis=1)
    tier = np.full(len(out), "Tier 4 (Borderline)", dtype=object)
    tier[f >= 0.995] = "Perfect Match"
    tier[(f < 0.995) & (f >= 0.95)] = "Tier 1 (Near-perfect)"
    tier[(f < 0.95) & (f >= 0.80)] = "Tier 2 (Conservative Core)"
    tier[(f < 0.80) & (f >= 0.50)] = "Tier 3 (Topological Over-expansion)"
    out["tier"] = tier
    return out


def build_comparison_matrix(ab_df: pd.DataFrame, canonical_fullcore_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    metric_cols = ["precision", "recall", "f1", "contam", "runtime_s"]
    out: Optional[pd.DataFrame] = None
    for variant, g in ab_df.groupby("variant"):
        g2 = g[["cluster"] + metric_cols].copy()
        g2 = g2.rename(columns={c: f"{variant}_{c}" for c in metric_cols})
        out = g2 if out is None else out.merge(g2, on="cluster", how="outer")
    if out is None:
        out = pd.DataFrame(columns=["cluster"])
    out = add_tier_column(out, canonical_fullcore_df)
    return out.sort_values("cluster").reset_index(drop=True)


def build_variant_delta_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    if "full_core_f1" not in comp_df.columns:
        return pd.DataFrame()
    parts = []
    for variant in ABLATION_ORDER:
        if variant == "full_core":
            continue
        col = f"{variant}_f1"
        if col not in comp_df.columns:
            continue
        d = pd.to_numeric(comp_df["full_core_f1"], errors="coerce") - pd.to_numeric(comp_df[col], errors="coerce")
        parts.append(
            {
                "variant": variant,
                "full_core_better_n": int((d > DELTA_TIE_TOL).sum()),
                "tie_n": int((np.abs(d) <= DELTA_TIE_TOL).sum()),
                "ablation_better_n": int((d < -DELTA_TIE_TOL).sum()),
                "mean_delta_f1_full_minus_variant": float(np.nanmean(d)),
                "median_delta_f1_full_minus_variant": float(np.nanmedian(d)),
                "p90_delta_f1_full_minus_variant": float(np.nanquantile(d, 0.90)),
            }
        )
    return pd.DataFrame(parts).sort_values("mean_delta_f1_full_minus_variant", ascending=False).reset_index(drop=True)


def build_tier_summary(comp_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tier, g in comp_df.groupby("tier", dropna=False):
        row = {"tier": tier, "n_clusters": int(len(g))}
        for variant in ABLATION_ORDER:
            for metric in ["f1", "precision", "recall", "contam", "runtime_s"]:
                col = f"{variant}_{metric}"
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors="coerce")
                    row[f"{variant}_mean_{metric}"] = float(vals.mean())
                    row[f"{variant}_median_{metric}"] = float(vals.median())
        parts.append(row)
    out = pd.DataFrame(parts)
    if out.empty:
        return out
    out["tier_order"] = out["tier"].map({name: i for i, name in enumerate(TIER_ORDER)}).fillna(len(TIER_ORDER))
    out = out.sort_values(["tier_order", "tier"]).drop(columns=["tier_order"]).reset_index(drop=True)
    return out


def build_method_rank_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    rank_methods = [v for v in ABLATION_ORDER if f"{v}_f1" in comp_df.columns]
    if not rank_methods:
        return pd.DataFrame()

    rank_store: Dict[str, List[float]] = {m: [] for m in rank_methods}
    best_count = {m: 0 for m in rank_methods}
    solo_best_count = {m: 0 for m in rank_methods}
    dense_rank_count = {m: {r: 0 for r in range(1, len(rank_methods) + 1)} for m in rank_methods}

    for _, row in comp_df.iterrows():
        vals = {m: pd.to_numeric(row.get(f"{m}_f1"), errors="coerce") for m in rank_methods}
        finite_items = [(m, float(v)) for m, v in vals.items() if pd.notna(v)]
        if not finite_items:
            continue
        best_val = max(v for _, v in finite_items)
        tied_best = [m for m, v in finite_items if abs(v - best_val) <= RANK_TIE_TOL]
        for m in tied_best:
            best_count[m] += 1
        if len(tied_best) == 1:
            solo_best_count[tied_best[0]] += 1
        finite_items = sorted(finite_items, key=lambda x: (-x[1], x[0]))
        dense_rank_map: Dict[str, int] = {}
        current_rank = 0
        previous_val = None
        for m, v in finite_items:
            if previous_val is None or abs(v - previous_val) > RANK_TIE_TOL:
                current_rank += 1
                previous_val = v
            dense_rank_map[m] = current_rank
        for m in rank_methods:
            if m in dense_rank_map:
                rank_val = dense_rank_map[m]
                rank_store[m].append(float(rank_val))
                dense_rank_count[m][rank_val] += 1

    rows = []
    for m in rank_methods:
        entry = {
            "variant": m,
            "best_n": int(best_count[m]),
            "solo_best_n": int(solo_best_count[m]),
            "mean_dense_rank": float(np.mean(rank_store[m])) if rank_store[m] else np.nan,
            "median_dense_rank": float(np.median(rank_store[m])) if rank_store[m] else np.nan,
        }
        for r in range(1, len(rank_methods) + 1):
            entry[f"rank{r}_n"] = int(dense_rank_count[m][r])
        entry["rank1_n"] = entry["best_n"]
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(["mean_dense_rank", "variant"]).reset_index(drop=True)


def build_reproduction_audit(comp_df: pd.DataFrame) -> pd.DataFrame:
    if "full_core_internal_f1" not in comp_df.columns or "full_core_f1" not in comp_df.columns:
        return pd.DataFrame()
    d = pd.to_numeric(comp_df["full_core_internal_f1"], errors="coerce") - pd.to_numeric(comp_df["full_core_f1"], errors="coerce")
    p = pd.to_numeric(comp_df["full_core_internal_precision"], errors="coerce") - pd.to_numeric(comp_df["full_core_precision"], errors="coerce")
    r = pd.to_numeric(comp_df["full_core_internal_recall"], errors="coerce") - pd.to_numeric(comp_df["full_core_recall"], errors="coerce")
    rows = [{
        "n_clusters": int(len(comp_df)),
        "exact_f1_match_n": int((np.abs(d) <= 1e-12).sum()),
        "median_abs_delta_f1": float(np.nanmedian(np.abs(d))),
        "p90_abs_delta_f1": float(np.nanquantile(np.abs(d), 0.90)),
        "mean_delta_f1_internal_minus_canonical": float(np.nanmean(d)),
        "mean_delta_precision_internal_minus_canonical": float(np.nanmean(p)),
        "mean_delta_recall_internal_minus_canonical": float(np.nanmean(r)),
    }]
    return pd.DataFrame(rows)


def build_design_table() -> pd.DataFrame:
    rows = []
    for spec in ABLATION_SPECS:
        rows.append(
            {
                "variant": spec.name,
                "title": spec.title,
                "quality_cut": int(spec.quality_cut),
                "candidate_protocol": int(spec.candidate_protocol),
                "anchor_prior": int(spec.anchor_prior),
                "adaptive_k": int(spec.adaptive_k),
                "support_gate": int(spec.support_gate),
                "objective_regularization": int(spec.objective_regularization),
                "center_refinement": int(spec.center_refinement),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)



# ============================================================================
# Plotting
# ============================================================================
def savefig(fig: plt.Figure, out: Path) -> None:
    ensure_dir(out.parent)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_design_matrix(design_df: pd.DataFrame, out: Path) -> None:
    cols = ["quality_cut", "candidate_protocol", "anchor_prior", "adaptive_k", "support_gate", "objective_regularization", "center_refinement"]
    labels = ["quality cut", "candidate protocol", "anchor prior", "adaptive k", "support gate", "objective regularization", "center refinement"]
    mat = design_df[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(design_df)))
    ax.set_yticklabels(design_df["variant"].tolist())
    ax.set_title("Ablation design matrix of the M-CTNC core suite")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "On" if mat[i, j] >= 0.5 else "Off", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Enabled")
    fig.tight_layout()
    savefig(fig, out)


def plot_overall_distributions(summary_df: pd.DataFrame, ab_df: pd.DataFrame, out: Path) -> None:
    methods = summary_df["variant"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("runtime_s", "Runtime (s)")]
    for ax, (col, lab) in zip(axes.ravel(), panels):
        data, labels = [], []
        for method in methods:
            vals = pd.to_numeric(ab_df.loc[ab_df["variant"] == method, col], errors="coerce").dropna().to_numpy(float)
            if vals.size > 0:
                data.append(vals)
                labels.append(method)
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(lab)
        ax.grid(True, alpha=0.25)
        if col == "runtime_s":
            ax.set_yscale("log")
    fig.suptitle("Population distributions of the canonical CORE_BENCHMARK ablation suite", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    savefig(fig, out)


def plot_overall_summary(summary_df: pd.DataFrame, out: Path) -> None:
    df = summary_df.copy()
    order = list(df.sort_values("mean_f1", ascending=True)["variant"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.barh(order, df.set_index("variant").loc[order, "mean_f1"])
    for i, name in enumerate(order):
        v = float(df.set_index("variant").loc[name, "mean_f1"])
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Mean F1")
    ax.set_title("Overall accuracy ranking")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1]
    rt = df.set_index("variant").loc[order, "median_runtime_s"].to_numpy(float)
    ax.barh(order, rt)
    for i, v in enumerate(rt):
        ax.text(v * 1.02 if np.isfinite(v) and v > 0 else 0.01, i, f"{v:.2f}", va="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Median runtime (s)")
    ax.set_title("Runtime ranking")
    ax.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Population-level summary of the M-CTNC ablation suite", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    savefig(fig, out)


def plot_delta_vs_full(comp_df: pd.DataFrame, out: Path) -> None:
    variants = [v for v in ABLATION_ORDER if v != "full_core" and f"{v}_f1" in comp_df.columns]
    if not variants or "full_core_f1" not in comp_df.columns:
        return
    ncols = min(3, len(variants))
    nrows = int(math.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, variant in zip(axes, variants):
        d = pd.to_numeric(comp_df["full_core_f1"], errors="coerce") - pd.to_numeric(comp_df[f"{variant}_f1"], errors="coerce")
        vals = np.sort(d.to_numpy(float))
        ax.plot(np.arange(1, len(vals) + 1), vals, lw=2)
        ax.axhline(0.0, ls="--", lw=1.2)
        ax.axhline(0.01, ls=":", lw=1.0)
        ax.axhline(-0.01, ls=":", lw=1.0)
        ax.set_title(f"full_core - {variant}")
        ax.set_xlabel("Cluster rank")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(variants):]:
        ax.axis("off")
    axes[0].set_ylabel(r"$\Delta F_1$")
    fig.suptitle("Cluster-wise full-core advantage over each ablation", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    savefig(fig, out)


def plot_tier_stratified(comp_df: pd.DataFrame, out: Path) -> None:
    present_tiers = [t for t in TIER_ORDER if t in set(comp_df["tier"].astype(str))]
    methods = [v for v in ABLATION_ORDER if f"{v}_f1" in comp_df.columns]
    rows = []
    for tier in present_tiers:
        sub = comp_df.loc[comp_df["tier"] == tier]
        for method in methods:
            col = f"{method}_f1"
            if col in sub.columns:
                rows.append({"tier": tier, "variant": method, "mean_f1": float(pd.to_numeric(sub[col], errors="coerce").mean()), "n_clusters": int(len(sub))})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(present_tiers))
    width = max(0.10, 0.78 / max(len(methods), 1))
    offsets = np.linspace(-0.39 + width / 2, 0.39 - width / 2, len(methods))
    for off, method in zip(offsets, methods):
        vals = [
            float(df.loc[(df["tier"] == t) & (df["variant"] == method), "mean_f1"].iloc[0])
            if not df.loc[(df["tier"] == t) & (df["variant"] == method)].empty else np.nan
            for t in present_tiers
        ]
        ax.bar(x + off, vals, width=width, label=method)
    tick_labels = []
    for t in present_tiers:
        n_clusters = int(df.loc[df["tier"] == t, "n_clusters"].iloc[0])
        tick_labels.append(f"{t}\n(n={n_clusters})")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=12, ha="right")
    ax.set_ylabel("Mean F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Tier-stratified comparison of the ablation suite")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    savefig(fig, out)


def plot_contam_recall_frontier(summary_df: pd.DataFrame, out: Path) -> None:
    df = summary_df.copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(df["mean_contam"], df["mean_recall"], s=120)
    for _, row in df.iterrows():
        ax.text(float(row["mean_contam"]) + 0.003, float(row["mean_recall"]) + 0.003, str(row["variant"]), fontsize=10)
    ax.set_xlabel("Mean contamination")
    ax.set_ylabel("Mean recall")
    ax.set_title("Contamination-recall operating frontier of the ablation suite")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, out)


def plot_runtime_accuracy(summary_df: pd.DataFrame, out: Path) -> None:
    df = summary_df.copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(df["median_runtime_s"], df["mean_f1"], s=120)
    for _, row in df.iterrows():
        ax.text(float(row["median_runtime_s"]) * 1.03 if float(row["median_runtime_s"]) > 0 else 0.01, float(row["mean_f1"]) + 0.002, str(row["variant"]), fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Median runtime (s)")
    ax.set_ylabel("Mean F1")
    ax.set_title("Accuracy-efficiency trade-off of the ablation suite")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, out)


def plot_top_cases(comp_df: pd.DataFrame, out: Path) -> pd.DataFrame:
    if "full_core_f1" not in comp_df.columns:
        return pd.DataFrame()
    best_other = []
    best_name = []
    for _, row in comp_df.iterrows():
        items = []
        for variant in ABLATION_ORDER:
            if variant == "full_core":
                continue
            col = f"{variant}_f1"
            if col in comp_df.columns and pd.notna(row.get(col)):
                items.append((variant, float(row[col])))
        if not items:
            best_other.append(np.nan)
            best_name.append("")
        else:
            items.sort(key=lambda x: (-x[1], x[0]))
            best_name.append(items[0][0])
            best_other.append(items[0][1])
    work = comp_df.copy()
    work["best_ablation"] = best_name
    work["best_ablation_f1"] = best_other
    work["full_minus_best_ablation"] = pd.to_numeric(work["full_core_f1"], errors="coerce") - pd.to_numeric(work["best_ablation_f1"], errors="coerce")
    up = work.sort_values("full_minus_best_ablation", ascending=False).head(TOP_CASES)
    dn = work.sort_values("full_minus_best_ablation", ascending=True).head(TOP_CASES)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for ax, df, ttl in [
        (axes[0], up.iloc[::-1], "Top full-core advantages over the best ablation"),
        (axes[1], dn, "Cases where an ablation approaches or exceeds the full core"),
    ]:
        ax.barh(df["cluster"], df["full_minus_best_ablation"])
        ax.axvline(0.0, ls="--", lw=1.2)
        ax.set_title(ttl)
        ax.set_xlabel(r"$F_1(\mathrm{full\_core}) - \max(F_1\ \mathrm{ablation})$")
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, out)
    return work[["cluster", "tier", "best_ablation", "best_ablation_f1", "full_core_f1", "full_minus_best_ablation"]].sort_values("full_minus_best_ablation", ascending=False).reset_index(drop=True)


def select_representative_cases(case_cache_map: Dict[Tuple[str, str], Dict[str, Any]], case_delta_df: pd.DataFrame, comp_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if case_delta_df.empty:
        return pd.DataFrame(), []
    chosen_rows = pd.concat([
        case_delta_df.head(3).assign(selection_group="core_favored"),
        case_delta_df.tail(3).assign(selection_group="ablation_favored"),
    ], axis=0).drop_duplicates("cluster").reset_index(drop=True)
    comp_lookup = comp_df.set_index("cluster")
    case_rows = []
    case_caches = []
    for _, row in chosen_rows.iterrows():
        cluster = str(row["cluster"])
        best_ablation = str(row.get("best_ablation", ""))
        cache = case_cache_map.get((cluster, best_ablation))
        if cache is None:
            continue
        cache = dict(cache)
        comp_row = comp_lookup.loc[cluster] if cluster in comp_lookup.index else pd.Series(dtype=object)
        cache["tier"] = row.get("tier", "")
        cache["delta_full_minus_best_ablation"] = row.get("full_minus_best_ablation", np.nan)
        cache["selection_group"] = row["selection_group"]
        cache["display_variant"] = best_ablation
        cache["canonical_fullcore_f1"] = float(pd.to_numeric(comp_row.get("full_core_f1"), errors="coerce")) if not comp_row.empty else np.nan
        cache["canonical_fullcore_precision"] = float(pd.to_numeric(comp_row.get("full_core_precision"), errors="coerce")) if not comp_row.empty else np.nan
        cache["canonical_fullcore_recall"] = float(pd.to_numeric(comp_row.get("full_core_recall"), errors="coerce")) if not comp_row.empty else np.nan
        cache["display_variant_f1"] = float(pd.to_numeric(comp_row.get(f"{best_ablation}_f1"), errors="coerce")) if not comp_row.empty else np.nan
        cache["display_variant_precision"] = float(pd.to_numeric(comp_row.get(f"{best_ablation}_precision"), errors="coerce")) if not comp_row.empty else np.nan
        cache["display_variant_recall"] = float(pd.to_numeric(comp_row.get(f"{best_ablation}_recall"), errors="coerce")) if not comp_row.empty else np.nan
        case_caches.append(cache)
        case_rows.append({
            "cluster": cluster,
            "selection_group": row["selection_group"],
            "display_variant": best_ablation,
            "best_ablation": best_ablation,
            "tier": row.get("tier", ""),
            "full_core_f1": row.get("full_core_f1", np.nan),
            "best_ablation_f1": row.get("best_ablation_f1", np.nan),
            "full_minus_best_ablation": row.get("full_minus_best_ablation", np.nan),
        })
    return pd.DataFrame(case_rows), case_caches


def plot_case_panels(case_caches: List[Dict[str, Any]], out: Path) -> None:
    if not case_caches:
        return
    n = len(case_caches)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.3 * n))
    if n == 1:
        axes = np.array([axes])

    cone_color = "#4C78A8"
    truth_color = "#54A24B"
    pred_color = "#E45756"

    for row_ax, cache in zip(axes, case_caches):
        df = cache["cone_df"]
        pred = cache["pred_mask"]
        true = cache["true_mask"]
        cluster = cache["cluster"]
        variant = cache.get("display_variant", cache.get("variant", ""))
        tier = str(cache.get("tier", ""))
        delta = float(cache.get("delta_full_minus_best_ablation", np.nan))
        group = str(cache.get("selection_group", ""))

        x = pd.to_numeric(df["pmra"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df["pmdec"], errors="coerce").to_numpy(float)
        row_ax[0].scatter(x, y, s=4, alpha=0.18, c=cone_color, edgecolors="none")
        row_ax[0].scatter(x[true], y[true], s=12, alpha=0.90, c=truth_color, marker="x")
        row_ax[0].scatter(x[pred], y[pred], s=10, alpha=0.70, c=pred_color, edgecolors="none")
        row_ax[0].set_xlabel("pmRA")
        row_ax[0].set_ylabel("pmDec")
        row_ax[0].set_title(f"{cluster} | best ablation = {variant} | PM plane")
        row_ax[0].grid(True, alpha=0.20)

        x2 = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
        y2 = pd.to_numeric(df["pmra"], errors="coerce").to_numpy(float)
        row_ax[1].scatter(x2, y2, s=4, alpha=0.18, c=cone_color, edgecolors="none")
        row_ax[1].scatter(x2[true], y2[true], s=12, alpha=0.90, c=truth_color, marker="x")
        row_ax[1].scatter(x2[pred], y2[pred], s=10, alpha=0.70, c=pred_color, edgecolors="none")
        row_ax[1].set_xlabel("Parallax")
        row_ax[1].set_ylabel("pmRA")
        row_ax[1].set_title("Parallax-motion projection")
        row_ax[1].grid(True, alpha=0.20)

        ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(float)
        dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float)
        row_ax[2].scatter(ra, dec, s=4, alpha=0.18, c=cone_color, edgecolors="none")
        row_ax[2].scatter(ra[true], dec[true], s=12, alpha=0.90, c=truth_color, marker="x")
        row_ax[2].scatter(ra[pred], dec[pred], s=10, alpha=0.70, c=pred_color, edgecolors="none")
        row_ax[2].set_xlabel("RA")
        row_ax[2].set_ylabel("Dec")
        row_ax[2].set_title("Sky-plane footprint")
        row_ax[2].grid(True, alpha=0.20)

        info = (
            f"{group} | {tier}\n"
            f"canonical full_core: F1={cache.get('canonical_fullcore_f1', np.nan):.3f}, "
            f"P={cache.get('canonical_fullcore_precision', np.nan):.3f}, R={cache.get('canonical_fullcore_recall', np.nan):.3f}\n"
            f"shown ablation ({variant}): F1={cache.get('display_variant_f1', np.nan):.3f}, "
            f"P={cache.get('display_variant_precision', np.nan):.3f}, R={cache.get('display_variant_recall', np.nan):.3f}\n"
            f"ΔF1(full-best_abl) = {delta:+.3f}"
        )
        row_ax[1].text(0.02, 0.98, info, transform=row_ax[1].transAxes, va="top", ha="left", fontsize=9,
                       bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"})

    fig.suptitle("Representative competitive case panels relative to the canonical CORE_BENCHMARK full-core reference", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    savefig(fig, out)


# ============================================================================
# Export helpers
# ============================================================================
def export_excel_package(out_xlsx: Path, tables: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)


def write_readme(out_path: Path, mctnc_path: Optional[Path], mctnc_meta: Dict[str, Any], canonical_source: str, internal_diag_enabled: bool) -> None:
    lines = [
        "Unified Part-Six ablation suite for M-CTNC",
        "=========================================",
        f"Version: {VERSION}",
        f"Variants: {', '.join(ABLATION_ORDER)}",
        f"Imported M-CTNC reference: {str(mctnc_path) if mctnc_path is not None else 'not found'}",
        "Benchmark policy: canonical CORE_BENCHMARK ablation suite",
        "Primary full-core policy: use imported CORE_BENCHMARK reference as the canonical full-core comparator whenever available",
        "Reference rejection policy: halo/exploration-style result files are excluded from canonical import",
        f"Canonical full-core source: {canonical_source}",
        f"Internal full-core diagnostic enabled: {int(internal_diag_enabled)}",
        f"M-CTNC mode filter policy: {mctnc_meta.get('filter_policy', 'not_found')}",
        f"M-CTNC source-name policy: {mctnc_meta.get('source_name_policy', 'not_found')}",
        f"M-CTNC duplicate policy: {mctnc_meta.get('duplicate_policy', 'none')}",
        "",
        "Key exported tables:",
        "  tables/ablation_cluster_results.csv",
        "  tables/ablation_overall_summary.csv",
        "  tables/ablation_comparison_matrix.csv",
        "  tables/ablation_delta_vs_fullcore.csv",
        "  tables/ablation_design_table.csv",
        "  tables/ablation_rank_table.csv",
        "  tables/ablation_tier_stratified_summary.csv",
        "  tables/ablation_import_audit_table.csv",
        "  tables/ablation_fullcore_source_audit.csv",
        "  tables/ablation_fullcore_vs_internal_audit.csv",
        "  tables/ablation_case_selection.csv",
        "",
        "Key exported figures:",
        "  FigA00_ablation_design_matrix.png",
        "  FigA01_ablation_distributions.png",
        "  FigA02_ablation_overall_summary.png",
        "  FigA03_ablation_delta_vs_fullcore.png",
        "  FigA04_ablation_tier_stratified_comparison.png",
        "  FigA05_ablation_contam_recall_frontier.png",
        "  FigA06_ablation_runtime_accuracy_tradeoff.png",
        "  FigA07_ablation_top_case_deltas.png",
        "  FigA08_ablation_representative_cases.png",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified release-grade M-CTNC ablation suite.")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cone_dir", type=str, default=None)
    parser.add_argument("--mctnc_csv", type=str, default=None, help="Optional imported M-CTNC CORE results csv. Defaults to the canonical ApJS CORE benchmark table when available.")
    parser.add_argument("--max_clusters", type=int, default=0, help="0 means all clusters.")
    parser.add_argument("--run_internal_fullcore_diag", action="store_true", help="Additionally run the internal full-core engine as an appendix diagnostic.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (base_dir / "data")
    cone_dir = Path(args.cone_dir).resolve() if args.cone_dir else None
    ensure_dir(data_dir)

    out_root = data_dir / f"mctnc_ablation_package_{VERSION}"
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    print(f"[SYSTEM] {PIPELINE_NAME}")
    print(f"[SYSTEM] base_dir = {base_dir}")
    print(f"[SYSTEM] data_dir = {data_dir}")

    t1_idx, true_map = load_benchmark_tables(data_dir)
    cone_files = discover_cone_files(base_dir, data_dir, cone_dir)
    tasks = [(infer_cluster_name(fp), fp) for fp in cone_files if infer_cluster_name(fp) in t1_idx]
    if args.max_clusters and args.max_clusters > 0:
        tasks = tasks[: int(args.max_clusters)]
    if len(tasks) == 0:
        raise FileNotFoundError("No gaia_cone_*.csv files aligned with table1 clusters were found.")

    requested_mctnc = Path(args.mctnc_csv).resolve() if args.mctnc_csv else None
    user_mctnc = resolve_preferred_mctnc_core_reference(base_dir, data_dir, requested_mctnc)
    mctnc_ref_df, mctnc_ref_path, mctnc_meta = read_mctnc_results(data_dir, user_mctnc)
    if mctnc_ref_path is not None:
        print(f"[SYSTEM] imported M-CTNC reference = {mctnc_ref_path}")
        print(f"[SYSTEM] M-CTNC import policy = {mctnc_meta.get('filter_policy', 'unknown')}")
        print(f"[SYSTEM] M-CTNC source-name policy = {mctnc_meta.get('source_name_policy', 'unknown')}")
    else:
        print("[SYSTEM] no imported M-CTNC reference found; canonical full-core will be produced internally.")

    use_imported_as_fullcore = bool(USE_IMPORTED_CORE_AS_CANONICAL_FULLCORE and (mctnc_ref_df is not None) and (mctnc_ref_path is not None))
    canonical_source = "imported_core_benchmark_reference" if use_imported_as_fullcore else "internally_reproduced_full_core"
    if use_imported_as_fullcore:
        print("[SYSTEM] canonical full_core policy = imported CORE_BENCHMARK reference")
    else:
        print("[SYSTEM] canonical full_core policy = internal full_core reproduction")

    active_specs: List[AblationSpec] = []
    if use_imported_as_fullcore:
        active_specs.extend([spec for spec in ABLATION_SPECS if spec.name != "full_core"])
    else:
        active_specs.extend(ABLATION_SPECS)

    internal_diag_spec = None
    if args.run_internal_fullcore_diag and use_imported_as_fullcore:
        internal_diag_spec = AblationSpec(
            name="full_core_internal",
            title="Internal full core diagnostic",
            description="Internal reproduction of the full core engine for appendix-level audit only.",
        )
        active_specs = [internal_diag_spec] + active_specs

    task_clusters = [cluster for cluster, _ in tasks]
    canonical_runs: List[RunResult] = []
    diag_runs: List[RunResult] = []
    case_cache_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    t_all = _now()
    for vi, spec in enumerate(active_specs, start=1):
        cache_name = spec.name
        cache_fp = tab_dir / f"cache_{cache_name}_cluster_results.csv"
        print(f"\n[VARIANT {vi}/{len(active_specs)}] {spec.name}")

        reusable = find_reusable_cache(data_dir, cache_fp, cache_name, task_clusters)
        if reusable is not None:
            cache_df, cache_src = reusable
            target = diag_runs if spec.name == "full_core_internal" else canonical_runs
            for _, r in cache_df.iterrows():
                target.append(
                    RunResult(
                        cluster=str(r["cluster"]),
                        variant=str(r["variant"]),
                        n_cone=int(r["n_cone"]) if pd.notna(r["n_cone"]) else 0,
                        n_raw_cone=int(r["n_raw_cone"]) if pd.notna(r["n_raw_cone"]) else 0,
                        n_quality_rejected=int(r["n_quality_rejected"]) if pd.notna(r["n_quality_rejected"]) else 0,
                        n_true_in_cone=int(r["n_true_in_cone"]) if pd.notna(r["n_true_in_cone"]) else 0,
                        n_pred=int(r["n_pred"]) if pd.notna(r["n_pred"]) else 0,
                        precision=float(r["precision"]),
                        recall=float(r["recall"]),
                        f1=float(r["f1"]),
                        contam=float(r["contam"]),
                        runtime_s=float(r["runtime_s"]),
                        center_mode=str(r["center_mode"]),
                        center_shift_arcmin=float(r["center_shift_arcmin"]),
                        objective=float(r["objective"]),
                        tag=str(r["tag"]),
                        chosen_k=float(r.get("chosen_k", np.nan)),
                        chosen_tau=float(r.get("chosen_tau", np.nan)),
                        mean_support=float(r.get("mean_support", np.nan)),
                        median_radius=float(r.get("median_radius", np.nan)),
                    )
                )
            print(f"  -> reused cached results for {spec.name}: {cache_src}")
            continue

        variant_runs: List[RunResult] = []
        for i, (cluster, fp) in enumerate(tasks, start=1):
            df_cone_raw = read_cone_csv(fp)
            bench_spec = AblationSpec(
                name="full_core" if spec.name == "full_core_internal" else spec.name,
                title=spec.title,
                description=spec.description,
                quality_cut=spec.quality_cut,
                candidate_protocol=spec.candidate_protocol,
                anchor_prior=spec.anchor_prior,
                adaptive_k=spec.adaptive_k,
                support_gate=spec.support_gate,
                objective_regularization=spec.objective_regularization,
                center_refinement=spec.center_refinement,
            )
            res, cache = process_single_cluster(bench_spec, cluster, df_cone_raw, t1_idx[cluster], true_map[cluster])
            if spec.name == "full_core_internal":
                res.variant = "full_core_internal"
            variant_runs.append(res)
            case_cache_map[(cluster, res.variant)] = cache
            if (i % 25 == 0) or (i == len(tasks)):
                print(f"  -> processed {i}/{len(tasks)} clusters for {spec.name}")

        variant_df = runs_to_frame(variant_runs)
        variant_df.to_csv(cache_fp, index=False)
        if spec.name == "full_core_internal":
            diag_runs.extend(variant_runs)
        else:
            canonical_runs.extend(variant_runs)

    canonical_ab_df = runs_to_frame(canonical_runs)
    if use_imported_as_fullcore:
        imported_full_core_df = build_canonical_fullcore_from_import(mctnc_ref_df)
        canonical_ab_df = canonical_ab_df.loc[canonical_ab_df["variant"] != "full_core"].reset_index(drop=True)
        canonical_ab_df = pd.concat([imported_full_core_df, canonical_ab_df], axis=0, ignore_index=True)
        canonical_tier_source = mctnc_ref_df.copy()
    else:
        canonical_tier_source = None

    diag_df = runs_to_frame(diag_runs)
    summary_df = build_overall_summary(canonical_ab_df)
    comp_df = build_comparison_matrix(canonical_ab_df, canonical_tier_source)
    delta_df = build_variant_delta_table(comp_df)
    tier_df = build_tier_summary(comp_df)
    rank_df = build_method_rank_table(comp_df)
    design_df = build_design_table()
    fullcore_source_audit_df = build_fullcore_source_audit(mctnc_ref_path, mctnc_meta, canonical_source, args.run_internal_fullcore_diag and use_imported_as_fullcore)

    if not diag_df.empty:
        comp_internal = build_comparison_matrix(pd.concat([canonical_ab_df, diag_df], axis=0, ignore_index=True), canonical_tier_source)
        repr_audit_df = compare_internal_fullcore_to_canonical(comp_internal)
    else:
        repr_audit_df = pd.DataFrame()

    case_delta_df = plot_top_cases(comp_df, fig_dir / "FigA07_ablation_top_case_deltas.png")
    case_select_df, case_caches = select_representative_cases(case_cache_map, case_delta_df, comp_df)

    canonical_ab_df.to_csv(tab_dir / "ablation_cluster_results.csv", index=False)
    summary_df.to_csv(tab_dir / "ablation_overall_summary.csv", index=False)
    comp_df.to_csv(tab_dir / "ablation_comparison_matrix.csv", index=False)
    delta_df.to_csv(tab_dir / "ablation_delta_vs_fullcore.csv", index=False)
    tier_df.to_csv(tab_dir / "ablation_tier_stratified_summary.csv", index=False)
    rank_df.to_csv(tab_dir / "ablation_rank_table.csv", index=False)
    design_df.to_csv(tab_dir / "ablation_design_table.csv", index=False)
    pd.DataFrame([mctnc_meta]).to_csv(tab_dir / "ablation_import_audit_table.csv", index=False)
    fullcore_source_audit_df.to_csv(tab_dir / "ablation_fullcore_source_audit.csv", index=False)
    repr_audit_df.to_csv(tab_dir / "ablation_fullcore_vs_internal_audit.csv", index=False)
    case_select_df.to_csv(tab_dir / "ablation_case_selection.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(tab_dir / "cache_full_core_internal_cluster_results.csv", index=False)

    plot_design_matrix(design_df, fig_dir / "FigA00_ablation_design_matrix.png")
    plot_overall_distributions(summary_df, canonical_ab_df, fig_dir / "FigA01_ablation_distributions.png")
    plot_overall_summary(summary_df, fig_dir / "FigA02_ablation_overall_summary.png")
    plot_delta_vs_full(comp_df, fig_dir / "FigA03_ablation_delta_vs_fullcore.png")
    plot_tier_stratified(comp_df, fig_dir / "FigA04_ablation_tier_stratified_comparison.png")
    plot_contam_recall_frontier(summary_df, fig_dir / "FigA05_ablation_contam_recall_frontier.png")
    plot_runtime_accuracy(summary_df, fig_dir / "FigA06_ablation_runtime_accuracy_tradeoff.png")
    plot_case_panels(case_caches, fig_dir / "FigA08_ablation_representative_cases.png")

    export_excel_package(
        out_root / "mctnc_ablation_package.xlsx",
        {
            "ablation_cluster_results": canonical_ab_df,
            "ablation_overall_summary": summary_df,
            "ablation_comparison_matrix": comp_df,
            "ablation_delta_vs_fullcore": delta_df,
            "ablation_tier_summary": tier_df,
            "ablation_rank_table": rank_df,
            "ablation_design_table": design_df,
            "ablation_import_audit": pd.DataFrame([mctnc_meta]),
            "fullcore_source_audit": fullcore_source_audit_df,
            "fullcore_vs_internal": repr_audit_df,
            "ablation_case_selection": case_select_df,
        },
    )
    write_readme(out_root / "README_ablation.txt", mctnc_ref_path, mctnc_meta, canonical_source, args.run_internal_fullcore_diag and use_imported_as_fullcore)

    elapsed = _now() - t_all
    print("\n============================================================")
    print("M-CTNC ablation package completed.")
    print(f"Output folder: {out_root}")
    print(f"Elapsed time : {elapsed:.2f} s")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary_df.to_string(index=False))
    print("============================================================")


if __name__ == "__main__":
    main()
