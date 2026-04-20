from __future__ import annotations

import argparse
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pandas.errors import EmptyDataError
from sklearn.mixture import GaussianMixture

# HDBSCAN compatibility
try:
    from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
except Exception:
    try:
        from hdbscan import HDBSCAN  # fallback package
    except Exception as e:
        raise ImportError(
            "Neither sklearn.cluster.HDBSCAN nor hdbscan.HDBSCAN is available. "
            "Please install scikit-learn>=1.3 or hdbscan."
        ) from e

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")


# ============================================================================
# Unified configuration
# ============================================================================
PIPELINE_NAME = "MCTNC_PART5_BASELINE_SUITE"
VERSION = "part5_baselines_v1_6_final_coremode_closure_corecsvfix"

CACHE_PIPELINE_SIGNATURE = "PART5_SHARED_HDBSCAN_RELEASE_v1_6_COREMODE_CORECSVFIX"

ENABLE_QUALITY_CUTS = True
RUWE_MAX = 1.6

# Shared astrometric normalization
W_POS = 1.00
W_PLX = 1.00
W_PM = 1.00
FLOOR_PLX = 0.10
FLOOR_PM = 0.20

# Shared candidate subset
CAP_MIN, CAP_MAX = 300, 1200
CAP_A, CAP_B = 12, 250
CAND_SCORE_RATIO, CAND_POS_RATIO, CAND_KIN_RATIO = 0.60, 0.25, 0.15

# Shared center refinement
ENABLE_CENTER_REFINEMENT = True
CENTER_REFINE_TOP_R = 600
CENTER_REFINE_K = 24
CENTER_SHIFT_BLEND = 0.70
CENTER_SHIFT_LIMIT_ARCMIN = 20.0

# Heuristic-Cut
HC_SIGMA = 2.50
HC_MAX_ITER = 4
HC_MIN_MEMBERS = 5
HC_SEED_TOP_SCORE = 24
HC_ANCHOR_TOPK = 20
SIGMA_FLOOR_VEC = np.array([0.16, 0.16, 0.26, 0.30, 0.30], dtype=np.float32)

# GMM+BIC
GMM_N_COMPONENTS_MAX = 5
GMM_COVARIANCE_TYPES = ["full", "diag"]
GMM_PROB_THRESH = 0.40
GMM_KIN_CROP_MAX = 2500
GMM_KIN_CROP_THRESHOLD = 11.34

# HDBSCAN
HDB_MIN_CLUSTER_SIZE = 8
HDB_MIN_SAMPLES = 5
HDB_CLUSTER_SELECTION_EPS = 0.0
HDB_ALPHA = 1.0

# Shared-benchmark policy
# All baselines are evaluated under the same upstream data-conditioning protocol:
# RUWE quality control, catalog-seeded normalization, and the same shared-cone input files.
# For the strong density baseline reported in the paper, HDBSCAN is coupled to the
# shared candidate protocol and shared center-refinement path, but never to truth-guided
# model selection. This benchmark should therefore be described in the manuscript as
# a shared-preprocessing HDBSCAN baseline, abbreviated simply as HDBSCAN after the
# protocol is defined once in the methods section.
ALLOW_CENTER_REFINEMENT_FOR_HEURISTIC = False
ALLOW_CENTER_REFINEMENT_FOR_HDBSCAN = True
REUSE_COMPLETED_METHODS = True
BASELINE_METHODS = ["gmm_bic", "heuristic_cut", "hdbscan"]

# Package controls
FIG_DPI = 220
TOP_CASES = 12
RANK_TIE_TOL = 1e-10
MCTNC_REQUIRE_CORE_MODE = True
OFFICIAL_MCTNC_CORE_FILENAME = "ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv"
TIER_ORDER = [
    "Perfect Match",
    "Tier 1 (Near-perfect)",
    "Tier 2 (Conservative Core)",
    "Tier 3 (Topological Over-expansion)",
    "Tier 4 (Borderline)",
]


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass
class RunResult:
    cluster: str
    method: str
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
    extra_1: float = np.nan
    extra_2: float = np.nan


# ============================================================================
# Utilities
# ============================================================================
def _now() -> float:
    return time.perf_counter()


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    candidates.extend(
        [data_dir, base_dir, base_dir / "cones", base_dir / "cone", base_dir / "data-gaia", base_dir / "gaia"]
    )
    seen = set()
    dirs: List[Path] = []
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
            files.extend(
                [
                    p
                    for p in d.rglob("*")
                    if p.is_file() and p.name.lower().startswith("gaia_cone_") and p.name.lower().endswith(".csv")
                ]
            )
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
        "phot_g_mean_mag": ["phot_g_mean_mag", "Gmag"],
        "phot_bp_mean_mag": ["phot_bp_mean_mag", "BPmag"],
        "phot_rp_mean_mag": ["phot_rp_mean_mag", "RPmag"],
    }
    for std, alts in aliases.items():
        for a in alts:
            if a.lower() in lower:
                rename_map[lower[a.lower()]] = std
                break
    df = df.rename(columns=rename_map)
    return df


def apply_quality_cuts(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if (not ENABLE_QUALITY_CUTS) or ("ruwe" not in df.columns):
        return df.reset_index(drop=True).copy(), 0
    ruwe = pd.to_numeric(df["ruwe"], errors="coerce").to_numpy(np.float64)
    keep = np.isfinite(ruwe) & (ruwe <= float(RUWE_MAX))
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
    *,
    use_central_core_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    ra = df["ra"].to_numpy(np.float64)
    dec = df["dec"].to_numpy(np.float64)
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

    if use_central_core_scale:
        ref_idx = np.argsort(np.hypot(x_deg, y_deg))[: min(600, max(80, int(0.10 * len(df))))]
    else:
        ref_idx = np.arange(len(df), dtype=np.int32)

    sig_plx_intr = robust_mad_scale(dplx[ref_idx], floor=FLOOR_PLX)
    sig_pm_intr = robust_mad_scale(np.hypot(dpmra[ref_idx], dpmdec[ref_idx]), floor=FLOOR_PM)

    plx_err = (
        pd.to_numeric(df["parallax_error"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        if "parallax_error" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    pmra_err = (
        pd.to_numeric(df["pmra_error"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        if "pmra_error" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    pmdec_err = (
        pd.to_numeric(df["pmdec_error"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        if "pmdec_error" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )

    sig_plx = np.sqrt(sig_plx_intr**2 + np.maximum(plx_err, 0.0) ** 2).astype(np.float32)
    sig_pm = np.sqrt(sig_pm_intr**2 + 0.5 * (np.maximum(pmra_err, 0.0) ** 2 + np.maximum(pmdec_err, 0.0) ** 2)).astype(
        np.float32
    )

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
        "scale_scope": "central_core" if use_central_core_scale else "whole_cone",
    }
    return score, X, pos2, kin2, aux


def sample_candidate_subset(score: np.ndarray, pos2: np.ndarray, kin2: np.ndarray, cap: int) -> np.ndarray:
    n_total = len(score)
    cap = int(min(max(cap, 80), n_total))

    n_score = max(20, min(int(round(cap * float(CAND_SCORE_RATIO))), cap))
    n_pos = max(15, min(int(round(cap * float(CAND_POS_RATIO))), cap))
    n_kin = max(10, min(cap - n_score - n_pos, cap))

    cand = np.unique(
        np.concatenate(
            [
                np.argsort(score)[:n_score],
                np.argsort(pos2)[:n_pos],
                np.argsort(kin2)[:n_kin],
            ]
        ).astype(np.int32)
    )
    if len(cand) < cap:
        extra = np.argsort(score)[n_score : n_score + (cap - len(cand))].astype(np.int32)
        cand = np.unique(np.concatenate([cand, extra]))
    return cand.astype(np.int32, copy=False)


def refine_center_if_needed(
    df_cone: pd.DataFrame,
    ra0: float,
    dec0: float,
) -> List[Tuple[str, float, float, float]]:
    centers = [("center0", float(ra0), float(dec0), 0.0)]
    if not ENABLE_CENTER_REFINEMENT:
        return centers

    ra = pd.to_numeric(df_cone["ra"], errors="coerce").to_numpy(np.float64)
    dec = pd.to_numeric(df_cone["dec"], errors="coerce").to_numpy(np.float64)
    x0, y0 = compute_tangent_plane(ra, dec, ra0, dec0)
    top_idx = np.argsort(np.hypot(x0, y0))[: min(int(CENTER_REFINE_TOP_R), len(ra))]
    if len(top_idx) < max(32, CENTER_REFINE_K + 5):
        return centers

    P = np.column_stack([x0[top_idx], y0[top_idx]]).astype(np.float32, copy=False)
    tree = cKDTree(P)
    dist, _ = robust_tree_query(tree, P, k=min(int(CENTER_REFINE_K) + 1, len(top_idx)))
    peak_idx = top_idx[int(np.argmin(dist[:, -1]))]
    ra_peak = float(ra[peak_idx])
    dec_peak = float(dec[peak_idx])

    dx, dy = compute_tangent_plane(np.array([ra_peak]), np.array([dec_peak]), ra0, dec0)
    shift_arcmin = float(np.hypot(dx[0], dy[0]) * 60.0)
    if shift_arcmin <= 0.0:
        return centers

    shift_arcmin = min(shift_arcmin, float(CENTER_SHIFT_LIMIT_ARCMIN))
    blend = float(CENTER_SHIFT_BLEND)
    ra1 = float((1.0 - blend) * ra0 + blend * ra_peak)
    dec1 = float((1.0 - blend) * dec0 + blend * dec_peak)
    centers.append(("center1_refined", ra1, dec1, shift_arcmin))
    return centers


def metric_from_sets(pred_set: set, true_set: set) -> Tuple[float, float, float]:
    tp = sum(x in true_set for x in pred_set)
    fp = len(pred_set) - tp
    fn = len(true_set) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


# ============================================================================
# Baseline 1: GMM+BIC
# ============================================================================
def run_gmm_bic(X: np.ndarray, global_N: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    best_bic = float("inf")
    best_gmm = None

    for n_comp in range(1, GMM_N_COMPONENTS_MAX + 1):
        for cv_type in GMM_COVARIANCE_TYPES:
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type=cv_type,
                    random_state=42,
                    reg_covar=1e-5,
                    n_init=2,
                    max_iter=200,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception:
                continue

    if best_gmm is None:
        return np.zeros(len(X), dtype=bool), np.zeros(len(X), dtype=np.float32), {
            "tag": "GMM_FAIL",
            "objective": float("inf"),
            "n_components": 0,
            "cap_used": 0,
            "expected_size": 0,
            "n_base": 0,
        }

    if best_gmm.n_components == 1:
        return np.zeros(len(X), dtype=bool), np.zeros(len(X), dtype=np.float32), {
            "tag": "GMM_DEGENERATE_k=1",
            "objective": float(best_bic),
            "n_components": 1,
            "cap_used": 0,
            "expected_size": len(X),
            "n_base": len(X),
        }

    best_target_score = float("inf")
    target_label = 0
    cov_type = best_gmm.covariance_type

    for j in range(best_gmm.n_components):
        dist = float(np.linalg.norm(best_gmm.means_[j]))
        if cov_type == "full":
            sign, logdet = np.linalg.slogdet(best_gmm.covariances_[j])
            if sign <= 0:
                logdet = 1e6
        elif cov_type == "diag":
            logdet = np.sum(np.log(np.maximum(best_gmm.covariances_[j], 1e-12)))
        else:
            logdet = 0.0
        weight_penalty = float(best_gmm.weights_[j])
        score_j = dist + 0.5 * float(logdet) + 2.0 * weight_penalty
        if score_j < best_target_score:
            best_target_score = score_j
            target_label = j

    probs_all = best_gmm.predict_proba(X)
    probs = probs_all[:, target_label].astype(np.float32)
    base_mask = probs >= float(GMM_PROB_THRESH)

    expected_size = int(best_gmm.weights_[target_label] * len(X))
    cap = max(50, int(expected_size * 1.5))
    cap = min(cap, int(0.05 * global_N))
    cap = min(cap, len(X))

    pred_local = np.zeros(len(X), dtype=bool)
    if int(base_mask.sum()) > cap:
        idx = np.flatnonzero(base_mask)
        idx = idx[np.argsort(probs[idx])[::-1]]
        pred_local[idx[:cap]] = True
    else:
        pred_local = base_mask.copy()

    pack = {
        "tag": f"GMM|k={best_gmm.n_components}|cv={best_gmm.covariance_type[:3]}",
        "objective": float(best_bic),
        "n_components": int(best_gmm.n_components),
        "cap_used": int(cap),
        "expected_size": int(expected_size),
        "n_base": int(base_mask.sum()),
    }
    return pred_local.astype(bool, copy=False), probs.astype(np.float32), pack


# ============================================================================
# Baseline 2: Heuristic Cut
# ============================================================================
def robust_center_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) < 3:
        return np.nanmedian(X, axis=0).astype(np.float32), SIGMA_FLOOR_VEC.copy()
    center = np.nanmedian(X, axis=0).astype(np.float32)
    scale = np.array(
        [max(robust_mad_scale(X[:, i], float(SIGMA_FLOOR_VEC[i])), float(SIGMA_FLOOR_VEC[i])) for i in range(X.shape[1])],
        dtype=np.float32,
    )
    return center, scale


def build_seed_indices(sc: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    top_score = np.argsort(sc)[: min(HC_SEED_TOP_SCORE, len(sc))].astype(np.int32)
    kin_proxy = Xc[:, 2] ** 2 + Xc[:, 3] ** 2 + Xc[:, 4] ** 2
    top_anchor = np.argsort(kin_proxy)[: min(HC_ANCHOR_TOPK, len(sc))].astype(np.int32)
    seed = np.unique(np.concatenate([top_score, top_anchor])).astype(np.int32)
    if len(seed) < min(HC_MIN_MEMBERS, len(sc)):
        seed = np.argsort(sc)[: min(max(HC_MIN_MEMBERS, 12), len(sc))].astype(np.int32)
    return seed


def ellipsoidal_cut_once(Xc: np.ndarray, center: np.ndarray, scale: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    z = (Xc - center[None, :]) / scale[None, :]
    pos_r = np.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2)
    plx_r = np.abs(z[:, 2])
    pm_r = np.sqrt(z[:, 3] ** 2 + z[:, 4] ** 2)
    mask = (pos_r <= sigma) & (plx_r <= sigma) & (pm_r <= sigma)
    hc_metric = np.maximum.reduce([pos_r / sigma, plx_r / sigma, pm_r / sigma]).astype(np.float32)
    return mask.astype(bool, copy=False), hc_metric


def run_heuristic_cut(Xc: np.ndarray, sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n = len(Xc)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32), {"tag": "HC_EMPTY", "objective": 1e9, "n_seed": 0}

    seed_idx = build_seed_indices(sc, Xc)
    center, scale = robust_center_scale(Xc[seed_idx])

    mask = np.zeros(n, dtype=bool)
    hc_metric = np.full(n, np.inf, dtype=np.float32)

    for _ in range(HC_MAX_ITER):
        new_mask, hc_metric = ellipsoidal_cut_once(Xc, center, scale, HC_SIGMA)
        if new_mask.sum() < HC_MIN_MEMBERS:
            break
        fit_idx = np.flatnonzero(new_mask)
        new_center, new_scale = robust_center_scale(Xc[fit_idx])

        if np.array_equal(new_mask, mask):
            mask = new_mask
            center, scale = new_center, new_scale
            break
        mask = new_mask
        center, scale = new_center, new_scale

    if mask.sum() < HC_MIN_MEMBERS:
        mask = np.zeros(n, dtype=bool)
        keep = seed_idx[: min(len(seed_idx), max(HC_MIN_MEMBERS, min(12, n)))]
        mask[keep] = True
        _, hc_metric = ellipsoidal_cut_once(Xc, center, scale, HC_SIGMA)

    support = np.clip(1.0 - hc_metric, 0.0, 1.0).astype(np.float32)
    objective = float(np.mean(sc[mask]) + 0.20 * math.log(1.0 + int(mask.sum()))) if int(mask.sum()) >= HC_MIN_MEMBERS else 1e9
    pack = {
        "tag": f"HC_SINGLE_SIGMA_{HC_SIGMA:.2f}",
        "objective": float(objective),
        "n_seed": int(len(seed_idx)),
        "cap_used": int(len(Xc)),
    }
    return mask.astype(bool, copy=False), support, pack


# ============================================================================
# Baseline 3: HDBSCAN
# ============================================================================
def score_hdb_cluster(Xc: np.ndarray, labels: np.ndarray, probs: np.ndarray, cid: int) -> Tuple[float, int]:
    idx = np.flatnonzero(labels == cid)
    if idx.size == 0:
        return float("inf"), 0
    Xm = Xc[idx]
    center = np.nanmedian(Xm, axis=0)
    dist = float(np.linalg.norm(center))
    mad_sum = float(np.sum([robust_mad_scale(Xm[:, j], 1e-3) for j in range(Xm.shape[1])]))
    mean_prob = float(np.nanmean(probs[idx])) if idx.size > 0 else 0.0
    size = int(idx.size)
    score = dist + 0.35 * mad_sum - 0.25 * math.log1p(size) - 0.60 * mean_prob
    return score, size


def _run_hdbscan_common(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clusterer = HDBSCAN(
        min_cluster_size=HDB_MIN_CLUSTER_SIZE,
        min_samples=HDB_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=HDB_CLUSTER_SELECTION_EPS,
        alpha=HDB_ALPHA,
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(Xc)
    probs = getattr(clusterer, "probabilities_", np.zeros(len(Xc), dtype=np.float32))
    return np.asarray(labels), np.asarray(probs, dtype=np.float32)


def run_hdbscan(Xc: np.ndarray, global_N: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(Xc) < max(HDB_MIN_CLUSTER_SIZE, HDB_MIN_SAMPLES) + 2:
        return np.zeros(len(Xc), dtype=bool), np.zeros(len(Xc), dtype=np.float32), {
            "tag": "HDBSCAN_TOO_SMALL",
            "objective": float("inf"),
            "cap_used": 0,
            "n_clusters": 0,
            "n_base": 0,
        }

    try:
        labels, probs = _run_hdbscan_common(Xc)
    except Exception:
        return np.zeros(len(Xc), dtype=bool), np.zeros(len(Xc), dtype=np.float32), {
            "tag": "HDBSCAN_FAIL",
            "objective": float("inf"),
            "cap_used": 0,
            "n_clusters": 0,
            "n_base": 0,
        }

    cluster_ids = [cid for cid in np.unique(labels) if cid >= 0]
    if len(cluster_ids) == 0:
        return np.zeros(len(Xc), dtype=bool), probs, {
            "tag": "HDBSCAN_NO_CLUSTER",
            "objective": float("inf"),
            "cap_used": 0,
            "n_clusters": 0,
            "n_base": 0,
        }

    ranked = []
    for cid in cluster_ids:
        score, size = score_hdb_cluster(Xc, labels, probs, int(cid))
        ranked.append((score, int(cid), size))
    ranked.sort(key=lambda x: x[0])

    best_score, best_cid, best_size = ranked[0]
    pred_local = labels == best_cid

    cap = min(len(Xc), max(50, min(int(0.05 * global_N), int(best_size * 1.5))))
    if int(pred_local.sum()) > cap:
        idx = np.flatnonzero(pred_local)
        order = idx[np.argsort(probs[idx])[::-1]]
        trimmed = np.zeros(len(Xc), dtype=bool)
        trimmed[order[:cap]] = True
        pred_local = trimmed

    pack = {
        "tag": "HDBSCAN_SHARED_PREPROC_NOLEAK",
        "objective": float(best_score),
        "cap_used": int(cap),
        "n_clusters": int(len(cluster_ids)),
        "n_base": int(best_size),
    }
    return pred_local.astype(bool, copy=False), probs, pack


# ============================================================================
# Unified per-center orchestration
# ============================================================================
def method_core(method: str, X: np.ndarray, score: np.ndarray, pos2: np.ndarray, kin2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    if method == "gmm_bic":
        crop_size = min(len(kin2), GMM_KIN_CROP_MAX)
        threshold = min(GMM_KIN_CROP_THRESHOLD, float(np.sort(kin2)[crop_size - 1]))
        work_mask = kin2 <= threshold
        if int(work_mask.sum()) < 10:
            pred_local = np.zeros(int(work_mask.sum()), dtype=bool)
            support = np.zeros(int(work_mask.sum()), dtype=np.float32)
            pack = {"tag": "GMM_FAIL_EMPTY", "objective": float("inf"), "cap_used": 0, "expected_size": 0, "n_base": 0}
        else:
            pred_local, support, pack = run_gmm_bic(X[work_mask], global_N=len(X))
        full_pred = np.zeros(len(X), dtype=bool)
        full_sup = np.zeros(len(X), dtype=np.float32)
        full_pred[work_mask] = pred_local
        full_sup[work_mask] = support
        cand_idx = np.arange(len(X), dtype=np.int32)
        return cand_idx, full_pred, pack, full_sup

    if method == "heuristic_cut":
        cap = max(CAP_MIN, min(CAP_MAX, int(CAP_A * HC_ANCHOR_TOPK + CAP_B)))
        cand_idx = sample_candidate_subset(score, pos2, kin2, cap)
        Xc = X[cand_idx]
        sc = score[cand_idx]
        pred_local, support_local, pack = run_heuristic_cut(Xc, sc)
        return cand_idx, pred_local, pack, support_local

    if method == "hdbscan":
        cap = max(CAP_MIN, min(CAP_MAX, int(CAP_A * HC_ANCHOR_TOPK + CAP_B)))
        cand_idx = sample_candidate_subset(score, pos2, kin2, cap)
        Xc = X[cand_idx]
        pred_local, support_local, pack = run_hdbscan(Xc, global_N=len(X))
        return cand_idx, pred_local, pack, support_local

    raise ValueError(f"Unknown method: {method}")


def process_single_cluster(
    method: str,
    cluster: str,
    df_cone_raw: pd.DataFrame,
    row1: pd.Series,
    true_ids: np.ndarray,
) -> Tuple[RunResult, Dict[str, Any]]:
    t0 = _now()
    df_cone, n_rej = apply_quality_cuts(df_cone_raw)

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
    if not np.isfinite(plx0):
        plx0 = None
    if not np.isfinite(pmra0):
        pmra0 = None
    if not np.isfinite(pmdec0):
        pmdec0 = None
    if not np.isfinite(rdeg) or rdeg <= 0:
        rdeg = 1.0

    true_set = set(int(x) for x in np.asarray(true_ids, dtype=np.int64))
    if len(df_cone) == 0:
        empty = RunResult(
            cluster=cluster,
            method=method,
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
            objective=float("inf"),
            tag="EMPTY_AFTER_QUALITY_CUT",
        )
        cache = {
            "cluster": cluster,
            "method": method,
            "pred_ids": np.array([], dtype=np.int64),
            "true_ids": np.asarray(sorted(true_set), dtype=np.int64),
            "support": np.array([], dtype=np.float32),
            "center_mode": "center0",
            "center_shift_arcmin": 0.0,
            "cone_df": df_cone.copy(),
            "pred_mask": np.array([], dtype=bool),
            "true_mask": np.array([], dtype=bool),
            "tag": "EMPTY_AFTER_QUALITY_CUT",
        }
        return empty, cache

    sid = pd.to_numeric(df_cone["source_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    true_mask = np.array([int(s) in true_set for s in sid], dtype=bool)

    best_payload = None
    best_objective = float("inf")

    centers = [("center0", float(ra0), float(dec0), 0.0)]
    allow_refine = (method == "heuristic_cut" and ALLOW_CENTER_REFINEMENT_FOR_HEURISTIC) or (method == "hdbscan" and ALLOW_CENTER_REFINEMENT_FOR_HDBSCAN)
    if allow_refine:
        centers = refine_center_if_needed(df_cone, ra0, dec0)

    for center_mode, cra, cdec, cshift in centers:
        use_central_core_scale = True
        score, X, pos2, kin2, aux = preprocess_astrometry(
            df_cone,
            cra,
            cdec,
            rdeg,
            plx0,
            pmra0,
            pmdec0,
            use_central_core_scale=use_central_core_scale,
        )
        cand_idx, pred_local, pack, support = method_core(method, X, score, pos2, kin2)

        pred_global = np.zeros(len(df_cone), dtype=bool)
        pred_global[cand_idx[pred_local]] = True
        pred_ids = sid[pred_global]
        objective = float(pack.get("objective", np.inf))

        if objective < best_objective:
            best_objective = objective
            best_payload = {
                "center_mode": str(center_mode),
                "center_shift_arcmin": float(cshift),
                "pred_global": pred_global.copy(),
                "pred_ids": pred_ids.astype(np.int64),
                "support": np.asarray(support[pred_local], dtype=np.float32) if len(support) == len(pred_local) else np.asarray([], dtype=np.float32),
                "tag": str(pack.get("tag", "")),
                "objective": float(objective),
                "cap_used": float(pack.get("cap_used", np.nan)),
                "n_base": float(pack.get("n_base", np.nan)),
            }

    assert best_payload is not None
    pred_set = set(int(x) for x in best_payload["pred_ids"])
    prec, rec, f1 = metric_from_sets(pred_set, true_set)

    best_res = RunResult(
        cluster=cluster,
        method=method,
        n_cone=int(len(df_cone)),
        n_raw_cone=int(len(df_cone_raw)),
        n_quality_rejected=int(n_rej),
        n_true_in_cone=int(true_mask.sum()),
        n_pred=int(len(pred_set)),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        contam=float(1.0 - prec if len(pred_set) > 0 else 1.0),
        runtime_s=float(_now() - t0),
        center_mode=str(best_payload["center_mode"]),
        center_shift_arcmin=float(best_payload["center_shift_arcmin"]),
        objective=float(best_payload["objective"]),
        tag=str(best_payload["tag"]),
        extra_1=float(best_payload["cap_used"]),
        extra_2=float(best_payload["n_base"]),
    )
    best_cache = {
        "cluster": cluster,
        "method": method,
        "pred_ids": best_payload["pred_ids"],
        "true_ids": np.asarray(sorted(true_set), dtype=np.int64),
        "support": best_payload["support"],
        "center_mode": best_payload["center_mode"],
        "center_shift_arcmin": float(best_payload["center_shift_arcmin"]),
        "cone_df": df_cone.copy(),
        "pred_mask": best_payload["pred_global"].copy(),
        "true_mask": true_mask.copy(),
        "tag": str(best_payload["tag"]),
    }
    return best_res, best_cache


# ============================================================================
# Loading benchmark tables and robust optional M-CTNC results
# ============================================================================
def load_benchmark_tables(data_dir: Path) -> Tuple[Dict[str, pd.Series], Dict[str, np.ndarray]]:
    t1_path = data_dir / "ocfinder_table1.csv"
    t2_path = data_dir / "ocfinder_table2.csv"
    if not t1_path.exists() or not t2_path.exists():
        raise FileNotFoundError("ocfinder_table1.csv / ocfinder_table2.csv not found in data_dir.")

    table1, table2 = pd.read_csv(t1_path), pd.read_csv(t2_path)

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
    if not c2_sid:
        raise KeyError("Table 2 missing source_id / GaiaEDR3.")

    t2 = table2.copy()
    t2["cluster"] = t2[c2_cl].astype(str).str.strip()
    sid2 = pd.to_numeric(t2[c2_sid], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    true_map = {str(cl): np.unique(sid2[g.index][sid2[g.index] > 0]) for cl, g in t2.groupby("cluster")}
    return t1_idx, true_map


def locate_mctnc_candidates(data_dir: Path, user_path: Optional[Path] = None) -> List[Path]:
    cands: List[Path] = []

    if user_path is not None:
        if user_path.exists() and user_path.is_file():
            cands.append(user_path.resolve())

    patterns = [
        "*mctnc*core*.csv",
        "*MCTNC*core*.csv",
        "*results*core*.csv",
        "*benchmark*core*.csv",
        "*M-CTNC*.csv",
        "*mctnc*.csv",
        "*core*.csv",
    ]
    for pat in patterns:
        cands.extend([p.resolve() for p in data_dir.glob(pat) if p.is_file()])
        cands.extend([p.resolve() for p in data_dir.rglob(pat) if p.is_file()])

    uniq = []
    seen = set()
    for p in cands:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)

    # 优先：文件名里同时包含 mctnc/core，且文件非空
    def sort_key(p: Path):
        name = p.name.lower()
        size = p.stat().st_size if p.exists() else 0
        score = 0
        if "mctnc" in name or "m-ctnc" in name:
            score += 10
        if "core" in name:
            score += 6
        if "benchmark" in name:
            score += 3
        if size > 0:
            score += 2
        if "halo" in name or "explor" in name:
            score -= 8
        return (-score, len(str(p)), str(p))

    uniq.sort(key=sort_key)
    return uniq



def normalize_mctnc_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty M-CTNC dataframe.")

    if "cluster" not in df.columns:
        c = get_column_safely(df, ["Cluster", "cluster", "name", "Name"])
        if c:
            df["cluster"] = df[c].astype(str).str.strip()
        else:
            raise KeyError("M-CTNC results csv lacks cluster column.")

    rename = {}
    for std, alts in {
        "precision": ["precision", "prec", "P"],
        "recall": ["recall", "rec", "R"],
        "f1": ["f1", "F1"],
        "runtime_s": ["runtime_s", "time_s", "runtime"],
        "contam": ["contam", "contamination"],
        "tier": ["tier", "performance_tier", "Tier"],
        "mctnc_mode": [
            "mctnc_mode",
            "mode",
            "run_mode",
            "profile",
            "variant",
            "membership_mode",
            "pipeline_mode",
            "result_mode",
            "solution_mode",
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

    out_cols = ["cluster", "precision", "recall", "f1", "runtime_s", "contam"]
    if "tier" in df.columns:
        out_cols.append("tier")
    if "mctnc_mode" in df.columns:
        out_cols.append("mctnc_mode")
    out = df[out_cols].copy()

    out["cluster"] = out["cluster"].astype(str).str.strip()
    out["precision"] = pd.to_numeric(out["precision"], errors="coerce")
    out["recall"] = pd.to_numeric(out["recall"], errors="coerce")
    out["f1"] = pd.to_numeric(out["f1"], errors="coerce")
    out["runtime_s"] = pd.to_numeric(out["runtime_s"], errors="coerce")
    out["contam"] = pd.to_numeric(out["contam"], errors="coerce")
    if "mctnc_mode" in out.columns:
        out["mctnc_mode"] = out["mctnc_mode"].astype(str).str.strip()

    out = out.dropna(subset=["cluster", "precision", "recall", "f1"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("M-CTNC dataframe becomes empty after cleaning.")
    return out


def filter_mctnc_core_mode(df: pd.DataFrame, source_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = df.copy()
    meta: Dict[str, Any] = {
        "mctnc_source": str(source_path) if source_path is not None else "",
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
    }

    if "mctnc_mode" in work.columns:
        mode = work["mctnc_mode"].astype(str).str.strip().str.lower()
        core_mask = mode.str.contains(r"(^|[^a-z])(core|locked[_\- ]?core|core[_\- ]?mode|core[_\- ]?only)([^a-z]|$)", regex=True, na=False)
        halo_mask = mode.str.contains(r"halo|explor", regex=True, na=False)
        meta["core_rows_detected"] = int(core_mask.sum())
        meta["halo_rows_detected"] = int(halo_mask.sum())

        if MCTNC_REQUIRE_CORE_MODE and int(core_mask.sum()) > 0:
            work = work.loc[core_mask].copy()
            meta["filter_policy"] = "explicit_core_mode_filter"
        elif int(core_mask.sum()) == 0 and int((~halo_mask).sum()) > 0 and int(halo_mask.sum()) > 0:
            work = work.loc[~halo_mask].copy()
            meta["filter_policy"] = "excluded_halo_exploration_rows"
        else:
            meta["filter_policy"] = "mode_column_present_but_no_core_subset"

    if work["cluster"].duplicated().any():
        tmp = work.copy()
        if "mctnc_mode" in tmp.columns:
            mode = tmp["mctnc_mode"].astype(str).str.strip().str.lower()
            tmp["_mode_priority"] = 0
            tmp.loc[mode.str.fullmatch(r"core"), "_mode_priority"] = 4
            tmp.loc[mode.str.contains(r"locked[_\- ]?core|core[_\- ]?mode|core[_\- ]?only", regex=True, na=False), "_mode_priority"] = 3
            tmp.loc[mode.str.contains(r"core", regex=True, na=False), "_mode_priority"] = 2
            tmp.loc[mode.str.contains(r"halo|explor", regex=True, na=False), "_mode_priority"] = -2
        else:
            tmp["_mode_priority"] = 0
        tmp["_runtime_key"] = pd.to_numeric(tmp["runtime_s"], errors="coerce").fillna(np.inf)
        tmp["_row_order"] = np.arange(len(tmp))
        tmp = (
            tmp.sort_values(
                ["cluster", "_mode_priority", "_runtime_key", "_row_order"],
                ascending=[True, False, True, True],
            )
            .drop_duplicates("cluster", keep="first")
            .drop(columns=["_mode_priority", "_runtime_key", "_row_order"])
            .reset_index(drop=True)
        )
        work = tmp
        meta["duplicate_policy"] = "cluster_first_after_core_priority_then_runtime"
    else:
        work = work.reset_index(drop=True)

    meta["kept_rows"] = int(len(work))
    meta["kept_cluster_rows"] = int(work["cluster"].nunique())
    return work, meta


def try_read_mctnc_csv(fp: Path) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    try:
        if (not fp.exists()) or (not fp.is_file()):
            return None
        if fp.stat().st_size == 0:
            return None
        df = pd.read_csv(fp)
        df = normalize_mctnc_frame(df)
        df, meta = filter_mctnc_core_mode(df, fp)
        return df, meta
    except (EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError, KeyError):
        return None
    except Exception:
        return None


def read_mctnc_results(
    base_dir: Path,
    data_dir: Path,
    user_path: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Path], Dict[str, Any]]:
    """Read the official M-CTNC CORE benchmark table only.

    Reader policy after the CORE/HALO mix-up fix:
    1) If --mctnc_csv is explicitly provided, read that exact file only.
    2) Otherwise, read base_dir / OFFICIAL_MCTNC_CORE_FILENAME.
    3) If that file is absent, fall back once to data_dir / OFFICIAL_MCTNC_CORE_FILENAME.

    No wildcard scan or heuristic candidate ranking is allowed here, because the
    Part-Five benchmark must be anchored to the official CORE_BENCHMARK release
    table rather than to an arbitrary M-CTNC result export discovered under
    data_dir.
    """
    if user_path is not None:
        candidates = [user_path.resolve()]
    else:
        candidates = []
        primary = (base_dir / OFFICIAL_MCTNC_CORE_FILENAME).resolve()
        fallback = (data_dir / OFFICIAL_MCTNC_CORE_FILENAME).resolve()
        candidates.append(primary)
        if fallback != primary:
            candidates.append(fallback)

    for fp in candidates:
        payload = try_read_mctnc_csv(fp)
        if payload is not None:
            df, meta = payload
            if df is not None and not df.empty:
                meta = dict(meta)
                meta["reader_policy"] = "strict_official_core_table_only"
                meta["official_core_filename"] = OFFICIAL_MCTNC_CORE_FILENAME
                return df, fp, meta

    return None, None, {
        "mctnc_source": "",
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
        "reader_policy": "strict_official_core_table_only",
        "official_core_filename": OFFICIAL_MCTNC_CORE_FILENAME,
    }
# ============================================================================
# Export tables
# ============================================================================
def runs_to_frame(runs: List[RunResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in runs])
    if len(df) > 0:
        df["cache_pipeline_signature"] = CACHE_PIPELINE_SIGNATURE
    return df.sort_values(["method", "cluster"]).reset_index(drop=True)


def build_overall_summary(base_df: pd.DataFrame, mctnc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for method, g in base_df.groupby("method"):
        parts.append(
            {
                "method": method,
                "n_clusters": int(len(g)),
                "mean_f1": float(g["f1"].mean()),
                "median_f1": float(g["f1"].median()),
                "p90_f1": float(g["f1"].quantile(0.90)),
                "mean_precision": float(g["precision"].mean()),
                "mean_recall": float(g["recall"].mean()),
                "mean_contam": float(g["contam"].mean()),
                "median_runtime_s": float(g["runtime_s"].median()),
                "success_frac_f1_ge_0_9": float((g["f1"] >= 0.90).mean()),
            }
        )
    if mctnc_df is not None:
        g = mctnc_df.copy()
        parts.append(
            {
                "method": "mctnc",
                "n_clusters": int(len(g)),
                "mean_f1": float(g["f1"].mean()),
                "median_f1": float(g["f1"].median()),
                "p90_f1": float(g["f1"].quantile(0.90)),
                "mean_precision": float(g["precision"].mean()),
                "mean_recall": float(g["recall"].mean()),
                "mean_contam": float(g["contam"].mean()),
                "median_runtime_s": float(pd.to_numeric(g["runtime_s"], errors="coerce").median()),
                "success_frac_f1_ge_0_9": float((g["f1"] >= 0.90).mean()),
            }
        )
    return pd.DataFrame(parts).sort_values("mean_f1", ascending=False).reset_index(drop=True)


def add_tier_column(comp_df: pd.DataFrame, mctnc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = comp_df.copy()
    if mctnc_df is not None and "tier" in mctnc_df.columns:
        tier_map = mctnc_df.set_index("cluster")["tier"].to_dict()
        out["tier"] = out["cluster"].map(tier_map)
        return out
    if "mctnc_f1" in out.columns:
        f = out["mctnc_f1"].to_numpy(float)
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


def build_comparison_matrix(base_df: pd.DataFrame, mctnc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    metric_cols = ["precision", "recall", "f1", "contam", "runtime_s"]
    out = None
    for method, g in base_df.groupby("method"):
        g2 = g[["cluster"] + metric_cols].copy()
        g2 = g2.rename(columns={c: f"{method}_{c}" for c in metric_cols})
        out = g2 if out is None else out.merge(g2, on="cluster", how="outer")
    if mctnc_df is not None:
        g = mctnc_df[["cluster"] + metric_cols].copy()
        g = g.rename(columns={c: f"mctnc_{c}" for c in metric_cols})
        out = out.merge(g, on="cluster", how="left")
    out = add_tier_column(out, mctnc_df)
    return out.sort_values("cluster").reset_index(drop=True)


def build_winloss_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    if "mctnc_f1" not in comp_df.columns:
        return pd.DataFrame()
    parts = []
    for method in BASELINE_METHODS:
        col = f"{method}_f1"
        d = pd.to_numeric(comp_df["mctnc_f1"], errors="coerce") - pd.to_numeric(comp_df[col], errors="coerce")
        parts.append(
            {
                "baseline": method,
                "mctnc_better_n": int((d > 0.01).sum()),
                "tie_n": int((np.abs(d) <= 0.01).sum()),
                "baseline_better_n": int((d < -0.01).sum()),
                "median_delta_f1_mctnc_minus_baseline": float(np.nanmedian(d)),
                "p90_delta_f1_mctnc_minus_baseline": float(np.nanquantile(d, 0.90)),
            }
        )
    return pd.DataFrame(parts).sort_values("median_delta_f1_mctnc_minus_baseline", ascending=False).reset_index(drop=True)




def build_tier_summary(comp_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tier, g in comp_df.groupby("tier", dropna=False):
        row = {"tier": tier, "n_clusters": int(len(g))}
        for method in BASELINE_METHODS + (["mctnc"] if "mctnc_f1" in g.columns else []):
            for metric in ["f1", "precision", "recall", "contam", "runtime_s"]:
                col = f"{method}_{metric}"
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors="coerce")
                    row[f"{method}_mean_{metric}"] = float(vals.mean())
                    row[f"{method}_median_{metric}"] = float(vals.median())
        parts.append(row)
    out = pd.DataFrame(parts)
    if out.empty:
        return out
    out["tier_order"] = out["tier"].map({name: i for i, name in enumerate(TIER_ORDER)}).fillna(len(TIER_ORDER))
    out = out.sort_values(["tier_order", "tier"]).drop(columns=["tier_order"]).reset_index(drop=True)
    return out


def build_method_rank_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    if comp_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "best_n",
                "solo_best_n",
                "rank1_n",
                "rank2_n",
                "rank3_n",
                "rank4_n",
                "mean_dense_rank",
                "median_dense_rank",
            ]
        )

    rank_methods = BASELINE_METHODS + (["mctnc"] if "mctnc_f1" in comp_df.columns else [])
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
            "method": m,
            "best_n": int(best_count[m]),
            "solo_best_n": int(solo_best_count[m]),
            "mean_dense_rank": float(np.mean(rank_store[m])) if rank_store[m] else np.nan,
            "median_dense_rank": float(np.median(rank_store[m])) if rank_store[m] else np.nan,
        }
        for r in range(1, len(rank_methods) + 1):
            entry[f"rank{r}_n"] = int(dense_rank_count[m][r])
        entry["rank1_n"] = entry["best_n"]
        rows.append(entry)

    out = pd.DataFrame(rows)
    out = out.sort_values(["mean_dense_rank", "method"]).reset_index(drop=True)
    return out


def build_topcase_table(comp_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if "mctnc_f1" not in comp_df.columns:
        return pd.DataFrame()
    tmp = comp_df.copy()
    base_stack = np.column_stack([pd.to_numeric(tmp[f"{m}_f1"], errors="coerce").to_numpy(float) for m in BASELINE_METHODS])
    best_idx = np.nanargmax(base_stack, axis=1)
    tmp["best_baseline"] = [BASELINE_METHODS[i] for i in best_idx]
    tmp["best_baseline_f1"] = np.nanmax(base_stack, axis=1)
    tmp["delta_mctnc_minus_best"] = pd.to_numeric(tmp["mctnc_f1"], errors="coerce") - tmp["best_baseline_f1"]
    keep = ["cluster", "tier", "mctnc_f1", "best_baseline", "best_baseline_f1", "delta_mctnc_minus_best"]
    for m in BASELINE_METHODS:
        keep.append(f"{m}_f1")
    return tmp.sort_values("delta_mctnc_minus_best", ascending=False)[keep].head(top_n).reset_index(drop=True)


def build_mctnc_import_audit_table(meta: Dict[str, Any], mctnc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    row = {
        "mctnc_source": meta.get("mctnc_source", ""),
        "mctnc_mode_column": meta.get("mctnc_mode_column", ""),
        "core_mode_required": int(meta.get("core_mode_required", 0)),
        "raw_rows": int(meta.get("raw_rows", 0)),
        "kept_rows": int(meta.get("kept_rows", 0)),
        "raw_cluster_rows": int(meta.get("raw_cluster_rows", 0)),
        "kept_cluster_rows": int(meta.get("kept_cluster_rows", 0)),
        "core_rows_detected": int(meta.get("core_rows_detected", 0)),
        "halo_rows_detected": int(meta.get("halo_rows_detected", 0)),
        "filter_policy": str(meta.get("filter_policy", "")),
        "duplicate_policy": str(meta.get("duplicate_policy", "")),
        "usable_after_import": int(mctnc_df is not None and not mctnc_df.empty),
    }
    return pd.DataFrame([row])
def build_fairness_audit_table() -> pd.DataFrame:
    rows = [
        {
            "method": "gmm_bic",
            "shared_quality_cut": 1,
            "shared_catalog_center": 1,
            "shared_astrometric_normalization": 1,
            "shared_candidate_protocol": 0,
            "shared_center_refinement": 0,
            "label_based_model_selection": 0,
            "prediction_cap": 1,
            "kinematic_crop": 1,
            "notes": "GMM+BIC baseline with shared normalization; downstream kinematic crop and posterior cap retained.",
        },
        {
            "method": "heuristic_cut",
            "shared_quality_cut": 1,
            "shared_catalog_center": 1,
            "shared_astrometric_normalization": 1,
            "shared_candidate_protocol": 1,
            "shared_center_refinement": 0,
            "label_based_model_selection": 0,
            "prediction_cap": 0,
            "kinematic_crop": 0,
            "notes": "Rule-based weak baseline evaluated under the shared candidate subset used in the benchmark package.",
        },
        {
            "method": "hdbscan",
            "shared_quality_cut": 1,
            "shared_catalog_center": 1,
            "shared_astrometric_normalization": 1,
            "shared_candidate_protocol": 1,
            "shared_center_refinement": int(ALLOW_CENTER_REFINEMENT_FOR_HDBSCAN),
            "label_based_model_selection": 0,
            "prediction_cap": 1,
            "kinematic_crop": 0,
            "notes": "Primary strong baseline in the paper: HDBSCAN under the shared benchmark protocol, with no truth-guided selection.",
        },
    ]
    return pd.DataFrame(rows)


def try_load_method_cache(cache_csv: Path, method: str, expected_clusters: List[str]) -> Optional[pd.DataFrame]:
    if (not REUSE_COMPLETED_METHODS) or (not cache_csv.exists()):
        return None
    try:
        df = pd.read_csv(cache_csv)
    except Exception:
        return None

    required = {"cluster", "method", "precision", "recall", "f1", "contam", "runtime_s", "tag"}
    if not required.issubset(df.columns):
        return None
    if set(df["cluster"].astype(str)) != set(expected_clusters):
        return None
    if set(df["method"].astype(str)) != {method}:
        return None

    if "cache_pipeline_signature" in df.columns:
        sigs = set(df["cache_pipeline_signature"].astype(str))
        if sigs != {CACHE_PIPELINE_SIGNATURE}:
            return None

    if method == "hdbscan":
        valid_prefixes = (
            "HDBSCAN_SHARED_PREPROC_NOLEAK",
            "HDBSCAN_FAIL",
            "HDBSCAN_NO_CLUSTER",
            "HDBSCAN_TOO_SMALL",
        )
        tags = tuple(str(x) for x in df["tag"].astype(str).tolist())
        if not all(t.startswith(valid_prefixes) for t in tags):
            return None

    return df.sort_values(["method", "cluster"]).reset_index(drop=True)


# ============================================================================
# Figures
# ============================================================================
def savefig(fig: plt.Figure, out: Path) -> None:
    ensure_dir(out.parent)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def find_reusable_cache(data_dir: Path, current_cache: Path, method: str, expected_clusters: List[str]) -> Optional[Tuple[pd.DataFrame, Path]]:
    local_df = try_load_method_cache(current_cache, method, expected_clusters)
    if local_df is not None:
        return local_df, current_cache

    pattern = f"part5_baseline_package_*/tables/cache_{method}_cluster_results.csv"
    candidates = [p for p in data_dir.glob(pattern) if p.resolve() != current_cache.resolve()]
    candidates = sorted(candidates, key=lambda x: x.stat().st_mtime if x.exists() else 0.0, reverse=True)
    for fp in candidates:
        df = try_load_method_cache(fp, method, expected_clusters)
        if df is not None:
            return df, fp
    return None



def plot_overall_method_boxplot(summary_df: pd.DataFrame, base_df: pd.DataFrame, out: Path, mctnc_df: Optional[pd.DataFrame] = None) -> None:
    methods = [m for m in summary_df["method"].tolist()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"), ("runtime_s", "Runtime (s)")]

    for ax, (col, lab) in zip(axes.ravel(), panels):
        data, labels = [], []
        for method in methods:
            if method == "mctnc":
                if mctnc_df is None or col not in mctnc_df.columns:
                    continue
                vals = pd.to_numeric(mctnc_df[col], errors="coerce").dropna().to_numpy(float)
            else:
                vals = pd.to_numeric(base_df.loc[base_df["method"] == method, col], errors="coerce").dropna().to_numpy(float)
            if vals.size > 0:
                data.append(vals)
                labels.append(method)
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(lab)
        ax.grid(True, alpha=0.25)
        if col == "runtime_s":
            ax.set_yscale("log")

    fig.suptitle("Population distributions under the unified CORE benchmark", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    savefig(fig, out)


def plot_overall_summary_bar(summary_df: pd.DataFrame, out: Path) -> None:
    df = summary_df.copy()
    order = list(df.sort_values("mean_f1", ascending=True)["method"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.barh(order, df.set_index("method").loc[order, "mean_f1"])
    for i, m in enumerate(order):
        v = float(df.set_index("method").loc[m, "mean_f1"])
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Mean F1")
    ax.set_title("Overall accuracy ranking")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1]
    rt = df.set_index("method").loc[order, "median_runtime_s"].to_numpy(float)
    ax.barh(order, rt)
    for i, v in enumerate(rt):
        ax.text(v * 1.01 if np.isfinite(v) and v > 0 else 0.01, i, f"{v:.2f}", va="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Median runtime (s)")
    ax.set_title("Runtime ranking")
    ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Population-level summary of the baseline suite", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    savefig(fig, out)


def plot_winloss_vs_mctnc(comp_df: pd.DataFrame, out: Path) -> None:
    if "mctnc_f1" not in comp_df.columns:
        return
    methods = BASELINE_METHODS
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, method in zip(axes, methods):
        d = pd.to_numeric(comp_df["mctnc_f1"], errors="coerce") - pd.to_numeric(comp_df[f"{method}_f1"], errors="coerce")
        vals = np.sort(d.to_numpy(float))
        ax.plot(np.arange(1, len(vals) + 1), vals, lw=2)
        ax.axhline(0.0, ls="--", lw=1.2)
        ax.axhline(0.01, ls=":", lw=1.0)
        ax.axhline(-0.01, ls=":", lw=1.0)
        ax.set_title(f"M-CTNC - {method}")
        ax.set_xlabel("Cluster rank")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\Delta F_1$")
    fig.suptitle("Cluster-wise M-CTNC advantage over each baseline", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    savefig(fig, out)



def plot_tier_stratified(comp_df: pd.DataFrame, out: Path) -> None:
    methods = BASELINE_METHODS + (["mctnc"] if "mctnc_f1" in comp_df.columns else [])
    present_tiers = [t for t in TIER_ORDER if t in set(comp_df["tier"].astype(str))]
    rows = []
    for tier in present_tiers:
        sub = comp_df.loc[comp_df["tier"] == tier]
        for method in methods:
            col = f"{method}_f1"
            if col in sub.columns:
                rows.append(
                    {
                        "tier": tier,
                        "method": method,
                        "mean_f1": float(pd.to_numeric(sub[col], errors="coerce").mean()),
                        "n_clusters": int(len(sub)),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(present_tiers))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))
    for off, method in zip(offsets, methods):
        vals = [
            float(df.loc[(df["tier"] == t) & (df["method"] == method), "mean_f1"].iloc[0])
            if not df.loc[(df["tier"] == t) & (df["method"] == method)].empty
            else np.nan
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
    ax.set_title("Tier-stratified comparison under the unified benchmark")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    savefig(fig, out)

def plot_contam_recall_frontier(summary_df: pd.DataFrame, out: Path) -> None:
    df = summary_df.copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(df["mean_contam"], df["mean_recall"], s=120)
    for _, row in df.iterrows():
        ax.text(float(row["mean_contam"]) + 0.003, float(row["mean_recall"]) + 0.003, str(row["method"]), fontsize=11)
    ax.set_xlabel("Mean contamination")
    ax.set_ylabel("Mean recall")
    ax.set_title("Contamination-recall operating frontier")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, out)


def plot_runtime_accuracy(summary_df: pd.DataFrame, out: Path) -> None:
    df = summary_df.copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(df["median_runtime_s"], df["mean_f1"], s=120)
    for _, row in df.iterrows():
        ax.text(
            float(row["median_runtime_s"]) * 1.03 if float(row["median_runtime_s"]) > 0 else 0.01,
            float(row["mean_f1"]) + 0.002,
            str(row["method"]),
            fontsize=11,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median runtime (s)")
    ax.set_ylabel("Mean F1")
    ax.set_title("Accuracy-efficiency trade-off")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, out)


def plot_fairness_audit_matrix(audit_df: pd.DataFrame, out: Path) -> None:
    cols = [
        "shared_quality_cut",
        "shared_catalog_center",
        "shared_astrometric_normalization",
        "shared_candidate_protocol",
        "shared_center_refinement",
        "label_based_model_selection",
        "prediction_cap",
        "kinematic_crop",
    ]
    labels = [
        "quality cut",
        "catalog center",
        "astrometric norm",
        "candidate protocol",
        "center refinement",
        "label-based selection",
        "prediction cap",
        "kinematic crop",
    ]
    mat = audit_df[cols].to_numpy(float)

    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(audit_df)))
    ax.set_yticklabels(audit_df["method"].tolist())
    ax.set_title("Fairness audit of the Part-Five baseline suite")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = "Yes" if mat[i, j] >= 0.5 else "No"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Enabled")
    fig.tight_layout()
    savefig(fig, out)


def plot_top_cases(comp_df: pd.DataFrame, out: Path) -> None:
    if "mctnc_f1" not in comp_df.columns:
        return
    deltas = []
    for _, row in comp_df.iterrows():
        base_best = np.nanmax([row.get(f"{method}_f1", np.nan) for method in BASELINE_METHODS])
        deltas.append(float(row["mctnc_f1"] - base_best))
    work = comp_df.copy()
    work["mctnc_minus_best_baseline"] = deltas
    up = work.sort_values("mctnc_minus_best_baseline", ascending=False).head(TOP_CASES)
    dn = work.sort_values("mctnc_minus_best_baseline", ascending=True).head(TOP_CASES)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for ax, df, ttl in [
        (axes[0], up.iloc[::-1], "Top M-CTNC advantages over the best baseline"),
        (axes[1], dn, "Cases where the best baseline approaches or exceeds M-CTNC"),
    ]:
        ax.barh(df["cluster"], df["mctnc_minus_best_baseline"])
        ax.axvline(0.0, ls="--", lw=1.2)
        ax.set_title(ttl)
        ax.set_xlabel(r"$F_1(\mathrm{M\!-\!CTNC}) - \max(F_1\ \mathrm{baseline})$")
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, out)



def plot_case_panels(case_caches: List[Dict[str, Any]], out: Path) -> None:
    if not case_caches:
        return
    n = len(case_caches)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4.2 * n))
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
        method = cache["method"]
        tier = str(cache.get("tier", ""))
        delta = float(cache.get("delta_mctnc_minus_best", np.nan))
        group = str(cache.get("selection_group", ""))

        x = pd.to_numeric(df["pmra"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df["pmdec"], errors="coerce").to_numpy(float)
        row_ax[0].scatter(x, y, s=4, alpha=0.18, c=cone_color, edgecolors="none")
        row_ax[0].scatter(x[true], y[true], s=12, alpha=0.90, c=truth_color, marker="x")
        row_ax[0].scatter(x[pred], y[pred], s=10, alpha=0.70, c=pred_color, edgecolors="none")
        row_ax[0].set_xlabel("pmRA")
        row_ax[0].set_ylabel("pmDec")
        row_ax[0].set_title(f"{cluster} | {method} | PM plane")
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

        meta = f"{group} | {tier} | ΔF1(M-CTNC-best) = {delta:+.3f}" if tier or group else ""
        if meta:
            row_ax[1].text(
                0.02,
                0.98,
                meta,
                transform=row_ax[1].transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    fig.suptitle("Representative best-baseline case panels", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    savefig(fig, out)

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Part-Five baseline suite for M-CTNC.")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cone_dir", type=str, default=None)
    parser.add_argument("--mctnc_csv", type=str, default=None, help="Optional M-CTNC CORE benchmark results csv.")
    parser.add_argument("--max_clusters", type=int, default=0, help="0 means all clusters.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (base_dir / "data")
    cone_dir = Path(args.cone_dir).resolve() if args.cone_dir else None
    ensure_dir(data_dir)

    out_root = data_dir / f"part5_baseline_package_{VERSION}"
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

    user_mctnc = Path(args.mctnc_csv).resolve() if args.mctnc_csv else None
    mctnc_df, mctnc_path, mctnc_meta = read_mctnc_results(base_dir, data_dir, user_mctnc)

    if mctnc_path is not None:
        print(f"[SYSTEM] usable M-CTNC comparison file = {mctnc_path}")
        print(f"[SYSTEM] M-CTNC import policy = {mctnc_meta.get('filter_policy', 'not_found')}")
    else:
        print("[SYSTEM] no usable M-CTNC comparison file found; running baseline-only package.")

    methods = BASELINE_METHODS
    all_caches: List[Dict[str, Any]] = []
    method_frames: List[pd.DataFrame] = []
    expected_clusters = [cluster for cluster, _ in tasks]

    t_all = _now()
    for mi, method in enumerate(methods, start=1):
        print(f"\n[METHOD {mi}/{len(methods)}] {method}")
        cache_csv = tab_dir / f"cache_{method}_cluster_results.csv"
        cache_hit = find_reusable_cache(data_dir, cache_csv, method, expected_clusters)
        if cache_hit is not None:
            cached_df, cache_src = cache_hit
            print(f"  -> reused cached results for {method}: {cache_src}")
            if cache_src.resolve() != cache_csv.resolve():
                cached_df.to_csv(cache_csv, index=False)
            method_frames.append(cached_df)
            continue

        method_runs: List[RunResult] = []
        for i, (cluster, fp) in enumerate(tasks, start=1):
            df_cone_raw = read_cone_csv(fp)
            res, cache = process_single_cluster(method, cluster, df_cone_raw, t1_idx[cluster], true_map[cluster])
            method_runs.append(res)
            all_caches.append(cache)
            if (i % 25 == 0) or (i == len(tasks)):
                print(f"  -> processed {i}/{len(tasks)} clusters for {method}")

        method_df = runs_to_frame(method_runs)
        method_df.to_csv(cache_csv, index=False)
        method_frames.append(method_df)

    base_df = pd.concat(method_frames, ignore_index=True).sort_values(["method", "cluster"]).reset_index(drop=True)
    summary_df = build_overall_summary(base_df, mctnc_df)
    comp_df = build_comparison_matrix(base_df, mctnc_df)
    winloss_df = build_winloss_table(comp_df)
    audit_df = build_fairness_audit_table()
    mctnc_import_df = build_mctnc_import_audit_table(mctnc_meta, mctnc_df)
    tier_summary_df = build_tier_summary(comp_df)
    rank_df = build_method_rank_table(comp_df)
    topcase_df = build_topcase_table(comp_df, top_n=40)

    base_df.to_csv(tab_dir / "baseline_cluster_results.csv", index=False)
    summary_df.to_csv(tab_dir / "baseline_overall_summary.csv", index=False)
    comp_df.to_csv(tab_dir / "comparison_matrix.csv", index=False)
    winloss_df.to_csv(tab_dir / "mctnc_vs_baseline_winloss.csv", index=False)
    audit_df.to_csv(tab_dir / "fairness_audit_table.csv", index=False)
    mctnc_import_df.to_csv(tab_dir / "mctnc_import_audit_table.csv", index=False)
    tier_summary_df.to_csv(tab_dir / "tier_stratified_summary.csv", index=False)
    rank_df.to_csv(tab_dir / "method_rank_table.csv", index=False)
    topcase_df.to_csv(tab_dir / "top40_mctnc_minus_best_baseline_cases.csv", index=False)

    selected_caches: List[Dict[str, Any]] = []
    representative_case_df = pd.DataFrame()
    if "mctnc_f1" in comp_df.columns:
        tmp = comp_df.copy()
        base_stack = np.column_stack([pd.to_numeric(tmp[f"{m}_f1"], errors="coerce").to_numpy(float) for m in BASELINE_METHODS])
        best_idx = np.nanargmax(base_stack, axis=1)
        tmp["best_baseline"] = [BASELINE_METHODS[i] for i in best_idx]
        tmp["best_baseline_f1"] = np.nanmax(base_stack, axis=1)
        tmp["delta_mctnc_minus_best"] = pd.to_numeric(tmp["mctnc_f1"], errors="coerce") - tmp["best_baseline_f1"]

        core_positive = tmp.loc[
            tmp["tier"].isin(["Perfect Match", "Tier 1 (Near-perfect)", "Tier 2 (Conservative Core)"])
            & (tmp["delta_mctnc_minus_best"] > 0.02)
        ].sort_values("delta_mctnc_minus_best", ascending=False)

        boundary_negative = tmp.loc[
            tmp["tier"].isin(["Tier 3 (Topological Over-expansion)", "Tier 4 (Borderline)"])
            & (tmp["best_baseline"] == "hdbscan")
            & (tmp["delta_mctnc_minus_best"] < -0.02)
        ].sort_values("delta_mctnc_minus_best", ascending=True)

        selected_rows = []
        selected_rows.extend(core_positive.head(2).to_dict("records"))
        selected_rows.extend(boundary_negative.head(2).to_dict("records"))

        if len(selected_rows) < 4:
            used = {r["cluster"] for r in selected_rows}
            fillers = tmp.loc[~tmp["cluster"].isin(list(used))].sort_values("delta_mctnc_minus_best", ascending=False)
            for _, row in fillers.iterrows():
                selected_rows.append(row.to_dict())
                used.add(str(row["cluster"]))
                if len(selected_rows) >= 4:
                    break

        representative_case_df = pd.DataFrame(selected_rows)
        cache_map = {(c["cluster"], c["method"]): c for c in all_caches}
        for _, row in representative_case_df.iterrows():
            cl = str(row["cluster"])
            best_method = str(row["best_baseline"])
            if (cl, best_method) in cache_map:
                payload = dict(cache_map[(cl, best_method)])
                payload["tier"] = str(row["tier"])
                payload["delta_mctnc_minus_best"] = float(row["delta_mctnc_minus_best"])
                payload["selection_group"] = "M-CTNC-favored core case" if float(row["delta_mctnc_minus_best"]) >= 0 else "HDBSCAN-favored boundary case"
                selected_caches.append(payload)

        if not representative_case_df.empty:
            representative_case_df.to_csv(tab_dir / "representative_case_selection.csv", index=False)

    plot_fairness_audit_matrix(audit_df, fig_dir / "Fig00_fairness_audit_matrix.png")
    plot_overall_method_boxplot(summary_df, base_df, fig_dir / "Fig01_baseline_distributions.png", mctnc_df=mctnc_df)
    plot_overall_summary_bar(summary_df, fig_dir / "Fig02_overall_summary.png")
    plot_tier_stratified(comp_df, fig_dir / "Fig03_tier_stratified_comparison.png")
    plot_contam_recall_frontier(summary_df, fig_dir / "Fig04_contam_recall_frontier.png")
    plot_runtime_accuracy(summary_df, fig_dir / "Fig05_runtime_accuracy_tradeoff.png")
    plot_winloss_vs_mctnc(comp_df, fig_dir / "Fig06_mctnc_winloss_vs_baselines.png")
    plot_top_cases(comp_df, fig_dir / "Fig07_top_case_deltas.png")
    plot_case_panels(selected_caches, fig_dir / "Fig08_representative_baseline_cases.png")

    with pd.ExcelWriter(out_root / "part5_baseline_package.xlsx", engine="openpyxl") as writer:
        base_df.to_excel(writer, sheet_name="baseline_cluster_results", index=False)
        summary_df.to_excel(writer, sheet_name="baseline_overall_summary", index=False)
        comp_df.to_excel(writer, sheet_name="comparison_matrix", index=False)
        audit_df.to_excel(writer, sheet_name="fairness_audit", index=False)
        mctnc_import_df.to_excel(writer, sheet_name="mctnc_import_audit", index=False)
        tier_summary_df.to_excel(writer, sheet_name="tier_summary", index=False)
        rank_df.to_excel(writer, sheet_name="method_rank", index=False)
        if not topcase_df.empty:
            topcase_df.to_excel(writer, sheet_name="top40_cases", index=False)
        if not representative_case_df.empty:
            representative_case_df.to_excel(writer, sheet_name="representative_cases", index=False)
        if not winloss_df.empty:
            winloss_df.to_excel(writer, sheet_name="mctnc_vs_baselines", index=False)

    with open(out_root / "README_part5.txt", "w", encoding="utf-8") as f:
        f.write(
            "Unified Part-Five baseline suite for M-CTNC\n"
            "===========================================\n"
            f"Version: {VERSION}\n"
            f"Clusters processed: {len(tasks)}\n"
            f"Methods: {', '.join(methods)}\n"
            f"M-CTNC comparison file: {str(mctnc_path) if mctnc_path is not None else 'NOT PROVIDED / NOT USABLE'}\n"
            "Benchmark policy: shared-preprocessing baseline suite\n"
            "M-CTNC import policy: prefer core-mode rows over halo/exploration rows whenever mode metadata are available\n"
            f"Heuristic center refinement enabled: {ALLOW_CENTER_REFINEMENT_FOR_HEURISTIC}\n"
            f"HDBSCAN center refinement enabled: {ALLOW_CENTER_REFINEMENT_FOR_HDBSCAN}\n"
            f"M-CTNC mode filter policy: {mctnc_meta.get('filter_policy', 'not_found')}\n"
            f"M-CTNC duplicate policy: {mctnc_meta.get('duplicate_policy', 'none')}\n\n"
            "Key exported tables:\n"
            "  tables/baseline_cluster_results.csv\n"
            "  tables/baseline_overall_summary.csv\n"
            "  tables/comparison_matrix.csv\n"
            "  tables/mctnc_vs_baseline_winloss.csv\n"
            "  tables/fairness_audit_table.csv\n"
            "  tables/mctnc_import_audit_table.csv\n"
            "  tables/representative_case_selection.csv\n\n"
            "Key exported figures:\n"
            "  Fig00_fairness_audit_matrix.png\n"
            "  Fig01_baseline_distributions.png\n"
            "  Fig02_overall_summary.png\n"
            "  Fig03_tier_stratified_comparison.png\n"
            "  Fig04_contam_recall_frontier.png\n"
            "  Fig05_runtime_accuracy_tradeoff.png\n"
            "  Fig06_mctnc_winloss_vs_baselines.png\n"
            "  Fig07_top_case_deltas.png\n"
            "  Fig08_representative_baseline_cases.png\n"
        )

    print("\n============================================================")
    print("Part-Five baseline package completed.")
    print(f"Output folder: {out_root}")
    print(f"Elapsed time : {_now() - t_all:.2f} s")
    print(summary_df.to_string(index=False))
    print("============================================================")


if __name__ == "__main__":
    main()

