# -*- coding: utf-8 -*-
"""
M-CTNC CORE_BENCHMARK Sensitivity Analysis (V12: APJS Final Audit)
----------------------------------------------------------------------
核心修复与升级:
1. [V11 核心修复] 灵敏度分析默认与生产版 M-CTNC 保持同构：objective 与 candidate-cap 逻辑均回归正式发布分支。
2. 保留代表性目标的内层参数扫描与外层中心修正测试，但不再将 shift penalty 混入生产版 objective。
3. 提供可选的 analysis branch 开关；只有在显式启用时，才允许 anchor-cap 解耦等“单因子隔离”实验。
4. 重新引入 `apply_local_photometry_cache` 函数，CMD 面板继续通过 source_id 跨表拼接光度缓存。
5. 引入 hashlib 固定散点绘图随机种子，确保图像跨环境 100% 严格复现。
6. 自动生成全局跨样本对比图 (Fig 4-18 至 Fig 4-22)。
"""

from __future__ import annotations

import math
import re
import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, coo_matrix, csgraph, triu as sp_triu

import warnings
warnings.filterwarnings("ignore")

plt.switch_backend("Agg")

# =========================================================
# [全局出版级绘图风格设置]
# =========================================================
plt.rcParams.update({
    'font.size': 12, 
    'axes.linewidth': 1.5, 
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5, 
    'legend.frameon': True, 
    'legend.edgecolor': 'black'
})

# =========================================================
# [1. 分层物理代表测试集 (Representative Targets)]
# =========================================================
SENSITIVITY_TARGETS: Dict[str, List[str]] = {
    "Tier1_Robust": ["UBC1002", "UBC1037", "UBC1190"],        
    "Tier2_Conditional": ["UBC1015", "UBC1049", "UBC1131"],   
    "Tier3_Boundary_Pathological": ["UBC1171", "UBC1265", "UBC1194"] 
}

# =========================================================
# [Baseline CORE_BENCHMARK configuration]
# =========================================================
# =========================================================
# [Analysis branch control]
# production_consistent: 与正式 M-CTNC 发布分支完全同构
# isolated_factor: 仅在明确需要做“单参数隔离解释”时启用
# =========================================================
ANALYSIS_BRANCH = "production_consistent"
ENABLE_ISOLATED_ANCHOR_CAP_DECOUPLING = False
ENABLE_OBJECTIVE_SHIFT_PENALTY = False
OBJECTIVE_SHIFT_WEIGHT = 0.05

BASELINE = {
    "K_LIST": (16, 24, 32),
    "BETA_LIST": (0.45, 0.50, 0.55),
    "SUPPORT_TAU": 0.75,
    "ANCHOR_SUPPORT_TAU": 0.45,
    "MIN_PTS": 5,
    "CAP_MIN": 300,
    "CAP_MAX": 1200,
    "CAP_A": 12,
    "CAP_B": 250,
    "CAP_BACKOFF_RATIO": 0.60,
    "CAND_SCORE_RATIO": 0.60,
    "CAND_POS_RATIO": 0.25,
    "CAND_KIN_RATIO": 0.15,
    "ANCHOR_N_TARGET": 36,
    "ANCHOR_SPLIT_W1": 0.40,
    "ANCHOR_SPLIT_W2": 0.30,
    "ANCHOR_SPREAD_ENABLE": True,
    "ANCHOR_SPREAD_K": 28,
    "W_POS": 1.00,
    "W_PLX": 1.00,
    "W_PM": 1.00,
    "ENABLE_SELF_TUNING": True,
    "K_QUERY_FACTOR": 3,
    "K_QUERY_MIN": 96,
    "FLOOR_PLX": 0.10,
    "FLOOR_PM": 0.20,
    "ENABLE_CENTER_REFINEMENT": True,
    "CENTER_REFINE_TOP_R": 600,
    "CENTER_REFINE_K": 24,
    "CENTER_SHIFT_BLEND": 0.70,
    "CENTER_SHIFT_LIMIT_ARCMIN": 20.0,
    "ENABLE_QUALITY_CUTS": True,
    "RUWE_MAX": 1.6,
    "OBJ_SUPPORT_W": 0.35,
    "OBJ_SIZE_W": 0.40,
}

# =========================================================
# [Sensitivity scan families & ±1 Step Definitions]
# =========================================================
SCAN_FAMILIES = {
    "support_tau": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "beta_shift": [-0.08, -0.04, 0.00, 0.04, 0.08],
    "k_set": [(12, 16, 24), (14, 20, 28), (16, 24, 32), (20, 28, 36), (24, 32, 40)],
    "anchor_n": [24, 30, 36, 42, 48, 54, 60],
}

PERTURBATION_STEPS = {
    "support_tau": {"minus": 0.70, "plus": 0.80},
    "beta_shift": {"minus": -0.04, "plus": 0.04},
    "k_set": {"minus": (14, 20, 28), "plus": (20, 28, 36)},
    "anchor_n": {"minus": 30, "plus": 42},
}

GALLERY_LABELS = {
    "support_tau": {v: f"tau={v:.2f}" for v in SCAN_FAMILIES["support_tau"]},
    "beta_shift": {v: f"beta={v:+.2f}" if v != 0 else "baseline" for v in SCAN_FAMILIES["beta_shift"]},
    "k_set": {v: f"K={v}" for v in SCAN_FAMILIES["k_set"]},
    "anchor_n": {v: f"anchor={v}" for v in SCAN_FAMILIES["anchor_n"]},
}


# =========================================================
# [APJS-level population perturbation audit families]
# =========================================================
POPULATION_PERTURBATIONS = {
    "support_tau": {"loop": "inner", "minus": 0.70, "plus": 0.80},
    "beta_shift": {"loop": "inner", "minus": -0.04, "plus": 0.04},
    "k_set": {"loop": "inner", "minus": (14, 20, 28), "plus": (20, 28, 36)},
    "anchor_n": {"loop": "inner", "minus": 30, "plus": 42},
    "ruwe_max": {"loop": "inner", "minus": 1.5, "plus": 1.7},
    "cap_backoff_ratio": {"loop": "inner", "minus": 0.50, "plus": 0.70},
    "objective_weights": {"loop": "inner", "minus": (0.25, 0.30), "plus": (0.45, 0.50)},
    "candidate_mix": {"loop": "inner", "minus": (0.70, 0.20, 0.10), "plus": (0.50, 0.30, 0.20)},
    "center_blend": {"loop": "outer", "minus": 0.60, "plus": 0.80},
    "center_limit": {"loop": "outer", "minus": 10.0, "plus": 30.0},
}

FAMILY_DISPLAY = {
    "support_tau": r"Support $\tau$",
    "beta_shift": r"$\Delta\beta$",
    "k_set": r"Scale $K$",
    "anchor_n": r"Anchor $N$",
    "ruwe_max": r"RUWE$_{\max}$",
    "cap_backoff_ratio": r"Backoff ratio",
    "objective_weights": r"Objective weights",
    "candidate_mix": r"Candidate mix",
    "center_blend": r"Center blend",
    "center_limit": r"Center limit",
}

# =========================================================
# [Utilities]
# =========================================================
def _safe_float(x: Any) -> float:
    try: return float(x)
    except Exception: return float("nan")

def resolve_column_name(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols_lower:
            return df.columns[cols_lower.index(cand.lower())]
    return None

def infer_cluster_name(path: Path) -> str:
    s = path.stem
    if s.lower().startswith("gaia_cone_"): s = s[len("gaia_cone_") :]
    m = re.match(r"([A-Za-z]+\d+)", s.strip())
    if m: return m.group(1)
    return s.strip()

def discover_cone_files(base_dir: Path, data_dir: Path) -> List[Path]:
    candidates = [data_dir, base_dir, base_dir / "cones", base_dir / "data_cones", base_dir / "data-gaia"]
    seen, dirs = set(), []
    for d in candidates:
        try: d = d.resolve()
        except Exception: pass
        if d.exists() and d.is_dir() and str(d) not in seen:
            seen.add(str(d))
            dirs.append(d)
            
    files: List[Path] = []
    for d in dirs:
        files.extend(list(d.glob("gaia_cone_*.csv")))
        files.extend(list(d.glob("gaia_cone_*.CSV")))
    return sorted({p.resolve() for p in files})

def wrap_ra_coordinate(ra_deg: np.ndarray, ra0_deg: float) -> np.ndarray:
    dra = ra_deg - ra0_deg
    return (dra + 180.0) % 360.0 - 180.0

def compute_tangent_plane(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    dra = wrap_ra_coordinate(ra_deg, ra0_deg)
    x = dra * math.cos(math.radians(dec0_deg))
    y = dec_deg - dec0_deg
    return x.astype(np.float32), y.astype(np.float32)

def compute_robust_mad_scale(x: np.ndarray, floor: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 10: return float(floor)
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    if np.isfinite(s) and s >= floor: return float(s)
    return float(floor)

def format_limits_apjs(x: np.ndarray, q: Tuple[float, float] = (0.01, 0.99), pad: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0: return (-1.0, 1.0)
    lo, hi = np.percentile(x, [q[0] * 100.0, q[1] * 100.0])
    lo, hi = float(lo), float(hi)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo == hi):
        lo, hi = float(np.min(x)), float(np.max(x))
    rng = max(hi - lo, 1e-6)
    return lo - pad * rng, hi + pad * rng

def transform_to_absolute_mag(g_mag: pd.Series, plx_mas: pd.Series) -> np.ndarray:
    g = g_mag.to_numpy(float)
    p = plx_mas.to_numpy(float)
    out = np.full_like(g, np.nan, dtype=float)
    m = (p > 0) & np.isfinite(p) & np.isfinite(g)
    out[m] = g[m] - 10.0 + 5.0 * np.log10(p[m])
    return out

def downsample_idx(n: int, max_n: int, seed: int = 0) -> np.ndarray:
    if n <= max_n: return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)

def get_stable_seed(cluster_name: str) -> int:
    hash_hex = hashlib.md5(cluster_name.encode('utf-8')).hexdigest()
    return int(hash_hex, 16) % (2**32 - 1)

# =========================================================
# [V10 核心修复: 跨表拼接 Photometry Cache]
# =========================================================
def apply_local_photometry_cache(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """
    通过 source_id 从外部缓存文件读取光度数据(G, BP, RP)。
    原始的 cone search 文件通常不包含这三列，必须依赖此函数跨表合并。
    """
    if not cache_path.exists():
        print(f"  -> [WARNING] Photometry cache not found at {cache_path}. CMD will fail.")
        return df
        
    sid_col = resolve_column_name(df, ["source_id", "GaiaEDR3", "gaiaedr3", "SOURCE_ID"])
    if not sid_col: 
        return df
        
    try: 
        cache_df = pd.read_csv(cache_path, dtype={"source_id": str}).set_index("source_id")
    except Exception as e: 
        print(f"  -> [ERROR] Failed to read cache: {e}")
        return df
        
    df_out = df.copy()
    sids_str = df_out[sid_col].astype(str)
    
    # 强制灌入光度列
    for c in ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']:
        if c in cache_df.columns:
            if c not in df_out.columns: 
                df_out[c] = np.nan
            df_out[c] = df_out[c].fillna(sids_str.map(cache_df[c]))
            
    return df_out

# =========================================================
# [Core M-CTNC mechanics]
# =========================================================
def build_adaptive_knn_graph(X: np.ndarray, k_max: int, k_query: int, self_tuning: bool) -> np.ndarray:
    X = np.atleast_2d(X)
    n_samples = len(X)
    if n_samples < 2: return np.empty((n_samples, 0), dtype=np.int32)
        
    k_max_safe = int(min(max(1, int(k_max)), n_samples - 1))
    k_query_safe = int(min(max(int(k_query), k_max_safe + 8), n_samples - 1))
    
    tree = cKDTree(X)
    try: dist, idx = tree.query(X, k=k_query_safe + 1, workers=-1)
    except TypeError: dist, idx = tree.query(X, k=k_query_safe + 1)
        
    dist = dist[:, 1:].astype(np.float32, copy=False)
    idx = idx[:, 1:].astype(np.int32, copy=False)
    
    if not self_tuning: return idx[:, :k_max_safe].copy()
        
    dk = dist[:, k_max_safe - 1].astype(np.float32, copy=False)
    dk[dk <= 0] = np.float32(1e-6)
    dk_j = dk[idx]
    
    denom = np.sqrt(dk[:, None] * dk_j)
    denom[denom <= 0] = np.float32(1e-6)
    
    normalized_dist = dist / denom
    part = np.argpartition(normalized_dist, kth=k_max_safe - 1, axis=1)[:, :k_max_safe]
    row = np.arange(n_samples)[:, None]
    idx_tight = idx[row, part][row, np.argsort(normalized_dist[row, part], axis=1)]
    return idx_tight.astype(np.int32, copy=False)

def extract_tight_components(idx_knn: np.ndarray, beta: float, min_pts: int) -> np.ndarray:
    idx_knn = np.atleast_2d(idx_knn)
    n_samples = len(idx_knn)
    k_neighbors = idx_knn.shape[1] if idx_knn.ndim == 2 else 0
        
    if n_samples == 0 or k_neighbors < 1 or k_neighbors >= n_samples:
        return np.arange(n_samples, dtype=np.int32)
        
    u = np.repeat(np.arange(n_samples, dtype=np.int32), k_neighbors)
    v = idx_knn.reshape(-1)
    A_dir = csr_matrix((np.ones_like(u, dtype=np.int8), (u, v)), shape=(n_samples, n_samples))
    
    adj_tn = A_dir.minimum(A_dir.T).astype(np.int8)
    E_up = sp_triu(adj_tn, k=1).tocoo()
    I, J = E_up.row.astype(np.int32), E_up.col.astype(np.int32)
    
    if len(I) == 0: return np.arange(n_samples, dtype=np.int32)
        
    ctn_matrix = (adj_tn @ adj_tn).tocsr()
    ctn_counts = np.asarray(ctn_matrix[I, J]).ravel().astype(np.int32)
    deg = np.asarray(adj_tn.sum(axis=1)).ravel().astype(np.float32)
    
    denom = np.minimum(deg[I], deg[J])
    denom[denom <= 0] = np.inf
    ctn_ratio = ctn_counts.astype(np.float32) / denom
    
    keep = ctn_ratio >= float(beta)
    if not np.any(keep): return np.arange(n_samples, dtype=np.int32)
        
    A_filtered = coo_matrix((np.ones(keep.sum(), dtype=np.int8), (I[keep], J[keep])), shape=(n_samples, n_samples))
    A_filtered = (A_filtered + A_filtered.T).tocsr()
    _, labels = csgraph.connected_components(A_filtered, directed=False)
    labels = labels.astype(np.int32, copy=False)
    
    cnts = np.bincount(labels, minlength=(labels.max() + 1))
    small = cnts < int(min_pts)
    if np.any(small):
        labels = labels.copy()
        labels[small[labels]] = -1
    return labels

def isolate_target_substructure(labels: np.ndarray, anchor_idx: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, str]:
    valid = labels >= 0
    if not np.any(valid): return np.zeros_like(labels, dtype=bool), "EMPTY_CC"
        
    uniq = np.unique(labels[valid])
    if len(uniq) == 1: return labels == int(uniq[0]), "ONE_CC"
        
    anchor_idx = anchor_idx[(anchor_idx >= 0) & (anchor_idx < len(labels))]
    votes: Dict[int, int] = {}
    for i in anchor_idx:
        cid = int(labels[i])
        if cid >= 0: votes[cid] = votes.get(cid, 0) + 1
            
    if votes:
        best_vote = max(votes.values())
        cands = [cid for cid, v in votes.items() if v == best_vote]
    else:
        cands = [int(c) for c in uniq.tolist()]
        
    best_c, best_s = None, float("inf")
    for cid in cands:
        mask = (labels == cid)
        s = float(np.mean(score[mask])) if np.any(mask) else float("inf")
        if s < best_s: best_s, best_c = s, cid
            
    best_choice = int(best_c if best_c is not None else uniq[0])
    return labels == best_choice, f"CC_ANCHOR={votes.get(best_choice, 0)}"

def preprocess_astrometry(df: pd.DataFrame, ra0: float, dec0: float, rdeg: float, plx0: Optional[float], pmra0: Optional[float], pmdec0: Optional[float], cfg: dict):
    ra, dec = df["ra"].to_numpy(np.float64), df["dec"].to_numpy(np.float64)
    x_deg, y_deg = compute_tangent_plane(ra, dec, ra0, dec0)
    sig_pos = max(0.10, float(rdeg) if np.isfinite(rdeg) and rdeg > 0 else 1.0)
        
    plx = df["parallax"].to_numpy(np.float32)
    pmra = df["pmra"].to_numpy(np.float32)
    pmdec = df["pmdec"].to_numpy(np.float32)
    
    plx0 = float(np.nanmedian(plx)) if plx0 is None or not np.isfinite(plx0) else float(plx0)
    pmra0 = float(np.nanmedian(pmra)) if pmra0 is None or not np.isfinite(pmra0) else float(pmra0)
    pmdec0 = float(np.nanmedian(pmdec)) if pmdec0 is None or not np.isfinite(pmdec0) else float(pmdec0)
        
    dplx = (plx - plx0).astype(np.float32, copy=False)
    dpmra = (pmra - pmra0).astype(np.float32, copy=False)
    dpmdec = (pmdec - pmdec0).astype(np.float32, copy=False)
    
    n_core = min(600, max(80, int(0.10 * len(df))))
    core_idx = np.argsort(np.hypot(x_deg, y_deg))[:n_core]
    
    sig_plx_intr = compute_robust_mad_scale(dplx[core_idx], floor=cfg["FLOOR_PLX"])
    sig_pm_intr = compute_robust_mad_scale(np.hypot(dpmra[core_idx], dpmdec[core_idx]), floor=cfg["FLOOR_PM"])
    
    plx_err = df["parallax_error"].to_numpy(np.float32) if "parallax_error" in df.columns else None
    pmra_err = df["pmra_error"].to_numpy(np.float32) if "pmra_error" in df.columns else None
    pmdec_err = df["pmdec_error"].to_numpy(np.float32) if "pmdec_error" in df.columns else None

    if plx_err is not None: sig_plx = np.sqrt(sig_plx_intr**2 + np.maximum(plx_err, 0.0) ** 2).astype(np.float32)
    else: sig_plx = np.full(len(df), sig_plx_intr, dtype=np.float32)
        
    if pmra_err is not None and pmdec_err is not None:
        sig_pm = np.sqrt(sig_pm_intr**2 + 0.5 * (np.maximum(pmra_err, 0.0) ** 2 + np.maximum(pmdec_err, 0.0) ** 2)).astype(np.float32)
    else:
        sig_pm = np.full(len(df), sig_pm_intr, dtype=np.float32)

    zx, zy = (x_deg / sig_pos).astype(np.float32, copy=False), (y_deg / sig_pos).astype(np.float32, copy=False)
    zplx, zpmra, zpmdec = (dplx / sig_plx).astype(np.float32, copy=False), (dpmra / sig_pm).astype(np.float32, copy=False), (dpmdec / sig_pm).astype(np.float32, copy=False)
    
    pos2 = (zx * zx + zy * zy).astype(np.float32, copy=False)
    kin2 = (zplx * zplx + zpmra * zpmra + zpmdec * zpmdec).astype(np.float32, copy=False)
    
    score = (cfg["W_POS"] * pos2 + cfg["W_PLX"] * (zplx * zplx) + cfg["W_PM"] * (zpmra * zpmra + zpmdec * zpmdec)).astype(np.float32)
    X = np.column_stack([zx, zy, zplx, zpmra, zpmdec]).astype(np.float32, copy=False)
    aux = {"sig_pos_deg": float(sig_pos), "sig_plx_intr": float(sig_plx_intr), "sig_pm_intr": float(sig_pm_intr)}
    
    return score, X, pos2, kin2, aux

def sample_candidates_and_anchors(score: np.ndarray, pos2: np.ndarray, kin2: np.ndarray, X: np.ndarray, cap: int, cfg: dict):
    n_total = len(score)
    cap = int(min(max(cap, 80), n_total))
    
    n_score = max(20, min(int(round(cap * float(cfg["CAND_SCORE_RATIO"]))), cap))
    n_pos = max(15, min(int(round(cap * float(cfg["CAND_POS_RATIO"]))), cap))
    n_kin = max(10, min(cap - n_score - n_pos, cap))
    
    cand = np.unique(np.concatenate([np.argsort(score)[:n_score], np.argsort(pos2)[:n_pos], np.argsort(kin2)[:n_kin]]).astype(np.int32))
    if len(cand) < cap:
        cand = np.unique(np.concatenate([cand, np.argsort(score)[n_score : n_score + (cap - len(cand))].astype(np.int32)]))

    aN = int(min(max(cfg["ANCHOR_N_TARGET"], 12), min(90, len(cand))))
    a_score = max(10, int(round(aN * float(cfg["ANCHOR_SPLIT_W1"]))))
    a_pos = max(8, int(round(aN * float(cfg["ANCHOR_SPLIT_W2"]))))
    a_kin = max(8, aN - a_score - a_pos)

    cs, cp, ck = score[cand], pos2[cand], kin2[cand]
    anc = np.unique(np.concatenate([cand[np.argsort(cs)[:a_score]], cand[np.argsort(cp)[:a_pos]], cand[np.argsort(ck)[:a_kin]]]).astype(np.int32))

    if cfg["ANCHOR_SPREAD_ENABLE"] and len(anc) > cfg["ANCHOR_SPREAD_K"]:
        P = X[anc, :2].astype(np.float32, copy=False)
        c = np.median(P, axis=0)
        dist_to_center = np.sum((P - c) ** 2, axis=1)
        first_sel = int(np.argmin(dist_to_center))
        sel = [first_sel]
        dist_min = np.sum((P - P[first_sel]) ** 2, axis=1)
        for _ in range(1, cfg["ANCHOR_SPREAD_K"]):
            j = int(np.argmax(dist_min))
            sel.append(j)
            dist_min = np.minimum(dist_min, np.sum((P - P[j]) ** 2, axis=1))
        anc = anc[sel]

    return cand.astype(np.int32, copy=False), anc.astype(np.int32, copy=False)

def generate_multiscale_consensus(Xc: np.ndarray, sc: np.ndarray, anchor_local: np.ndarray, cfg: dict):
    Xc = np.atleast_2d(Xc)
    n_xc = len(Xc)
    if n_xc == 0: return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32), "EMPTY"
        
    k_max = int(max(cfg["K_LIST"]))
    k_query = max(int(k_max * cfg["K_QUERY_FACTOR"]), int(cfg["K_QUERY_MIN"]))
    idx_knn_max = build_adaptive_knn_graph(Xc, k_max=k_max, k_query=k_query, self_tuning=bool(cfg["ENABLE_SELF_TUNING"]))

    support = np.zeros(n_xc, dtype=np.float32)
    n_cfg = 0
    
    for beta in cfg["BETA_LIST"]:
        for k in cfg["K_LIST"]:
            k_int = int(k)
            if k_int >= n_xc: continue
            labels = extract_tight_components(idx_knn_max[:, :k_int], float(beta), int(cfg["MIN_PTS"]))
            mask, _ = isolate_target_substructure(labels, anchor_local, sc)
            support += mask.astype(np.float32)
            n_cfg += 1

    if n_cfg == 0:
        pred = np.zeros(n_xc, dtype=bool)
        pred[np.argsort(sc)[: max(cfg["MIN_PTS"], min(30, n_xc))]] = True
        return pred, pred.astype(np.float32), "DEGENERATE"
        
    sup_ratio = support / float(n_cfg)
    pred = sup_ratio >= float(cfg["SUPPORT_TAU"])
    
    if pred.sum() < cfg["MIN_PTS"] and len(anchor_local) > 0:
        pred = sup_ratio >= float(cfg["ANCHOR_SUPPORT_TAU"])
        
    return pred, sup_ratio, f"MS(cfg={n_cfg})|tau={cfg['SUPPORT_TAU']:.2f}"

def evaluate_extraction_objective(
    sc: np.ndarray,
    sup: np.ndarray,
    pred: np.ndarray,
    cap: int,
    min_pts: int,
    cfg: Optional[dict] = None,
    shift_arcmin: float = 0.0,
) -> float:
    """Production-consistent extraction objective.

    Default behavior is identical to the released M-CTNC production pipeline.
    When a configuration dictionary is supplied, the baseline production
    weights remain the defaults unless the user explicitly perturbs them in the
    population-level audit. A center-shift term is *not* part of the production
    objective and is only enabled inside the isolated-factor branch.
    """
    if pred.sum() < min_pts:
        return 1e9

    support_w = 0.35
    size_w = 0.40
    if cfg is not None:
        support_w = float(cfg.get("OBJ_SUPPORT_W", support_w))
        size_w = float(cfg.get("OBJ_SIZE_W", size_w))

    compactness_term = float(np.mean(sc[pred]))
    stability_term = -support_w * float(np.mean(sup[pred]))
    inflation_penalty = size_w * (math.log(1.0 + int(pred.sum())) / math.log(1.0 + cap))
    objective = compactness_term + stability_term + inflation_penalty

    if ANALYSIS_BRANCH == "isolated_factor" and ENABLE_OBJECTIVE_SHIFT_PENALTY:
        objective += float(OBJECTIVE_SHIFT_WEIGHT) * float(shift_arcmin)
    return objective


def compute_candidate_caps(cfg: dict) -> List[int]:
    """Candidate-cap schedule aligned with the production pipeline.

    In the default production-consistent branch, cap remains coupled to
    ANCHOR_N_TARGET exactly as in MCTNC.py. Decoupling is only permitted in the
    isolated analysis branch when explicitly enabled.
    """
    if ANALYSIS_BRANCH == "isolated_factor" and ENABLE_ISOLATED_ANCHOR_CAP_DECOUPLING:
        anchor_for_cap = int(BASELINE["ANCHOR_N_TARGET"])
    else:
        anchor_for_cap = int(cfg["ANCHOR_N_TARGET"])

    cap0 = max(cfg["CAP_MIN"], min(cfg["CAP_MAX"], int(cfg["CAP_A"] * anchor_for_cap + cfg["CAP_B"])))
    cap1 = max(cfg["CAP_MIN"], int(cap0 * cfg["CAP_BACKOFF_RATIO"]))
    return [int(cap0), int(cap1)]


def orchestrate_single_center(
    df_cone: pd.DataFrame, ra0: float, dec0: float, rdeg: float,
    plx0: Optional[float], pmra0: Optional[float], pmdec0: Optional[float],
    cfg: dict, shift_arcmin: float = 0.0
):
    score, X, pos2, kin2, aux = preprocess_astrometry(df_cone, ra0, dec0, rdeg, plx0, pmra0, pmdec0, cfg)
    caps_to_try = compute_candidate_caps(cfg)

    best_J = None
    best_pack = None
    
    for cap in caps_to_try:
        cand_idx, anchor_idx = sample_candidates_and_anchors(score, pos2, kin2, X, int(cap), cfg)
        Xc, sc = X[cand_idx], score[cand_idx]
        
        inv_map = {int(g): i for i, g in enumerate(cand_idx)}
        anchor_local = np.array([inv_map.get(int(a), -1) for a in anchor_idx], dtype=np.int32)
        anchor_local = anchor_local[anchor_local >= 0]

        pred, sup, tag = generate_multiscale_consensus(Xc, sc, anchor_local, cfg)
        J = evaluate_extraction_objective(sc, sup, pred, int(cap), int(cfg["MIN_PTS"]), cfg, shift_arcmin)

        pack = {"cap": int(cap), "n_candidates": len(cand_idx), "n_pred_local": int(pred.sum()), "tag": tag, "objective": float(J)}
        if best_J is None or J < best_J:
            best_J = J
            best_pack = (cand_idx, pred, sup, pack, score, X, pos2, kin2, aux)
            
    return best_pack

def run_cluster_once(cluster: str, df_cone_raw: pd.DataFrame, row1: pd.Series, true_ids: np.ndarray, cfg: dict):
    df_plot = df_cone_raw.copy()
    if cfg["ENABLE_QUALITY_CUTS"] and "ruwe" in df_plot.columns:
        m = df_plot["ruwe"].astype(np.float64) <= float(cfg["RUWE_MAX"])
        m = m.fillna(False) if isinstance(m, pd.Series) else m
        df_plot = df_plot.loc[m].copy()

    ra0 = _safe_float(row1.get("ra0", row1.get("ra", np.nan)))
    dec0 = _safe_float(row1.get("dec0", row1.get("dec", np.nan)))
    rdeg = _safe_float(row1.get("radius", 1.0))
    plx0 = _safe_float(row1.get("plx0", np.nan))
    pmra0 = _safe_float(row1.get("pmra0", np.nan))
    pmdec0 = _safe_float(row1.get("pmdec0", np.nan))
    
    if not np.isfinite(plx0): plx0 = None
    if not np.isfinite(pmra0): pmra0 = None
    if not np.isfinite(pmdec0): pmdec0 = None
    if not np.isfinite(rdeg) or rdeg <= 0: rdeg = 1.0

    sid = df_plot["source_id"].to_numpy(np.int64, copy=False)
    true_set = set(int(x) for x in true_ids)
    true_mask = np.array([int(s) in true_set for s in sid], dtype=bool)

    centers = [("center0", ra0, dec0, 0.0, False)]
    if cfg["ENABLE_CENTER_REFINEMENT"] and len(df_plot) >= max(32, int(cfg["CENTER_REFINE_K"]) + 5):
        ra = df_plot["ra"].to_numpy(np.float64)
        dec = df_plot["dec"].to_numpy(np.float64)
        x0, y0 = compute_tangent_plane(ra, dec, ra0, dec0)
        
        top_mask = np.zeros(len(ra), dtype=bool)
        top_idx = np.argsort(np.hypot(x0, y0))[: min(int(cfg["CENTER_REFINE_TOP_R"]), len(ra))]
        top_mask[top_idx] = True
        idx = np.flatnonzero(top_mask)
        
        if len(idx) >= max(32, int(cfg["CENTER_REFINE_K"]) + 5):
            P = np.column_stack([x0[idx], y0[idx]]).astype(np.float32, copy=False)
            try: dist, _ = cKDTree(P).query(P, k=min(int(cfg["CENTER_REFINE_K"]) + 1, len(idx)), workers=-1)
            except TypeError: dist, _ = cKDTree(P).query(P, k=min(int(cfg["CENTER_REFINE_K"]) + 1, len(idx)))
                
            peak_idx = idx[int(np.argmin(dist[:, -1]))]
            ra_peak = float(ra[peak_idx])
            dec_peak = float(dec[peak_idx])
            
            dx, dy = compute_tangent_plane(np.array([ra_peak]), np.array([dec_peak]), ra0, dec0)
            shift_arcmin = float(np.hypot(float(dx[0]), float(dy[0]))) * 60.0
            clipped = shift_arcmin > float(cfg["CENTER_SHIFT_LIMIT_ARCMIN"])
            
            if clipped:
                scale = float(cfg["CENTER_SHIFT_LIMIT_ARCMIN"]) / max(1e-6, shift_arcmin)
                ra_peak = ra0 + (ra_peak - ra0) * scale
                dec_peak = dec0 + (dec_peak - dec0) * scale
                shift_arcmin = float(cfg["CENTER_SHIFT_LIMIT_ARCMIN"])
                
            a_blend = float(cfg["CENTER_SHIFT_BLEND"])
            cra = ra0 * (1.0 - a_blend) + ra_peak * a_blend
            cdec = dec0 * (1.0 - a_blend) + dec_peak * a_blend
            centers.append(("center1_refined", cra, cdec, shift_arcmin, clipped))

    best_out = None
    best_J = None
    best_meta = None
    
    for cname, cra, cdec, cshift, cclip in centers:
        cand_idx, pred_local, sup, pack, score, X, pos2, kin2, aux = orchestrate_single_center(
            df_plot, cra, cdec, rdeg, plx0, pmra0, pmdec0, cfg, shift_arcmin=cshift
        )
        J = float(pack["objective"]) + (0.25 if cclip and pack["n_pred_local"] > 0.60 * pack["cap"] else 0.0)
        if best_J is None or J < best_J:
            best_J = J
            best_out = (cand_idx, pred_local, sup, pack)
            best_meta = (cname, cra, cdec, cshift, cclip)

    cand_idx, pred_local, sup, pack = best_out
    cname, cra, cdec, cshift, cclip = best_meta
    
    pred_ids = sid[cand_idx[pred_local]]
    pred_set = set(int(x) for x in pred_ids)

    tp = sum(x in true_set for x in pred_set)
    fp = len(pred_set) - tp
    fn = int(true_mask.sum()) - tp
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    result = {
        "cluster": cluster, 
        "n_cone_raw": int(len(df_cone_raw)), 
        "n_cone_work": int(len(df_plot)),
        "n_true": int(true_mask.sum()), 
        "cap": int(pack["cap"]), 
        "n_pred": int(len(pred_set)),
        "precision": float(prec), 
        "recall": float(rec), 
        "f1": float(f1),
        "center_mode": str(cname), 
        "center_shift_arcmin": float(cshift), 
        "objective": float(best_J),
        "pred_mask": np.isin(sid, list(pred_set)), 
        "true_mask": true_mask,
        "ra0_out": float(cra), 
        "dec0_out": float(cdec), 
        "analysis_branch": str(ANALYSIS_BRANCH),
        "objective_mode": "production_consistent" if ANALYSIS_BRANCH == "production_consistent" else "isolated_factor",
        "df_plot": df_plot.copy(),
    }
    return result, {}


# =========================================================
# [实验设计逻辑: Inner-loop / Outer-loop 隔离]
# =========================================================
def print_header(title: str):
    print("-" * 100)
    print(f"[{title}]")
    print("-" * 100)
    print(f"{'setting':>15} {'F1':>8} {'P':>8} {'R':>8} {'N_pred':>8} {'center_mode':>17} {'shift_arcmin':>13} {'objective':>10}")

def print_row(setting: str, res: dict):
    print(f"{setting:>15} {res['f1']:8.4f} {res['precision']:8.4f} {res['recall']:8.4f} {res['n_pred']:8} {res['center_mode']:>17} {res['center_shift_arcmin']:13.4f} {res['objective']:10.4f}")


def make_cfg_with_override(family: str, value: Any) -> dict:
    cfg = dict(BASELINE)
    if family == "support_tau":
        cfg["SUPPORT_TAU"] = float(value)
    elif family == "beta_shift":
        cfg["BETA_LIST"] = tuple(np.round(np.clip(np.array(BASELINE["BETA_LIST"]) + float(value), 0.30, 0.70), 2))
    elif family == "k_set":
        cfg["K_LIST"] = tuple(int(x) for x in value)
    elif family == "anchor_n":
        cfg["ANCHOR_N_TARGET"] = int(value)
    elif family == "ruwe_max":
        cfg["RUWE_MAX"] = float(value)
    elif family == "cap_backoff_ratio":
        cfg["CAP_BACKOFF_RATIO"] = float(value)
    elif family == "objective_weights":
        cfg["OBJ_SUPPORT_W"] = float(value[0]); cfg["OBJ_SIZE_W"] = float(value[1])
    elif family == "candidate_mix":
        cfg["CAND_SCORE_RATIO"] = float(value[0]); cfg["CAND_POS_RATIO"] = float(value[1]); cfg["CAND_KIN_RATIO"] = float(value[2])
    elif family == "center_blend":
        cfg["CENTER_SHIFT_BLEND"] = float(value)
    elif family == "center_limit":
        cfg["CENTER_SHIFT_LIMIT_ARCMIN"] = float(value)
    return cfg

def run_inner_loop_sensitivity(cluster: str, df_raw: pd.DataFrame, row1: pd.Series, true_ids: np.ndarray) -> Tuple[Dict, dict, dict]:
    print(f"\n[INNER-LOOP: Locked Center (center_refine=False)]")
    family_results = {}
    f1_dict = {}
    
    cfg_base = dict(BASELINE)
    cfg_base["ENABLE_CENTER_REFINEMENT"] = False
    res_base, _ = run_cluster_once(cluster, df_raw, row1, true_ids, cfg_base)
    f1_dict["baseline"] = res_base["f1"]
    
    print_header("baseline")
    print_row("baseline", res_base)
    
    for fam, values in SCAN_FAMILIES.items():
        print_header(fam)
        rows = []
        for val in values:
            cfg = make_cfg_with_override(fam, val)
            cfg["ENABLE_CENTER_REFINEMENT"] = False
            res, _ = run_cluster_once(cluster, df_raw, row1, true_ids, cfg)
            label = GALLERY_LABELS.get(fam, {}).get(val, str(val))
            print_row(label, res)
            f1_dict[f"{fam}_{val}"] = res["f1"]
            rows.append({
                "x": val, 
                "label": label, 
                "f1": res["f1"], 
                "precision": res["precision"], 
                "recall": res["recall"],
                "n_pred": res["n_pred"], 
                "pred_mask": res["pred_mask"], 
                "true_mask": res["true_mask"],
                "ra0_out": res["ra0_out"], 
                "dec0_out": res["dec0_out"],
            })
        family_results[fam] = rows
        
    return family_results, res_base, f1_dict

def run_outer_loop_sensitivity(cluster: str, df_raw: pd.DataFrame, row1: pd.Series, true_ids: np.ndarray) -> Tuple[dict, dict]:
    print(f"\n[OUTER-LOOP: Center Refinement Test]")
    print_header("center_refinement (production-consistent objective)")
    
    cfg_off = dict(BASELINE)
    cfg_off["ENABLE_CENTER_REFINEMENT"] = False
    res_off, _ = run_cluster_once(cluster, df_raw, row1, true_ids, cfg_off)
    print_row("center_off", res_off)
    
    cfg_on = dict(BASELINE)
    cfg_on["ENABLE_CENTER_REFINEMENT"] = True
    res_on, _ = run_cluster_once(cluster, df_raw, row1, true_ids, cfg_on)
    print_row("center_on", res_on)
    
    return res_off, res_on


# =========================================================
# [数据加载与完全修复的绘图模块]
# =========================================================
def load_reference_tables(data_dir: Path):
    t1_path = data_dir / "ocfinder_table1.csv"
    t2_path = data_dir / "ocfinder_table2.csv"
    table1 = pd.read_csv(t1_path)
    table2 = pd.read_csv(t2_path)
    
    c1_cl = resolve_column_name(table1, ["Cluster", "cluster", "Name", "name"])
    t1 = table1.copy()
    t1["cluster"] = t1[c1_cl].astype(str).str.strip()
    
    for col, alts in [
        ("ra0", ["RA_ICRS", "ra", "RA"]), 
        ("dec0", ["DE_ICRS", "dec", "DE"]), 
        ("radius", ["r_deg", "radius", "r"]), 
        ("plx0", ["plx", "parallax", "Plx"]), 
        ("pmra0", ["pmra", "pmRA"]), 
        ("pmdec0", ["pmdec", "pmDE"])
    ]:
        col_name = resolve_column_name(t1, alts)
        t1[col] = pd.to_numeric(t1[col_name], errors="coerce") if col_name else np.nan
            
    t1_idx = {str(r["cluster"]): r for _, r in t1.iterrows()}
    
    c2_cl = resolve_column_name(table2, ["Cluster", "cluster", "Name", "name"])
    c2_sid = resolve_column_name(table2, ["source_id", "GaiaEDR3", "gaiaedr3", "SOURCE_ID"])
    t2 = table2.copy()
    
    sid2 = pd.to_numeric(t2[c2_sid], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    t2["sid"] = np.where(sid2 > 0, sid2, -1)
    true_map = {str(cl): np.unique(g["sid"].to_numpy(np.int64)[g["sid"].to_numpy(np.int64) > 0]) for cl, g in t2.groupby(c2_cl)}
    return t1_idx, true_map

def load_cone_dataframe(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    
    for std, alts in [
        ("source_id", ["source_id", "GaiaEDR3", "SOURCE_ID"]), 
        ("ra", ["ra", "RA_ICRS", "RA"]), 
        ("dec", ["dec", "DE_ICRS", "DE"]),
        ("parallax", ["parallax", "plx"]), 
        ("pmra", ["pmra", "pmRA"]), 
        ("pmdec", ["pmdec", "pmDE"]),
        ("parallax_error", ["parallax_error", "e_plx"]), 
        ("pmra_error", ["pmra_error", "e_pmra"]), 
        ("pmdec_error", ["pmdec_error", "e_pmdec"]),
    ]:
        for a in alts:
            if a.lower() in colmap:
                df.rename(columns={colmap[a.lower()]: std}, inplace=True)
                break
                
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").fillna(-1).astype(np.int64)
    return df

@dataclass
class PlotConfig:
    dpi: int = 160
    alpha_all: float = 0.30
    alpha_fn: float = 0.80
    alpha_tp: float = 0.90
    alpha_fp: float = 0.85
    s_all: float = 6.0
    s_fn: float = 40.0
    s_fp: float = 70.0
    s_tp: float = 35.0
    max_points_all: int = 60000

def render_standard_diagnostic_panel(cluster: str, df_plot: pd.DataFrame, pred_mask: np.ndarray, true_mask: np.ndarray, ra0: float, dec0: float, out_png: Path, title: str) -> None:
    cfg = PlotConfig()
    stable_seed = get_stable_seed(cluster)
    
    mask_tp = pred_mask & true_mask
    mask_fn = (~pred_mask) & true_mask
    mask_fp = pred_mask & (~true_mask)
    
    bg_idx = downsample_idx(len(df_plot), cfg.max_points_all, seed=stable_seed)
    bg = df_plot.iloc[bg_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=cfg.dpi)
    ax_pm = axes[0, 0]
    ax_sky = axes[0, 1]
    ax_cmd = axes[1, 0]
    ax_plx = axes[1, 1]
    
    c_ra = resolve_column_name(df_plot, ["ra", "RA"])
    c_dec = resolve_column_name(df_plot, ["dec", "DE"])
    c_pmra = resolve_column_name(df_plot, ["pmra"])
    c_pmdec = resolve_column_name(df_plot, ["pmdec"])

    if c_pmra and c_pmdec:
        ax_pm.scatter(bg[c_pmra], bg[c_pmdec], s=cfg.s_all, alpha=cfg.alpha_all, c="lightgray", zorder=1)
        if mask_fn.sum() > 0: 
            ax_pm.scatter(df_plot.loc[mask_fn, c_pmra], df_plot.loc[mask_fn, c_pmdec], s=cfg.s_fn, alpha=cfg.alpha_fn, c="dodgerblue", marker="x", linewidths=1.5, zorder=4, label="FN")
        if mask_fp.sum() > 0: 
            ax_pm.scatter(df_plot.loc[mask_fp, c_pmra], df_plot.loc[mask_fp, c_pmdec], s=cfg.s_fp, alpha=cfg.alpha_fp, c="crimson", marker="*", edgecolor="white", linewidths=0.5, zorder=5, label="FP")
        if mask_tp.sum() > 0: 
            ax_pm.scatter(df_plot.loc[mask_tp, c_pmra], df_plot.loc[mask_tp, c_pmdec], s=cfg.s_tp, alpha=cfg.alpha_tp, c="darkorange", marker="o", edgecolor="black", linewidths=0.5, zorder=6, label="TP")
        ax_pm.set_xlabel(r"$\mu_{\alpha}^*$ (mas/yr)")
        ax_pm.set_ylabel(r"$\mu_{\delta}$ (mas/yr)")
        ax_pm.legend(loc="upper left", fontsize=10)

    if c_ra and c_dec:
        x_a, y_a = compute_tangent_plane(bg[c_ra].to_numpy(), bg[c_dec].to_numpy(), ra0, dec0)
        x_tp, y_tp = compute_tangent_plane(df_plot.loc[mask_tp, c_ra].to_numpy(), df_plot.loc[mask_tp, c_dec].to_numpy(), ra0, dec0)
        x_fn, y_fn = compute_tangent_plane(df_plot.loc[mask_fn, c_ra].to_numpy(), df_plot.loc[mask_fn, c_dec].to_numpy(), ra0, dec0)
        x_fp, y_fp = compute_tangent_plane(df_plot.loc[mask_fp, c_ra].to_numpy(), df_plot.loc[mask_fp, c_dec].to_numpy(), ra0, dec0)
        
        ax_sky.scatter(x_a, y_a, s=cfg.s_all, alpha=cfg.alpha_all, c="lightgray", zorder=1)
        if mask_fn.sum() > 0: 
            ax_sky.scatter(x_fn, y_fn, s=cfg.s_fn, alpha=cfg.alpha_fn, c="dodgerblue", marker="x", linewidths=1.5, zorder=4)
        if mask_fp.sum() > 0: 
            ax_sky.scatter(x_fp, y_fp, s=cfg.s_fp, alpha=cfg.alpha_fp, c="crimson", marker="*", edgecolor="white", linewidths=0.5, zorder=5)
        if mask_tp.sum() > 0: 
            ax_sky.scatter(x_tp, y_tp, s=cfg.s_tp, alpha=cfg.alpha_tp, c="darkorange", marker="o", edgecolor="black", linewidths=0.5, zorder=6)
        ax_sky.set_xlabel(r"$\Delta \alpha \cdot \cos(\delta)$ (deg)")
        ax_sky.set_ylabel(r"$\Delta \delta$ (deg)")
        ax_sky.invert_xaxis()

    # [V10] 彻底修复测光列别名映射，保证画出 CMD
    c_g = resolve_column_name(df_plot, ["phot_g_mean_mag", "gmag", "Gmag", "phot_g_mean_flux"])
    c_bp = resolve_column_name(df_plot, ["phot_bp_mean_mag", "bpmag", "BPmag"])
    c_rp = resolve_column_name(df_plot, ["phot_rp_mean_mag", "rpmag", "RPmag"])
    c_plx = resolve_column_name(df_plot, ["parallax", "plx", "Plx"])

    if c_g and c_bp and c_rp:
        C_tp = df_plot.loc[mask_tp, c_bp].to_numpy(float) - df_plot.loc[mask_tp, c_rp].to_numpy(float)
        C_fn = df_plot.loc[mask_fn, c_bp].to_numpy(float) - df_plot.loc[mask_fn, c_rp].to_numpy(float)
        C_fp = df_plot.loc[mask_fp, c_bp].to_numpy(float) - df_plot.loc[mask_fp, c_rp].to_numpy(float)
        
        if c_plx:
            M_tp = transform_to_absolute_mag(df_plot.loc[mask_tp, c_g], df_plot.loc[mask_tp, c_plx])
            M_fn = transform_to_absolute_mag(df_plot.loc[mask_fn, c_g], df_plot.loc[mask_fn, c_plx])
            M_fp = transform_to_absolute_mag(df_plot.loc[mask_fp, c_g], df_plot.loc[mask_fp, c_plx])
            ylab = "Absolute $M_G$ (mag)"
        else:
            M_tp = df_plot.loc[mask_tp, c_g].to_numpy(float)
            M_fn = df_plot.loc[mask_fn, c_g].to_numpy(float)
            M_fp = df_plot.loc[mask_fp, c_g].to_numpy(float)
            ylab = "Apparent G (mag)"
            
        if mask_fn.sum() > 0: 
            ax_cmd.scatter(C_fn, M_fn, s=cfg.s_fn, alpha=cfg.alpha_fn, c="dodgerblue", marker="x", linewidths=1.5, zorder=4)
        if mask_fp.sum() > 0: 
            ax_cmd.scatter(C_fp, M_fp, s=cfg.s_fp, alpha=cfg.alpha_fp, c="crimson", marker="*", edgecolor="white", linewidths=0.5, zorder=5)
        if mask_tp.sum() > 0: 
            ax_cmd.scatter(C_tp, M_tp, s=cfg.s_tp, alpha=cfg.alpha_tp, c="darkorange", marker="o", edgecolor="black", linewidths=0.5, zorder=6)
            
        ax_cmd.set_xlabel(r"Color $(G_{BP}-G_{RP})$ (mag)")
        ax_cmd.set_ylabel(ylab)
        ax_cmd.invert_yaxis()
    else:
        ax_cmd.set_axis_off()
        ax_cmd.text(0.5, 0.5, "CMD Unavailable\n(Photometry missing in cache)", ha='center', va='center', fontsize=14)

    if c_plx:
        p_all = df_plot[c_plx].to_numpy(float)
        p_all = p_all[np.isfinite(p_all)]
        if len(p_all) > 0:
            bins = np.linspace(np.quantile(p_all, 0.01), np.quantile(p_all, 0.99), 40)
        else:
            bins = 40
            
        ax_plx.hist(p_all, bins=bins, alpha=0.35, color="lightgray", label="Field", zorder=1)
        if mask_fn.sum() > 0: 
            ax_plx.hist(df_plot.loc[mask_fn, c_plx].dropna(), bins=bins, alpha=0.80, color="dodgerblue", label="FN", zorder=4)
        if mask_fp.sum() > 0: 
            ax_plx.hist(df_plot.loc[mask_fp, c_plx].dropna(), bins=bins, alpha=0.85, color="crimson", label="FP", zorder=5)
        if mask_tp.sum() > 0: 
            ax_plx.hist(df_plot.loc[mask_tp, c_plx].dropna(), bins=bins, alpha=0.90, color="darkorange", label="TP", zorder=6)
            
        ax_plx.set_xlabel("Parallax (mas)")
        ax_plx.set_ylabel("Count")
        ax_plx.legend(loc="upper right", fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.95)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_sensitivity_curves(cluster: str, family_results: Dict[str, List[dict]], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=160)
    axes = axes.ravel()

    families = ["support_tau", "beta_shift", "k_set", "anchor_n"]
    titles = {
        "support_tau": r"Support threshold $\tau$", 
        "beta_shift": r"Global $\beta$ offset", 
        "k_set": r"Neighborhood scale set $K$", 
        "anchor_n": r"Anchor number $N_{\rm anchor}$"
    }
    
    for ax, fam in zip(axes, families):
        rows = family_results[fam]
        x = [r["x"] for r in rows]
        f1 = [r["f1"] for r in rows]
        prec = [r["precision"] for r in rows]
        rec = [r["recall"] for r in rows]
        npred = [r["n_pred"] for r in rows]
        
        ax.plot(range(len(x)), f1, marker="o", linewidth=2, label="F1")
        ax.plot(range(len(x)), prec, marker="s", linewidth=2, label="Precision")
        ax.plot(range(len(x)), rec, marker="^", linewidth=2, label="Recall")
        ax2 = ax.twinx()
        ax2.plot(range(len(x)), npred, marker="d", linestyle="--", linewidth=1.8, label=r"$N_{\rm pred}$")
        
        ax.set_ylim(0, 1.05)
        ax.set_title(titles[fam], fontsize=12)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([str(v) for v in x], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Metric")
        ax2.set_ylabel(r"$N_{\rm pred}$")
        ax.grid(alpha=0.25)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
        
    fig.suptitle(f"{cluster}: parameter sensitivity curves (Inner-loop isolated)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_family_gallery(cluster: str, family_name: str, gallery_rows: List[dict], df_plot: pd.DataFrame, out_png: Path) -> None:
    n = len(gallery_rows)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 9), dpi=160)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
        
    c_ra = resolve_column_name(df_plot, ["ra", "RA"])
    c_dec = resolve_column_name(df_plot, ["dec", "DE"])
    c_pmra = resolve_column_name(df_plot, ["pmra"])
    c_pmdec = resolve_column_name(df_plot, ["pmdec"])
    
    stable_seed = get_stable_seed(cluster)
    bg_idx = downsample_idx(len(df_plot), 40000, seed=stable_seed)
    bg = df_plot.iloc[bg_idx]
    
    for j, row in enumerate(gallery_rows):
        pred_mask = row["pred_mask"]
        true_mask = row["true_mask"]
        ra0 = row["ra0_out"]
        dec0 = row["dec0_out"]
        
        mask_tp = pred_mask & true_mask
        mask_fn = (~pred_mask) & true_mask
        mask_fp = pred_mask & (~true_mask)
        
        ax_sky = axes[0, j]
        if c_ra and c_dec:
            x_a, y_a = compute_tangent_plane(bg[c_ra].to_numpy(), bg[c_dec].to_numpy(), ra0, dec0)
            x_tp, y_tp = compute_tangent_plane(df_plot.loc[mask_tp, c_ra].to_numpy(), df_plot.loc[mask_tp, c_dec].to_numpy(), ra0, dec0)
            x_fn, y_fn = compute_tangent_plane(df_plot.loc[mask_fn, c_ra].to_numpy(), df_plot.loc[mask_fn, c_dec].to_numpy(), ra0, dec0)
            x_fp, y_fp = compute_tangent_plane(df_plot.loc[mask_fp, c_ra].to_numpy(), df_plot.loc[mask_fp, c_dec].to_numpy(), ra0, dec0)
            
            ax_sky.scatter(x_a, y_a, s=5, alpha=0.20, c="lightgray", zorder=1)
            if mask_fn.sum() > 0: 
                ax_sky.scatter(x_fn, y_fn, s=28, alpha=0.80, c="dodgerblue", marker="x", zorder=4)
            if mask_fp.sum() > 0: 
                ax_sky.scatter(x_fp, y_fp, s=45, alpha=0.85, c="crimson", marker="*", zorder=5)
            if mask_tp.sum() > 0: 
                ax_sky.scatter(x_tp, y_tp, s=28, alpha=0.90, c="darkorange", marker="o", zorder=6)
                
            ax_sky.set_xlabel(r"$\Delta \alpha \cos\delta$ (deg)")
            ax_sky.set_ylabel(r"$\Delta \delta$ (deg)")
            ax_sky.invert_xaxis()
            ax_sky.set_title(f"{row['label']}\nF1={row['f1']:.3f}", fontsize=10)
            
        ax_pm = axes[1, j]
        if c_pmra and c_pmdec:
            ax_pm.scatter(bg[c_pmra], bg[c_pmdec], s=5, alpha=0.20, c="lightgray", zorder=1)
            if mask_fn.sum() > 0: 
                ax_pm.scatter(df_plot.loc[mask_fn, c_pmra], df_plot.loc[mask_fn, c_pmdec], s=28, alpha=0.80, c="dodgerblue", marker="x", zorder=4)
            if mask_fp.sum() > 0: 
                ax_pm.scatter(df_plot.loc[mask_fp, c_pmra], df_plot.loc[mask_fp, c_pmdec], s=45, alpha=0.85, c="crimson", marker="*", zorder=5)
            if mask_tp.sum() > 0: 
                ax_pm.scatter(df_plot.loc[mask_tp, c_pmra], df_plot.loc[mask_tp, c_pmdec], s=28, alpha=0.90, c="darkorange", marker="o", zorder=6)
                
            ax_pm.set_xlabel(r"$\mu_{\alpha}^*$ (mas/yr)")
            ax_pm.set_ylabel(r"$\mu_{\delta}$ (mas/yr)")
            
    fig.suptitle(f"{cluster}: {family_name} visual comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def analyze_cluster(cluster: str, tier_name: str, fp: Path, row1: pd.Series, true_ids: np.ndarray, out_root: Path, cache_file: Path) -> Tuple[str, dict, dict]:
    df_raw = load_cone_dataframe(fp)
    
    # [V10] 动态载入测光缓存以解决 CMD 为空的问题
    df_raw = apply_local_photometry_cache(df_raw, cache_file)
    
    out_dir = out_root / tier_name / cluster
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>> Running {cluster} ({tier_name}) ...")
    
    fam_results, res_base, f1_inner_dict = run_inner_loop_sensitivity(cluster, df_raw, row1, true_ids)
    res_off, res_on = run_outer_loop_sensitivity(cluster, df_raw, row1, true_ids)
    
    f1_outer_dict = {"off": res_off["f1"], "on": res_on["f1"]}

    if res_on["f1"] >= res_off["f1"]:
        best_outer = res_on
        outer_label = "center_on"
    else:
        best_outer = res_off
        outer_label = "center_off"

    plot_sensitivity_curves(cluster, fam_results, out_dir / "sensitivity_curves_locked_center.png")
    
    render_standard_diagnostic_panel(
        cluster, 
        best_outer["df_plot"], 
        best_outer["pred_mask"], 
        best_outer["true_mask"],
        best_outer["ra0_out"], 
        best_outer["dec0_out"], 
        out_dir / "baseline_outer_4panel.png",
        f"{cluster} | Outer-Loop Best ({outer_label}) | F1={best_outer['f1']:.3f}, P={best_outer['precision']:.3f}, R={best_outer['recall']:.3f}"
    )
    
    for family in ["support_tau", "beta_shift", "k_set", "anchor_n"]:
        rows = fam_results[family]
        gallery_values = list(GALLERY_LABELS[family].keys())
        selected = [r for r in rows if r["x"] in gallery_values]
        plot_family_gallery(cluster, family, selected, res_base["df_plot"], out_dir / f"{family}_gallery_locked.png")

    return cluster, f1_inner_dict, f1_outer_dict

# =========================================================
# [全局统计与跨样本对比图]
# =========================================================
def generate_global_summary_plots(all_f1_inner: Dict[str, dict], all_f1_outer: Dict[str, dict], out_root: Path):
    print("\n>>> Generating Cross-Cluster Summary Plots for ApJS...")
    
    target_cls = ["UBC1037", "UBC1049", "UBC1265"]
    avail_cls = [c for c in target_cls if c in all_f1_inner]
    
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    markers = ['o', 's', '^']
    
    if avail_cls:
        # Fig 4-18: support_tau
        fig, ax = plt.subplots(figsize=(7, 5))
        tau_vals = SCAN_FAMILIES["support_tau"]
        for i, cl in enumerate(avail_cls):
            y = [all_f1_inner[cl].get(f"support_tau_{v}", 0) for v in tau_vals]
            ax.plot(tau_vals, y, marker=markers[i], markersize=8, linewidth=2.5, color=colors[i], label=cl)
        ax.axvline(x=0.75, color='gray', linestyle='--', linewidth=1.5, zorder=0, label='Baseline (0.75)')
        ax.set_xlabel(r'Support Threshold ($\tau$)')
        ax.set_ylabel(r'$F_1$ Score')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(out_root / 'Fig4-18_support_tau_response.png', dpi=300)
        plt.close()

        # Fig 4-19: beta_shift
        fig, ax = plt.subplots(figsize=(7, 5))
        beta_vals = SCAN_FAMILIES["beta_shift"]
        for i, cl in enumerate(avail_cls):
            y = [all_f1_inner[cl].get(f"beta_shift_{v}", 0) for v in beta_vals]
            ax.plot(beta_vals, y, marker=markers[i], markersize=8, linewidth=2.5, color=colors[i], label=cl)
        ax.axvline(x=0.00, color='gray', linestyle='--', linewidth=1.5, zorder=0, label='Baseline (0.00)')
        ax.set_xticks(beta_vals)
        ax.set_xticklabels([f"{v:+.2f}" if v!=0 else "0.00" for v in beta_vals])
        ax.set_xlabel(r'Global $\beta$ Offset ($\Delta\beta$)')
        ax.set_ylabel(r'$F_1$ Score')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(out_root / 'Fig4-19_beta_shift_response.png', dpi=300)
        plt.close()

        # Fig 4-20: K-set
        fig, ax = plt.subplots(figsize=(8, 5))
        k_vals = SCAN_FAMILIES["k_set"]
        k_x = np.arange(len(k_vals))
        for i, cl in enumerate(avail_cls):
            y = [all_f1_inner[cl].get(f"k_set_{v}", 0) for v in k_vals]
            ax.plot(k_x, y, marker=markers[i], markersize=8, linewidth=2.5, color=colors[i], label=cl)
        ax.axvline(x=2, color='gray', linestyle='--', linewidth=1.5, zorder=0, label='Baseline')
        ax.set_xticks(k_x)
        ax.set_xticklabels([str(v).replace(' ','') for v in k_vals], rotation=15)
        ax.set_xlabel(r'Neighborhood Scale Set ($K$)')
        ax.set_ylabel(r'$F_1$ Score')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(out_root / 'Fig4-20_K_set_response.png', dpi=300)
        plt.close()

    # Fig 4-21: Representative Robustness
    n_targets = len(all_f1_inner)
    if n_targets > 0:
        robust_m, robust_p, robust_b = [], [], []
        for fam, steps in PERTURBATION_STEPS.items():
            minus_key = f"{fam}_{steps['minus']}"
            plus_key = f"{fam}_{steps['plus']}"
            cm = cp = cb = 0
            for cl, f1s in all_f1_inner.items():
                f_base = f1s.get("baseline", 0.0)
                f_m = f1s.get(minus_key, 0.0)
                f_p = f1s.get(plus_key, 0.0)
                
                if abs(f_m - f_base) < 0.05: cm += 1
                if abs(f_p - f_base) < 0.05: cp += 1
                if abs(f_m - f_base) < 0.05 and abs(f_p - f_base) < 0.05: cb += 1
                    
            robust_m.append(cm / n_targets * 100)
            robust_p.append(cp / n_targets * 100)
            robust_b.append(cb / n_targets * 100)

        params = [r'Support $\tau$', r'$\Delta\beta$ shift', r'Scale $K$', r'Anchor $N$']
        x = np.arange(len(params))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 5.5))
        rects1 = ax.bar(x - width, robust_m, width, label='-1 Step Robustness', color='#1f77b4', edgecolor='black')
        rects2 = ax.bar(x, robust_p, width, label='+1 Step Robustness', color='#ff7f0e', edgecolor='black')
        rects3 = ax.bar(x + width, robust_b, width, label='Combined (±1) Robustness', color='#2ca02c', edgecolor='black')

        ax.set_ylabel('Percentage of Targets (%)')
        ax.set_title(r'Representative Perturbation Robustness ($\Delta F_1 < 0.05$)')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.set_ylim(0, 115)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}', 
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points", 
                            ha='center', 
                            va='bottom', 
                            fontsize=9)
                            
        plt.tight_layout()
        plt.savefig(out_root / 'Fig4-21_robustness_stats.png', dpi=300)
        plt.close()

    # Fig 4-22: Center Refinement
    if all_f1_outer:
        targets = list(all_f1_outer.keys())
        f1_off = [all_f1_outer[t]["off"] for t in targets]
        f1_on  = [all_f1_outer[t]["on"] for t in targets]

        x = np.arange(len(targets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, f1_off, width, label='Center Refinement OFF', color='#aaaaaa', edgecolor='black')
        ax.bar(x + width/2, f1_on, width, label='Center Refinement ON', color='#d62728', edgecolor='black')

        ax.set_ylabel(r'$F_1$ Score')
        ax.set_title('Impact of Outer-Loop Center Refinement')
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=30)
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(out_root / 'Fig4-22_center_refinement.png', dpi=300)
        plt.close()
        
    print(f"  -> All 5 summary plots have been successfully generated in: {out_root}")


# =========================================================
# [APJS-level baseline reproduction & population perturbation audit]
# =========================================================
def load_official_tableA(base_dir: Path) -> Optional[pd.DataFrame]:
    fp = base_dir / "ApJS_TableA_Full_Benchmark_CORE_BENCHMARK.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(str).str.strip()
    return df

def collect_population_clusters(cone_map: Dict[str, Path], t1_idx: dict, true_map: dict, official_df: Optional[pd.DataFrame], max_targets: Optional[int] = None) -> List[str]:
    if official_df is not None and "cluster" in official_df.columns:
        clusters = [c for c in official_df["cluster"].astype(str).tolist() if c in cone_map and c in t1_idx and c in true_map]
    else:
        clusters = [c for c in cone_map.keys() if c in t1_idx and c in true_map]
    clusters = sorted(dict.fromkeys(clusters))
    if max_targets is not None and max_targets > 0:
        clusters = clusters[:int(max_targets)]
    return clusters

def run_population_baseline_audit(clusters: List[str], cone_map: Dict[str, Path], t1_idx: dict, true_map: dict,
                                  official_df: Optional[pd.DataFrame], out_root: Path) -> pd.DataFrame:
    rows = []
    for cluster in clusters:
        df_raw = load_cone_dataframe(cone_map[cluster])
        res_prod, _ = run_cluster_once(cluster, df_raw, t1_idx[cluster], true_map[cluster], dict(BASELINE))
        row = {
            "cluster": cluster,
            "rerun_f1": res_prod["f1"],
            "rerun_precision": res_prod["precision"],
            "rerun_recall": res_prod["recall"],
            "rerun_center_mode": res_prod["center_mode"],
            "rerun_center_shift_arcmin": res_prod["center_shift_arcmin"],
            "rerun_objective": res_prod["objective"],
        }
        if official_df is not None:
            sub = official_df.loc[official_df["cluster"] == cluster]
            if len(sub) > 0:
                off = sub.iloc[0]
                row.update({
                    "official_f1": float(off["f1"]),
                    "official_precision": float(off["precision"]),
                    "official_recall": float(off["recall"]),
                    "official_center_mode": str(off["center_mode"]),
                    "official_center_shift_arcmin": float(off["center_shift_arcmin"]),
                    "official_objective": float(off["objective"]),
                })
                row["delta_f1"] = row["rerun_f1"] - row["official_f1"]
                row["delta_precision"] = row["rerun_precision"] - row["official_precision"]
                row["delta_recall"] = row["rerun_recall"] - row["official_recall"]
                row["center_mode_match"] = int(row["rerun_center_mode"] == row["official_center_mode"])
                row["objective_abs_diff"] = abs(row["rerun_objective"] - row["official_objective"])
            else:
                row["delta_f1"] = np.nan
                row["center_mode_match"] = np.nan
                row["objective_abs_diff"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "ApJS_Baseline_Reproduction_Audit.csv", index=False)
    return df

def run_population_perturbation_audit(clusters: List[str], cone_map: Dict[str, Path], t1_idx: dict, true_map: dict,
                                      official_df: Optional[pd.DataFrame], out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    population_rows = []
    center_rows = []
    tier_map = {}
    if official_df is not None:
        for _, r in official_df.iterrows():
            tier_map[str(r["cluster"]).strip()] = {"tier": r.get("tier", np.nan), "tier_label": r.get("tier_label", "")}

    for idx, cluster in enumerate(clusters, 1):
        df_raw = load_cone_dataframe(cone_map[cluster])
        base_inner_cfg = dict(BASELINE); base_inner_cfg["ENABLE_CENTER_REFINEMENT"] = False
        base_inner, _ = run_cluster_once(cluster, df_raw, t1_idx[cluster], true_map[cluster], base_inner_cfg)
        base_outer_off, base_outer_on = run_outer_loop_sensitivity(cluster, df_raw, t1_idx[cluster], true_map[cluster])

        # center refinement audit rows
        cmeta = tier_map.get(cluster, {})
        center_rows.append({
            "cluster": cluster,
            "tier": cmeta.get("tier", np.nan),
            "tier_label": cmeta.get("tier_label", ""),
            "f1_off": base_outer_off["f1"],
            "f1_on": base_outer_on["f1"],
            "delta_f1": base_outer_on["f1"] - base_outer_off["f1"],
            "center_shift_arcmin_on": base_outer_on["center_shift_arcmin"],
            "selected_center_mode_on": base_outer_on["center_mode"],
            "objective_off": base_outer_off["objective"],
            "objective_on": base_outer_on["objective"],
        })

        for family, info in POPULATION_PERTURBATIONS.items():
            for step_name in ["minus", "plus"]:
                cfg = make_cfg_with_override(family, info[step_name])
                if info["loop"] == "inner":
                    cfg["ENABLE_CENTER_REFINEMENT"] = False
                    res, _ = run_cluster_once(cluster, df_raw, t1_idx[cluster], true_map[cluster], cfg)
                    base_f1 = base_inner["f1"]
                    base_n = base_inner["n_pred"]
                    base_obj = base_inner["objective"]
                else:
                    cfg["ENABLE_CENTER_REFINEMENT"] = True
                    res, _ = run_cluster_once(cluster, df_raw, t1_idx[cluster], true_map[cluster], cfg)
                    base_f1 = base_outer_on["f1"]
                    base_n = base_outer_on["n_pred"]
                    base_obj = base_outer_on["objective"]

                population_rows.append({
                    "cluster": cluster,
                    "tier": cmeta.get("tier", np.nan),
                    "tier_label": cmeta.get("tier_label", ""),
                    "family": family,
                    "loop": info["loop"],
                    "step": step_name,
                    "value": str(info[step_name]),
                    "baseline_f1": base_f1,
                    "perturbed_f1": res["f1"],
                    "delta_f1": res["f1"] - base_f1,
                    "abs_delta_f1": abs(res["f1"] - base_f1),
                    "baseline_n_pred": base_n,
                    "perturbed_n_pred": res["n_pred"],
                    "delta_n_pred": res["n_pred"] - base_n,
                    "baseline_objective": base_obj,
                    "perturbed_objective": res["objective"],
                    "center_mode": res["center_mode"],
                    "center_shift_arcmin": res["center_shift_arcmin"],
                })

        if idx % 25 == 0 or idx == len(clusters):
            print(f"  -> [POPULATION AUDIT] Completed {idx}/{len(clusters)} clusters.")

    df_pop = pd.DataFrame(population_rows)
    df_ctr = pd.DataFrame(center_rows)
    df_pop.to_csv(out_root / "ApJS_Population_Perturbation_Audit.csv", index=False)
    df_ctr.to_csv(out_root / "ApJS_Population_Center_Refinement_Audit.csv", index=False)
    return df_pop, df_ctr

def summarize_population_robustness(df_pop: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rows = []
    thresholds = [0.01, 0.03, 0.05]
    for family, g in df_pop.groupby("family"):
        row = {
            "family": family,
            "n_runs": len(g),
            "median_abs_delta_f1": float(g["abs_delta_f1"].median()),
            "p90_abs_delta_f1": float(g["abs_delta_f1"].quantile(0.90)),
            "mean_abs_delta_f1": float(g["abs_delta_f1"].mean()),
        }
        for thr in thresholds:
            row[f"frac_le_{thr:.2f}"] = float((g["abs_delta_f1"] <= thr).mean())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("median_abs_delta_f1")
    df.to_csv(out_root / "ApJS_Population_Robustness_Summary.csv", index=False)
    return df

def plot_baseline_reproduction(df_base: pd.DataFrame, out_root: Path) -> None:
    if "official_f1" not in df_base.columns:
        return
    sub = df_base.dropna(subset=["official_f1", "rerun_f1"]).copy()
    if len(sub) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220)
    ax = axes[0]
    ax.scatter(sub["official_f1"], sub["rerun_f1"], s=22, alpha=0.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlabel("Official benchmark $F_1$")
    ax.set_ylabel("Reproduced baseline $F_1$")
    ax.set_title("Baseline Reproduction Audit")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1]
    ax.hist(sub["delta_f1"], bins=30, alpha=0.85)
    ax.axvline(0.0, linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlabel(r"$\Delta F_1$ (rerun - official)")
    ax.set_ylabel("Cluster count")
    ax.set_title("Reproduction Residuals")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_root / "Fig4-23_baseline_reproduction_audit.png", bbox_inches="tight")
    plt.close(fig)

def plot_population_delta_box(df_pop: pd.DataFrame, out_root: Path) -> None:
    fams = list(dict.fromkeys(df_pop["family"].tolist()))
    data = [df_pop.loc[df_pop["family"] == fam, "delta_f1"].astype(float).to_numpy() for fam in fams]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=220)
    ax.boxplot(data, labels=[FAMILY_DISPLAY.get(f, f) for f in fams], showfliers=False)
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.5)
    ax.set_ylabel(r"$\Delta F_1$")
    ax.set_title("Population-level Perturbation Response")
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_root / "Fig4-24_population_deltaF1_boxplot.png", bbox_inches="tight")
    plt.close(fig)

def plot_population_ecdf(df_pop: pd.DataFrame, out_root: Path) -> None:
    principal = ["support_tau", "beta_shift", "k_set", "anchor_n", "ruwe_max"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    for fam in principal:
        sub = np.sort(df_pop.loc[df_pop["family"] == fam, "abs_delta_f1"].astype(float).to_numpy())
        if len(sub) == 0:
            continue
        y = np.arange(1, len(sub) + 1) / len(sub)
        ax.plot(sub, y, linewidth=2.2, label=FAMILY_DISPLAY.get(fam, fam))
    ax.axvline(0.05, linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlabel(r"$|\Delta F_1|$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("ECDF of Population-level Sensitivity")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_root / "Fig4-25_population_absdeltaF1_ecdf.png", bbox_inches="tight")
    plt.close(fig)

def plot_tier_heatmap(df_pop: pd.DataFrame, out_root: Path) -> None:
    if "tier_label" not in df_pop.columns:
        return
    tiers = [t for t in df_pop["tier_label"].dropna().astype(str).unique().tolist() if t != ""]
    fams = list(dict.fromkeys(df_pop["family"].tolist()))
    if len(tiers) == 0 or len(fams) == 0:
        return
    M = np.full((len(tiers), len(fams)), np.nan, dtype=float)
    for i, tier in enumerate(tiers):
        for j, fam in enumerate(fams):
            sub = df_pop.loc[(df_pop["tier_label"] == tier) & (df_pop["family"] == fam), "abs_delta_f1"]
            if len(sub) > 0:
                M[i, j] = float(sub.median())
    fig, ax = plt.subplots(figsize=(14, 5), dpi=220)
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels([FAMILY_DISPLAY.get(f, f) for f in fams], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(tiers)))
    ax.set_yticklabels(tiers)
    ax.set_title(r"Tier-stratified Median $|\Delta F_1|$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Median $|\Delta F_1|$")
    fig.tight_layout()
    fig.savefig(out_root / "Fig4-26_tier_stratified_heatmap.png", bbox_inches="tight")
    plt.close(fig)

def plot_center_refinement_scatter(df_ctr: pd.DataFrame, out_root: Path) -> None:
    if len(df_ctr) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    ax.scatter(df_ctr["center_shift_arcmin_on"], df_ctr["delta_f1"], s=26, alpha=0.85)
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.5)
    ax.axvline(20.0, linestyle="--", color="gray", linewidth=1.2)
    ax.set_xlabel("Center shift (arcmin, refinement ON)")
    ax.set_ylabel(r"$\Delta F_1$ (ON - OFF)")
    ax.set_title("Center Refinement Risk Map")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_root / "Fig4-27_center_shift_vs_deltaF1.png", bbox_inches="tight")
    plt.close(fig)

def plot_robustness_ladder(df_summary: pd.DataFrame, out_root: Path) -> None:
    fams = df_summary["family"].tolist()
    x = np.arange(len(fams))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6), dpi=220)
    for k, thr in enumerate([0.01, 0.03, 0.05]):
        ax.bar(x + (k - 1) * width,
               df_summary[f"frac_le_{thr:.2f}"].to_numpy() * 100.0,
               width=width, label=fr"$|\Delta F_1| \leq {thr:.2f}$")
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_DISPLAY.get(f, f) for f in fams], rotation=25, ha="right")
    ax.set_ylabel("Percentage of perturbation runs (%)")
    ax.set_title("Robustness Ladder Across Perturbation Families")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_root / "Fig4-28_robustness_ladder.png", bbox_inches="tight")
    plt.close(fig)

def plot_local_response_surfaces(cone_map: Dict[str, Path], t1_idx: dict, true_map: dict, out_root: Path) -> None:
    showcase = ["UBC1037", "UBC1049", "UBC1265"]
    showcase = [c for c in showcase if c in cone_map and c in t1_idx and c in true_map]
    if len(showcase) == 0:
        return
    tau_vals = [0.65, 0.70, 0.75, 0.80, 0.85]
    k_vals = SCAN_FAMILIES["k_set"]
    fig, axes = plt.subplots(1, len(showcase), figsize=(5 * len(showcase), 4.5), dpi=220, squeeze=False)
    axes = axes.ravel()
    for ax, cluster in zip(axes, showcase):
        df_raw = load_cone_dataframe(cone_map[cluster])
        M = np.zeros((len(tau_vals), len(k_vals)), dtype=float)
        for i, tau in enumerate(tau_vals):
            for j, kval in enumerate(k_vals):
                cfg = dict(BASELINE)
                cfg["ENABLE_CENTER_REFINEMENT"] = False
                cfg["SUPPORT_TAU"] = float(tau)
                cfg["K_LIST"] = tuple(int(x) for x in kval)
                res, _ = run_cluster_once(cluster, df_raw, t1_idx[cluster], true_map[cluster], cfg)
                M[i, j] = res["f1"]
        im = ax.imshow(M, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(k_vals)))
        ax.set_xticklabels([str(v) for v in k_vals], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(tau_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in tau_vals])
        # baseline cell
        i0 = tau_vals.index(0.75); j0 = k_vals.index((16, 24, 32))
        ax.scatter([j0], [i0], marker="s", s=130, facecolors="none", edgecolors="white", linewidths=1.8)
        ax.set_title(cluster)
        ax.set_xlabel(r"Neighborhood scale set $K$")
        ax.set_ylabel(r"Support threshold $\tau$")
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.9)
    cbar.set_label(r"$F_1$")
    fig.suptitle("Local Response Surfaces Around the Adopted Working Point", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_root / "Fig4-29_local_response_surfaces.png", bbox_inches="tight")
    plt.close(fig)

def print_population_summary(df_base: pd.DataFrame, df_summary: pd.DataFrame, df_ctr: pd.DataFrame) -> None:
    print("\n" + "=" * 105)
    print("APJS-LEVEL POPULATION SENSITIVITY SUMMARY")
    print("=" * 105)
    if "delta_f1" in df_base.columns:
        sub = df_base.dropna(subset=["delta_f1"])
        if len(sub) > 0:
            exact = int((sub["delta_f1"].abs() < 1e-10).sum())
            print(f"Baseline reproduction exact matches: {exact}/{len(sub)} ({100.0 * exact / len(sub):.1f}%)")
            print(f"Median |ΔF1| vs official benchmark: {sub['delta_f1'].abs().median():.4e}")
            print(f"Center-mode match fraction: {100.0 * sub['center_mode_match'].fillna(0).mean():.1f}%")
    if len(df_summary) > 0:
        print("\nPopulation robustness ranking by median |ΔF1|:")
        for _, r in df_summary.iterrows():
            print(f"  - {r['family']:<18} median={r['median_abs_delta_f1']:.4f} | p90={r['p90_abs_delta_f1']:.4f} | <=0.05={100.0*r['frac_le_0.05']:.1f}%")
    if len(df_ctr) > 0:
        improved = int((df_ctr["delta_f1"] > 0.01).sum())
        degraded = int((df_ctr["delta_f1"] < -0.01).sum())
        stable = int(len(df_ctr) - improved - degraded)
        print("\nCenter refinement population audit:")
        print(f"  Stable  : {stable}/{len(df_ctr)} ({100.0*stable/len(df_ctr):.1f}%)")
        print(f"  Improved: {improved}/{len(df_ctr)} ({100.0*improved/len(df_ctr):.1f}%)")
        print(f"  Degraded: {degraded}/{len(df_ctr)} ({100.0*degraded/len(df_ctr):.1f}%)")

def print_representative_summary(all_f1_inner: Dict[str, dict], all_f1_outer: Dict[str, dict]):
    n_targets = len(all_f1_inner)
    if n_targets == 0: return

    print("\n" + "=" * 105)
    print(f"ANALYSIS BRANCH: {ANALYSIS_BRANCH}")
    print("REPRESENTATIVE PERTURBATION ROBUSTNESS SUMMARY (ΔF1 < 0.05 under ±1 step perturbation)")
    print("=" * 105)
    
    stats = {}
    for fam, steps in PERTURBATION_STEPS.items():
        minus_key = f"{fam}_{steps['minus']}"
        plus_key = f"{fam}_{steps['plus']}"
        count_m = count_p = count_both = 0
        
        for cluster, f1s in all_f1_inner.items():
            f_base = f1s.get("baseline", 0.0)
            f_m = f1s.get(minus_key, 0.0)
            f_p = f1s.get(plus_key, 0.0)
            
            robust_m = abs(f_m - f_base) < 0.05
            robust_p = abs(f_p - f_base) < 0.05
            
            if robust_m: count_m += 1
            if robust_p: count_p += 1
            if robust_m and robust_p: count_both += 1
                
        stats[fam] = (count_m, count_p, count_both)

    print(f"{'Parameter':<15} {'-1 Step Robustness':<25} {'+1 Step Robustness':<25} {'Combined Robustness (±1)':<25}")
    print("-" * 105)
    for fam, (cm, cp, cb) in stats.items():
        print(f"{fam:<15} {cm}/{n_targets} ({cm/n_targets*100:.1f}%)"
              f"          {cp}/{n_targets} ({cp/n_targets*100:.1f}%)"
              f"          {cb}/{n_targets} ({cb/n_targets*100:.1f}%)")

    print("\n" + "=" * 105)
    print("OUTER-LOOP CENTER REFINEMENT SUMMARY (center_on vs center_off)")
    print("=" * 105)
    
    count_stable = count_improved = count_degraded = 0
    for cluster, f1s in all_f1_outer.items():
        df1 = f1s["on"] - f1s["off"]
        if df1 > 0.01: count_improved += 1
        elif df1 < -0.01: count_degraded += 1
        else: count_stable += 1
        
    print(f"Total Targets Evaluated: {n_targets}")
    print(f"  Stable (|\u0394F1| <= 0.01) : {count_stable}/{n_targets} ({count_stable/n_targets*100:.1f}%)")
    print(f"  Improved (\u0394F1 > 0.01)    : {count_improved}/{n_targets} ({count_improved/n_targets*100:.1f}%)")
    print(f"  Degraded (\u0394F1 < -0.01)   : {count_degraded}/{n_targets} ({count_degraded/n_targets*100:.1f}%)")
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--targets", type=str, nargs="*", default=None,
                        help="Optional representative targets to override the built-in showcase list.")
    parser.add_argument("--max_population_targets", type=int, default=None,
                        help="Optional cap on the number of clusters used in the population-level audit.")
    parser.add_argument("--skip_population_audit", action="store_true",
                        help="Skip the population-level APJS audit and only run the representative study.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (base_dir / "data")
    out_root = base_dir / "sensitivity_core_v12"
    out_root.mkdir(parents=True, exist_ok=True)

    cache_file = data_dir / "mctnc_photometry_cache.csv"

    t1_idx, true_map = load_reference_tables(data_dir)
    cone_files = discover_cone_files(base_dir, data_dir)
    cone_map = {infer_cluster_name(fp): fp for fp in cone_files}
    selected_map = {"Custom": [x.strip() for x in args.targets]} if args.targets else SENSITIVITY_TARGETS

    print("\n" + "=" * 105)
    print("M-CTNC CORE_BENCHMARK Sensitivity Analysis (V12: APJS Final Audit)")
    print("=" * 105)

    # -----------------------------------------------------
    # Representative audit
    # -----------------------------------------------------
    all_f1_inner, all_f1_outer = {}, {}
    for tier_name, clusters in selected_map.items():
        for cluster in clusters:
            if cluster in cone_map and cluster in t1_idx and cluster in true_map:
                cl_name, f1_in, f1_out = analyze_cluster(
                    cluster, tier_name, cone_map[cluster], t1_idx[cluster], true_map[cluster], out_root, cache_file
                )
                all_f1_inner[cl_name] = f1_in
                all_f1_outer[cl_name] = f1_out
            else:
                print(f"  -> [SKIP] {cluster}: missing cone/reference data.")

    print_representative_summary(all_f1_inner, all_f1_outer)
    generate_global_summary_plots(all_f1_inner, all_f1_outer, out_root)

    # -----------------------------------------------------
    # Population-level APJS audit
    # -----------------------------------------------------
    if not args.skip_population_audit:
        official_df = load_official_tableA(base_dir)
        if official_df is None:
            print("  -> [WARNING] Official Table A not found in base_dir. Population baseline reproduction will run without direct official comparison.")
        pop_clusters = collect_population_clusters(
            cone_map, t1_idx, true_map, official_df, max_targets=args.max_population_targets
        )
        print(f"\n>>> Population-level APJS audit will evaluate {len(pop_clusters)} clusters.")

        if len(pop_clusters) > 0:
            df_base = run_population_baseline_audit(pop_clusters, cone_map, t1_idx, true_map, official_df, out_root)
            df_pop, df_ctr = run_population_perturbation_audit(pop_clusters, cone_map, t1_idx, true_map, official_df, out_root)
            df_summary = summarize_population_robustness(df_pop, out_root)

            plot_baseline_reproduction(df_base, out_root)
            plot_population_delta_box(df_pop, out_root)
            plot_population_ecdf(df_pop, out_root)
            plot_tier_heatmap(df_pop, out_root)
            plot_center_refinement_scatter(df_ctr, out_root)
            plot_robustness_ladder(df_summary, out_root)
            plot_local_response_surfaces(cone_map, t1_idx, true_map, out_root)
            print_population_summary(df_base, df_summary, df_ctr)
        else:
            print("  -> [WARNING] No valid clusters available for population-level APJS audit.")

if __name__ == "__main__":
    main()