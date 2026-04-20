# -*- coding: utf-8 -*-
"""
M-CTNC: Multiscale Common Tightest Neighbors Consensus
=============================================================================
A robust, purity-oriented clustering pipeline for identifying open cluster 
members in dense astrometric fields (e.g., Gaia DR3).

This pipeline introduces a topologic consensus mechanism across multiple 
neighborhood scales (K) and structural strictness levels (Beta) to extract 
highly reliable, conservative cluster cores while systematically suppressing 
field contamination.

Core Features:
- Multiscale Topological Consensus filtering.
- Self-tuning K-Nearest Neighbors for adaptive density evaluation.
- Fully automated astrophysical audit (Recall Attribution & Truth Decontamination).

Author: Hao Wang et al.
Release Version: 1.0.0 (Open-Source Scientific Release - Anti-Clipboard-Corruption)
=============================================================================
"""
import sys
import time
import math
import argparse
import traceback
import re
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import csgraph, triu as sp_triu
import matplotlib.pyplot as plt

# Switch matplotlib backend for batch processing on headless servers
plt.switch_backend('Agg')
import warnings
warnings.filterwarnings("ignore")

# Global Fallback Switch for ESA Query Limits
DISABLE_ONLINE_FETCH = False

# =============================================================================
# [Pipeline Configuration]
# =============================================================================
PIPELINE_NAME = "M-CTNC_Core_Extractor_v1.0"

# Operating Regime:
# "CORE_BENCHMARK"   -> Optimizes for maximal purity and robust core extraction.
# "HALO_EXPLORATION" -> Relaxes consensus bounds to search for tidal structures.
EXTRACTION_MODE = "CORE_BENCHMARK"  

if EXTRACTION_MODE == "CORE_BENCHMARK":
    K_LIST = (16, 24, 32)
    BETA_LIST = (0.45, 0.50, 0.55)
    SUPPORT_TAU = 0.75
    ANCHOR_SUPPORT_TAU = 0.45
else:
    K_LIST = (16, 32, 48)
    BETA_LIST = (0.40, 0.45, 0.50)
    SUPPORT_TAU = 0.67
    ANCHOR_SUPPORT_TAU = 0.34

# Candidate & Anchor Sampling Hyperparameters
MIN_PTS = 5
CAP_MIN, CAP_MAX = 300, 1200
CAP_A, CAP_B = 12, 250
CAP_BACKOFF_RATIO = 0.60
CAND_SCORE_RATIO, CAND_POS_RATIO, CAND_KIN_RATIO = 0.60, 0.25, 0.15

# ANTI-CORRUPTION: Separate variables instead of tuple to avoid index parsing errors
ANCHOR_N_TARGET = 36
ANCHOR_SPLIT_W1 = 0.40
ANCHOR_SPLIT_W2 = 0.30
ANCHOR_SPREAD_ENABLE = True
ANCHOR_SPREAD_K = 28

# Astrometric Weighting & Cuts
W_POS, W_PLX, W_PM = 1.00, 1.00, 1.00
ENABLE_SELF_TUNING = True
K_QUERY_FACTOR, K_QUERY_MIN = 3, 96
FLOOR_PLX, FLOOR_PM = 0.10, 0.20

ENABLE_CENTER_REFINEMENT = True
CENTER_REFINE_TOP_R = 600
CENTER_REFINE_K = 24
CENTER_SHIFT_BLEND = 0.70
CENTER_SHIFT_LIMIT_ARCMIN = 20

ENABLE_QUALITY_CUTS = True
RUWE_MAX = 1.6

# =============================================================================
# [Astrometric Transformations & Data Utilities]
# =============================================================================
def _now() -> float: 
    """Returns high-resolution performance counter."""
    return time.perf_counter()

def _safe_float(x: Any) -> float:
    """Safely casts arbitrary variables to float."""
    try: return float(x)
    except Exception: return float("nan")

def wrap_ra_coordinate(ra_deg: np.ndarray, ra0_deg: float) -> np.ndarray:
    """Safely wraps Right Ascension differences around the 180-degree boundary."""
    dra = ra_deg - ra0_deg
    return (dra + 180.0) % 360.0 - 180.0

def compute_tangent_plane(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Projects celestial coordinates (RA, Dec) onto a standard tangent plane."""
    dra = wrap_ra_coordinate(ra_deg, ra0_deg)
    x = dra * math.cos(math.radians(dec0_deg))
    y = dec_deg - dec0_deg
    return x.astype(np.float32), y.astype(np.float32)

def compute_robust_mad_scale(x: np.ndarray, floor: float) -> float:
    """Calculates Median Absolute Deviation to robustly estimate standard deviation."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 10: return float(floor)
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    return float(s) if (np.isfinite(s) and s >= floor) else float(floor)

def resolve_column_name(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Matches dataframe columns safely against a list of common aliases."""
    cols_lower = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols_lower:
            idx_val = cols_lower.index(cand.lower())
            col_iter = iter(df.columns)
            # Advance iterator to safely avoid bracket indexing
            for _ in range(idx_val): next(col_iter)
            return next(col_iter)
    return None

def find_first_col_strict(df: pd.DataFrame, candidates: List[str]) -> str:
    """Strictly resolves column aliases, raising KeyError if missing."""
    res = resolve_column_name(df, candidates)
    if res is None: raise KeyError(f"Missing required column among: {candidates}")
    return res

def robust_tree_query(tree: cKDTree, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-compatible cKDTree query handling both multi-threading and legacy APIs."""
    try: return tree.query(X, k=k, workers=-1)
    except TypeError: return tree.query(X, k=k)

def infer_cluster_name(path: Path) -> str:
    """Extracts standardized cluster names from target file paths."""
    s = path.stem
    if s.lower().startswith("gaia_cone_"): s = s[len("gaia_cone_"):]
    m = re.match(r"([A-Za-z]+\d+)", s.strip())
    if m:
        return m.group(1)
    # Safe negative indexing alternative
    parts = s.split("_")
    part_iter = iter(parts)
    last_part = next(part_iter)
    for p in part_iter: last_part = p
    dash_parts = last_part.split("-")
    dash_iter = iter(dash_parts)
    last_dash = next(dash_iter)
    for dp in dash_iter: last_dash = dp
    return last_dash.strip()

def discover_cone_files(base_dir: Path, data_dir: Path) -> List[Path]:
    """Dynamically locates Gaia cone search CSV files in standard directories."""
    candidates = [data_dir, base_dir, base_dir / "cones", base_dir / "data_cones", base_dir / "data-gaia"]
    seen, dirs = set(), []
    for d in candidates:
        try: d = d.resolve()
        except Exception: pass
        if d.exists() and d.is_dir() and str(d) not in seen:
            seen.add(str(d))
            dirs.append(d)
    files = []
    for d in dirs:
        files.extend(list(d.glob("gaia_cone_*.csv")))
        files.extend(list(d.glob("gaia_cone_*.CSV")))
    if not files:
        for d in dirs:
            for p in d.rglob("*"):
                if p.is_file() and p.name.lower().startswith("gaia_cone_") and p.name.lower().endswith(".csv"):
                    files.append(p)
    return sorted({p.resolve() for p in files})

# =============================================================================
# [M-CTNC Algorithm Engine]
# =============================================================================
def build_adaptive_knn_graph(X: np.ndarray, k_max: int, k_query: int, self_tuning: bool) -> np.ndarray:
    """Builds a K-Nearest Neighbor graph, optionally applying self-tuning density correction."""
    X = np.atleast_2d(X)
    n_samples = len(X)
    if n_samples < 2: return np.empty((n_samples, 0), dtype=np.int32) 
    
    k_max_safe = int(min(max(1, int(k_max)), n_samples - 1))
    k_query_safe = int(min(max(int(k_query), k_max_safe + 8), n_samples - 1))
    
    tree = cKDTree(X)
    try:
        dist, idx = tree.query(X, k=k_query_safe + 1, workers=-1)
    except TypeError:
        dist, idx = tree.query(X, k=k_query_safe + 1)
        
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
    """Extracts connected components strictly adhering to the Common Tightest Neighbor rule."""
    idx_knn = np.atleast_2d(idx_knn)
    n_samples = len(idx_knn)
    
    # Safe negative indexing alternative to avoid clipboard drop
    shape_iter = iter(idx_knn.shape)
    next(shape_iter) # skip dim 0
    try:
        k_neighbors = int(next(shape_iter))
    except StopIteration:
        k_neighbors = 0
        
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
    """Identifies the core cluster substructure among multiple topologies via anchor voting."""
    valid = labels >= 0
    if not np.any(valid): return np.zeros_like(labels, dtype=bool), "EMPTY_CC"
    
    uniq = np.unique(labels[valid])
    if len(uniq) == 1: 
        c_id = int(next(iter(uniq)))
        return labels == c_id, "ONE_CC"
        
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
        s = float(np.mean(score[labels == cid])) if np.any(labels == cid) else float("inf")
        if s < best_s: best_s, best_c = s, cid
        
    best_choice = int(best_c if best_c is not None else next(iter(uniq)))
    return labels == best_choice, f"CC_ANCHOR={votes.get(best_choice, 0)}"

def generate_multiscale_consensus(Xc: np.ndarray, sc: np.ndarray, anchor_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Aggregates topological subgraphs across multiple scales to form a stable consensus."""
    Xc = np.atleast_2d(Xc)
    n_xc = len(Xc)
    
    k_max = int(max(K_LIST))
    k_query = max(int(k_max * K_QUERY_FACTOR), int(K_QUERY_MIN))
    
    idx_knn_max = build_adaptive_knn_graph(Xc, k_max=k_max, k_query=k_query, self_tuning=bool(ENABLE_SELF_TUNING))
    support = np.zeros(n_xc, dtype=np.float32)
    cfgs = 0
    
    for beta in BETA_LIST:
        for k in K_LIST:
            k_int = int(k)
            if k_int >= n_xc: continue
            labels = extract_tight_components(idx_knn_max[:, :k_int], float(beta), int(MIN_PTS))
            mask, _ = isolate_target_substructure(labels, anchor_local, sc)
            support += mask.astype(np.float32)
            cfgs += 1
            
    if cfgs == 0:
        pred = np.zeros(n_xc, dtype=bool)
        pred[np.argsort(sc)[:max(MIN_PTS, 30)]] = True
        return pred, pred.astype(np.float32), "DEGENERATE"
        
    sup_ratio = support / float(cfgs)
    pred = sup_ratio >= float(SUPPORT_TAU)
    
    if pred.sum() < MIN_PTS and len(anchor_local) > 0: 
        pred = sup_ratio >= float(ANCHOR_SUPPORT_TAU)
        
    return pred, sup_ratio, f"MS(cfg={cfgs})|tau={SUPPORT_TAU:.2f}"

def sample_candidates_and_anchors(score: np.ndarray, pos2: np.ndarray, kin2: np.ndarray, X: np.ndarray, cap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Selects high-confidence anchors using a stratified strategy across astrometric dimensions."""
    n_total = len(score)
    cap = int(min(max(cap, 80), n_total))
    n_score = max(20, min(int(round(cap * float(CAND_SCORE_RATIO))), cap))
    n_pos   = max(15, min(int(round(cap * float(CAND_POS_RATIO))), cap))
    n_kin   = max(10, min(cap - n_score - n_pos, cap))
    
    cand = np.unique(np.concatenate([np.argsort(score)[:n_score], np.argsort(pos2)[:n_pos], np.argsort(kin2)[:n_kin]]).astype(np.int32))
    if len(cand) < cap: 
        cand = np.unique(np.concatenate([cand, np.argsort(score)[n_score:n_score+(cap-len(cand))].astype(np.int32)]))
        
    aN = int(min(max(ANCHOR_N_TARGET, 12), min(90, len(cand))))
    
    a_score = max(10, int(round(aN * float(ANCHOR_SPLIT_W1))))
    a_pos = max(8, int(round(aN * float(ANCHOR_SPLIT_W2))))
    a_kin = max(8, aN - a_score - a_pos)
    
    cs, cp, ck = score[cand], pos2[cand], kin2[cand]
    anc = np.unique(np.concatenate([cand[np.argsort(cs)[:a_score]], cand[np.argsort(cp)[:a_pos]], cand[np.argsort(ck)[:a_kin]]]).astype(np.int32))
    
    if ANCHOR_SPREAD_ENABLE and len(anc) > ANCHOR_SPREAD_K:
        P = X[anc, :2].astype(np.float32, copy=False)
        c = np.median(P, axis=0)
        dist_to_center = np.sum((P - c)**2, axis=1)
        first_sel = int(np.argmin(dist_to_center))
        
        sel = [first_sel]
        dist_min = np.sum((P - P[first_sel])**2, axis=1)
        for _ in range(1, ANCHOR_SPREAD_K):
            j = int(np.argmax(dist_min))
            sel.append(j)
            dist_min = np.minimum(dist_min, np.sum((P - P[j])**2, axis=1))
        anc = anc[sel]
        
    return cand.astype(np.int32, copy=False), anc.astype(np.int32, copy=False)

def preprocess_astrometry(df: pd.DataFrame, ra0: float, dec0: float, rdeg: float, plx0: Optional[float], pmra0: Optional[float], pmdec0: Optional[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Standardizes coordinates and kinematics against background noise distributions."""
    ra, dec = df["ra"].to_numpy(np.float64), df["dec"].to_numpy(np.float64)
    x_deg, y_deg = compute_tangent_plane(ra, dec, ra0, dec0)
    sig_pos = max(0.10, float(rdeg) if np.isfinite(rdeg) and rdeg > 0 else 1.0)
    
    plx = df["parallax"].to_numpy(np.float32)
    pmra, pmdec = df["pmra"].to_numpy(np.float32), df["pmdec"].to_numpy(np.float32)
    
    plx0 = float(np.nanmedian(plx)) if plx0 is None or not np.isfinite(plx0) else float(plx0)
    pmra0 = float(np.nanmedian(pmra)) if pmra0 is None or not np.isfinite(pmra0) else float(pmra0)
    pmdec0 = float(np.nanmedian(pmdec)) if pmdec0 is None or not np.isfinite(pmdec0) else float(pmdec0)
    
    dplx, dpmra, dpmdec = (plx - plx0).astype(np.float32, copy=False), (pmra - pmra0).astype(np.float32, copy=False), (pmdec - pmdec0).astype(np.float32, copy=False)
    
    core_idx = np.argsort(np.hypot(x_deg, y_deg))[:min(600, max(80, int(0.10 * len(df))))]
    sig_plx_intr = compute_robust_mad_scale(dplx[core_idx], floor=FLOOR_PLX)
    sig_pm_intr = compute_robust_mad_scale(np.hypot(dpmra[core_idx], dpmdec[core_idx]), floor=FLOOR_PM)
    
    plx_err = df["parallax_error"].to_numpy(np.float32) if "parallax_error" in df.columns else None
    pmra_err = df["pmra_error"].to_numpy(np.float32) if "pmra_error" in df.columns else None
    pmdec_err = df["pmdec_error"].to_numpy(np.float32) if "pmdec_error" in df.columns else None
    
    if plx_err is not None: sig_plx = np.sqrt(sig_plx_intr**2 + np.maximum(plx_err, 0.0)**2).astype(np.float32)
    else: sig_plx = np.full(len(df), sig_plx_intr, dtype=np.float32)
    
    if pmra_err is not None and pmdec_err is not None: sig_pm = np.sqrt(sig_pm_intr**2 + 0.5*(np.maximum(pmra_err,0.0)**2 + np.maximum(pmdec_err,0.0)**2)).astype(np.float32)
    else: sig_pm = np.full(len(df), sig_pm_intr, dtype=np.float32)
    
    zx, zy = (x_deg / sig_pos).astype(np.float32, copy=False), (y_deg / sig_pos).astype(np.float32, copy=False)
    zplx, zpmra, zpmdec = (dplx / sig_plx).astype(np.float32, copy=False), (dpmra / sig_pm).astype(np.float32, copy=False), (dpmdec / sig_pm).astype(np.float32, copy=False)
    
    pos2 = (zx*zx + zy*zy).astype(np.float32, copy=False)
    kin2 = (zplx*zplx + zpmra*zpmra + zpmdec*zpmdec).astype(np.float32, copy=False)
    score = (W_POS*pos2 + W_PLX*(zplx*zplx) + W_PM*(zpmra*zpmra + zpmdec*zpmdec)).astype(np.float32)
    
    X = np.column_stack([zx, zy, zplx, zpmra, zpmdec]).astype(np.float32, copy=False)
    aux = {"sig_pos_deg": float(sig_pos), "sig_plx_intr": float(sig_plx_intr), "sig_pm_intr": float(sig_pm_intr)}
    return score, X, pos2, kin2, aux

def evaluate_extraction_objective(sc: np.ndarray, sup: np.ndarray, pred: np.ndarray, cap: int) -> float:
    """Evaluates the stability and compactness of the extracted substructure."""
    if pred.sum() < MIN_PTS: return 1e9
    return float(np.mean(sc[pred]) - 0.35 * np.mean(sup[pred]) + 0.40 * (math.log(1.0 + int(pred.sum())) / math.log(1.0 + cap)))

def orchestrate_single_center(df_cone: pd.DataFrame, ra0: float, dec0: float, rdeg: float, plx0: Optional[float], pmra0: Optional[float], pmdec0: Optional[float]) -> Tuple:
    score, X, pos2, kin2, aux = preprocess_astrometry(df_cone, ra0, dec0, rdeg, plx0, pmra0, pmdec0)
    cap0 = max(CAP_MIN, min(CAP_MAX, int(CAP_A * ANCHOR_N_TARGET + CAP_B)))
    caps_to_try = [cap0, max(CAP_MIN, int(cap0 * CAP_BACKOFF_RATIO))]
    
    primary_cap = next(iter(caps_to_try))
    
    best_J, best_pack = None, None
    for cap in caps_to_try:
        cap_int = int(cap)
        cand_idx, anchor_idx = sample_candidates_and_anchors(score, pos2, kin2, X, cap_int)
        Xc, sc = X[cand_idx], score[cand_idx]
        inv_map = {int(g): i for i, g in enumerate(cand_idx)}
        anchor_local = np.array([inv_map.get(int(a), -1) for a in anchor_idx], dtype=np.int32)
        anchor_local = anchor_local[anchor_local >= 0]
        
        pred, sup, tag = generate_multiscale_consensus(Xc, sc, anchor_local)
        J = evaluate_extraction_objective(sc, sup, pred, cap_int)
        
        pack = {"cap": cap_int, "n_candidates": len(cand_idx), "n_pred_local": int(pred.sum()), "tag": tag, "objective": float(J)}
        if best_J is None or J < best_J: 
            best_J, best_pack = J, (cand_idx, pred, sup, pack, score, X, pos2, kin2, aux)
            
        if cap_int == primary_cap and MIN_PTS <= pred.sum() < int(0.55 * cap_int): 
            break
            
    return best_pack

def process_single_cluster(cluster: str, df_cone_raw: pd.DataFrame, row1: pd.Series, true_ids: np.ndarray) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Main pipeline execution for a single cluster."""
    t0 = _now()
    df_cone = df_cone_raw
    if ENABLE_QUALITY_CUTS and "ruwe" in df_cone.columns: df_cone = df_cone[df_cone["ruwe"].astype(np.float64) <= float(RUWE_MAX)]
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

    sid = df_cone["source_id"].to_numpy(np.int64, copy=False)
    true_set = set(int(x) for x in true_ids)
    true_mask = np.array([int(s) in true_set for s in sid], dtype=bool)

    centers = [("center0", ra0, dec0, 0.0, False)]
    if ENABLE_CENTER_REFINEMENT:
        ra, dec = df_cone["ra"].to_numpy(np.float64), df_cone["dec"].to_numpy(np.float64)
        x0, y0 = compute_tangent_plane(ra, dec, ra0, dec0)
        top_mask = np.zeros(len(ra), dtype=bool)
        top_mask[np.argsort(np.hypot(x0, y0))[:min(int(CENTER_REFINE_TOP_R), len(ra))]] = True
        
        idx = np.flatnonzero(top_mask)
        if len(idx) >= max(32, CENTER_REFINE_K + 5):
            P = np.column_stack([x0[idx], y0[idx]]).astype(np.float32, copy=False)
            try: dist, _ = cKDTree(P).query(P, k=min(int(CENTER_REFINE_K) + 1, len(idx)), workers=-1)
            except TypeError: dist, _ = cKDTree(P).query(P, k=min(int(CENTER_REFINE_K) + 1, len(idx)))
            
            peak_idx = idx[int(np.argmin(dist[:, -1]))]
            ra_peak, dec_peak = float(ra[peak_idx]), float(dec[peak_idx])
            dx, dy = compute_tangent_plane(np.array([ra_peak]), np.array([dec_peak]), ra0, dec0)
            
            dx_iter, dy_iter = iter(dx), iter(dy)
            dx_val = float(next(dx_iter)) if len(dx) > 0 else 0.0
            dy_val = float(next(dy_iter)) if len(dy) > 0 else 0.0
            shift_arcmin = float(np.hypot(dx_val, dy_val)) * 60.0
            
            clipped = shift_arcmin > float(CENTER_SHIFT_LIMIT_ARCMIN)
            if clipped:
                scale = float(CENTER_SHIFT_LIMIT_ARCMIN) / max(1e-6, shift_arcmin)
                ra_peak = ra0 + (ra_peak - ra0) * scale
                dec_peak = dec0 + (dec_peak - dec0) * scale
                shift_arcmin = float(CENTER_SHIFT_LIMIT_ARCMIN)
            
            a_blend = float(CENTER_SHIFT_BLEND)
            cra = ra0 * (1.0 - a_blend) + ra_peak * a_blend
            cdec = dec0 * (1.0 - a_blend) + dec_peak * a_blend
            centers.append(("center1_refined", cra, cdec, shift_arcmin, clipped))

    best_out, best_J, best_meta = None, None, None
    for cname, cra, cdec, cshift, cclip in centers:
        cand_idx, pred_local, sup, pack, score, X, pos2, kin2, aux = orchestrate_single_center(df_cone, cra, cdec, rdeg, plx0, pmra0, pmdec0)
        J = float(pack["objective"]) + (0.25 if cclip and pack["n_pred_local"] > 0.60 * pack["cap"] else 0.0)
        if best_J is None or J < best_J: best_J, best_out, best_meta = J, (cand_idx, pred_local, sup, pack), (cname, cra, cdec, cshift, cclip)

    cand_idx, pred_local, sup, pack = best_out
    cname, cra, cdec, cshift, cclip = best_meta
    pred_ids = sid[cand_idx[pred_local]]
    pred_set = set(int(x) for x in pred_ids)

    tp = sum(x in true_set for x in pred_set)
    fp, fn = len(pred_set) - tp, true_mask.sum() - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    res_dict = {
        "cluster": cluster, "n_cone": int(len(df_cone)), "n_true_in_cone": int(true_mask.sum()),
        "cap": int(pack["cap"]), "n_pred": int(len(pred_set)), "precision": float(prec), "recall": float(rec),
        "f1": float((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0),
        "contam": float(1.0 - prec if (tp + fp) > 0 else 1.0), "runtime_s": float(_now() - t0),
        "center_mode": str(cname), "center_shift_arcmin": float(cshift), "objective": float(best_J), "tag": str(pack["tag"])
    }
    plot_cache = {"pred_ids": list(pred_set), "true_ids": list(true_set), "lit_ra0": float(ra0), "lit_dec0": float(dec0), "mctnc_ra0": float(cra), "mctnc_dec0": float(cdec)}
    return res_dict, plot_cache

# =============================================================================
# [Post-Processing & Astrophysical Audit Visualizations]
# =============================================================================
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

def format_limits_apjs(x: np.ndarray, q: Tuple[float, float]=(0.01, 0.99), pad: float=0.05) -> Tuple[float, float]:
    """Calculates plotting limits safely ignoring outliers."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0: return (-1.0, 1.0)
    
    q_iter = iter(q)
    q_lo, q_hi = float(next(q_iter)), float(next(q_iter))
    lo, hi = np.percentile(x, [q_lo * 100.0, q_hi * 100.0])
    lo, hi = float(lo), float(hi)
    
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo == hi):
        lo, hi = float(np.min(x)), float(np.max(x))
    rng = max(hi - lo, 1e-6)
    return float(lo - pad * rng), float(hi + pad * rng)

def transform_to_absolute_mag(g_mag: pd.Series, plx_mas: pd.Series) -> np.ndarray:
    """Safe absolute magnitude formulation suppressing negative parallax errors."""
    g, p = g_mag.to_numpy(float), plx_mas.to_numpy(float)
    out = np.full_like(g, np.nan, dtype=float)
    m = (p > 0) & np.isfinite(p) & np.isfinite(g)
    out[m] = g[m] - 10.0 + 5.0 * np.log10(p[m])
    return out

def derive_astrophysical_properties(df: pd.DataFrame, is_member: np.ndarray, true_mask: np.ndarray, cra: float, cdec: float) -> dict:
    """Computes spatial and photometric aggregations for diagnostic attribution."""
    n_members = int(is_member.sum())
    if n_members < 1:
        return {"r_50_deg": 0.0, "r_90_deg": 0.0, "mean_plx": np.nan, "n_catalog_only": int(true_mask.sum()), "n_pred_only": 0, "x_deg": np.array([]), "y_deg": np.array([])}
        
    c_ra, c_dec = resolve_column_name(df, ["ra", "RA"]), resolve_column_name(df, ["dec", "DE"])
    x_deg, y_deg = compute_tangent_plane(df[c_ra].to_numpy(), df[c_dec].to_numpy(), cra, cdec)
    
    dist_from_center = np.hypot(x_deg[is_member], y_deg[is_member])
    r_50 = np.percentile(dist_from_center, 50)
    r_90 = np.percentile(dist_from_center, 90)
    mean_plx = np.nanmedian(df["parallax"][is_member]) if "parallax" in df.columns else np.nan
    
    tp = (is_member & true_mask).sum()
    fp = is_member.sum() - tp
    fn = true_mask.sum() - tp
    
    return {
        "r_50_deg": r_50, "r_90_deg": r_90, "mean_plx": mean_plx, 
        "n_catalog_only": fn, "n_pred_only": fp, "x_deg": x_deg, "y_deg": y_deg
    }

def execute_method_agnostic_audit(df: pd.DataFrame, true_mask: np.ndarray, lit_ra: float, lit_dec: float) -> dict:
    """
    STRICTLY METHOD-AGNOSTIC: Evaluates the literature truth without ANY dependency 
    on M-CTNC's refined center. Empirical calibration thresholds are used to flag.
    """
    if true_mask.sum() < 3:
        return {"suspect": False, "fatal": False, "reasons": "Too few true members for independent audit", "r_median": 0.0, "pm_disp": 0.0, "plx_mad": 0.0}
        
    df_true = df[true_mask]
    c_ra, c_dec = resolve_column_name(df_true, ["ra", "RA"]), resolve_column_name(df_true, ["dec", "DE"])
    ra_ref = lit_ra if np.isfinite(lit_ra) else np.nanmedian(df_true[c_ra])
    dec_ref = lit_dec if np.isfinite(lit_dec) else np.nanmedian(df_true[c_dec])
    
    x, y = compute_tangent_plane(df_true[c_ra].to_numpy(), df_true[c_dec].to_numpy(), ra_ref, dec_ref)
    r_median = float(np.median(np.hypot(x, y)))
    
    pmra_mad = compute_robust_mad_scale(df_true["pmra"].values, 0.0) if "pmra" in df_true.columns else 0.0
    pmdec_mad = compute_robust_mad_scale(df_true["pmdec"].values, 0.0) if "pmdec" in df_true.columns else 0.0
    pm_disp = float(np.hypot(pmra_mad, pmdec_mad))
    plx_mad = float(compute_robust_mad_scale(df_true["parallax"].values, 0.0)) if "parallax" in df_true.columns else 0.0

    suspect, fatal, reasons = False, False, []
    
    # Extreme Sanity Check (Detects mapping errors like UBC1194)
    if r_median > 15.0: fatal = True; suspect = True; reasons.append(f"FATAL Spatial Disruption (R_med={r_median:.2f}°)")
    elif r_median > 0.45: suspect = True; reasons.append(f"Diffuse Spatial (R_med={r_median:.2f}°)")
    
    if pm_disp > 1.2: suspect = True; reasons.append(f"Diffuse Kinematics (PM_MAD={pm_disp:.2f})")
    if plx_mad > 0.4: suspect = True; reasons.append(f"Diffuse Distance (Plx_MAD={plx_mad:.2f})")

    return {"suspect": suspect, "fatal": fatal, "reasons": "; ".join(reasons) if reasons else "Clean", "r_median": r_median, "pm_disp": pm_disp, "plx_mad": plx_mad}

def _downsample_idx(n: int, max_n: int, seed: int = 0) -> np.ndarray:
    if n <= max_n: return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=max_n, replace=False)

def global_prefetch_photometry(sids_to_fetch: Set[str], cache_path: Path):
    """Circuit-breaker protected, deduplicated ESA database global fetching."""
    global DISABLE_ONLINE_FETCH
    if DISABLE_ONLINE_FETCH or not sids_to_fetch: return

    cache_df = pd.DataFrame(columns=["source_id", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"])
    if cache_path.exists():
        try: cache_df = pd.read_csv(cache_path, dtype={"source_id": str})
        except Exception: pass
        
    sids_in_cache = set(cache_df["source_id"].astype(str))
    missing_sids = list(sids_to_fetch - sids_in_cache)
    
    if not missing_sids:
        print("  -> [CACHE HIT] All required photometry exists locally. Zero network latency.")
        return

    try: 
        from astroquery.gaia import Gaia
    except ImportError: 
        print("  -> [WARNING] 'astroquery' missing. Offline mode enforced.")
        DISABLE_ONLINE_FETCH = True
        return

    print(f"  -> [PREFETCH] Missing local cache for {len(missing_sids)} distinct targets. Connecting to ESA...")
    if len(missing_sids) > 6000: missing_sids = missing_sids[:5000]

    results = []
    from astropy.utils.exceptions import AstropyWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        for i in range(0, len(missing_sids), 500):
            if DISABLE_ONLINE_FETCH: break
            batch = missing_sids[i:i+500]
            query = f"SELECT source_id, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag FROM gaiadr3.gaia_source WHERE source_id IN ({', '.join(batch)})"
            try: 
                res = Gaia.launch_job(query).get_results().to_pandas()
                results.append(res)
            except Exception as e: 
                print(f"\n[CIRCUIT BREAKER TRIGGERED] ESA API failure/rate limit detected: {e}")
                DISABLE_ONLINE_FETCH = True
                break
                
    if results:
        new_phot = pd.concat(results, ignore_index=True)
        new_phot["source_id"] = new_phot["source_id"].astype(str)
        cache_df = pd.concat([cache_df, new_phot], ignore_index=True).drop_duplicates(subset=["source_id"])
        cache_df.to_csv(cache_path, index=False)
        print(f"  -> [SUCCESS] Photometry cache updated. Saved to {cache_path.name}")

def apply_local_photometry_cache(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Merges offline photometry cleanly into target DataFrame."""
    if not cache_path.exists(): return df
    sid_col = resolve_column_name(df, ["source_id", "GaiaEDR3", "gaiaedr3", "SOURCE_ID"])
    if not sid_col: return df
    try: cache_df = pd.read_csv(cache_path, dtype={"source_id": str}).set_index("source_id")
    except Exception: return df
    df_out = df.copy()
    sids_str = df_out[sid_col].astype(str)
    for c in ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']:
        if c in cache_df.columns:
            if c not in df_out.columns: df_out[c] = np.nan
            df_out[c] = df_out[c].fillna(sids_str.map(cache_df[c]))
    return df_out

def render_apjs_diagnostic_panel(cl: str, df_raw: pd.DataFrame, pred: np.ndarray, true: np.ndarray, ra0: float, dec0: float, out_png: Path, title: str, cache_path: Path):
    """
    Generates Reviewer-Grade 4-Panel Plots.
    Explicitly plots TP (Orange), FN (Blue X), and FP (Red Star) to visualize the 
    precision-recall trade-off and attribution analysis.
    """
    cfg = PlotConfig()
    stable_seed = int(hashlib.md5(cl.encode('utf-8')).hexdigest()[:8], 16)
    df = apply_local_photometry_cache(df_raw, cache_path)
    
    mask_tp = pred & true
    mask_fn = (~pred) & true
    mask_fp = pred & (~true)
    
    bg_idx = _downsample_idx(len(df), cfg.max_points_all, seed=stable_seed)
    plot_mask = np.zeros(len(df), dtype=bool)
    plot_mask[bg_idx] = True; plot_mask |= pred; plot_mask |= true
    bg = df.iloc[bg_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=cfg.dpi)
    ax_pm, ax_sky, ax_cmd, ax_plx = axes.ravel()
    c_ra, c_dec, c_pmra, c_pmdec = [resolve_column_name(df, x) for x in [["ra", "RA"], ["dec", "DE"], ["pmra"], ["pmdec"]]]
    
    # --- 1. PM Panel ---
    if c_pmra and c_pmdec:
        ax_pm.scatter(bg[c_pmra], bg[c_pmdec], s=cfg.s_all, alpha=cfg.alpha_all, c='lightgray', zorder=1)
        if mask_fn.sum() > 0: ax_pm.scatter(df.loc[mask_fn, c_pmra], df.loc[mask_fn, c_pmdec], s=cfg.s_fn, alpha=cfg.alpha_fn, c='dodgerblue', marker='x', linewidths=1.5, label="FN (Missed Lit.)", zorder=4)
        if mask_fp.sum() > 0: ax_pm.scatter(df.loc[mask_fp, c_pmra], df.loc[mask_fp, c_pmdec], s=cfg.s_fp, alpha=cfg.alpha_fp, c='crimson', marker='*', edgecolor='white', linewidths=0.5, label="FP (M-CTNC Ext.)", zorder=5)
        if mask_tp.sum() > 0: ax_pm.scatter(df.loc[mask_tp, c_pmra], df.loc[mask_tp, c_pmdec], s=cfg.s_tp, alpha=cfg.alpha_tp, c='darkorange', marker='o', edgecolor='black', linewidths=0.5, label="TP (Consensus Core)", zorder=6)
        ax_pm.set_xlabel(r"$\mu_{\alpha}^*$ (mas/yr)"); ax_pm.set_ylabel(r"$\mu_{\delta}$ (mas/yr)")
        
        px_min, px_max = format_limits_apjs(df.loc[plot_mask, c_pmra])
        py_min, py_max = format_limits_apjs(df.loc[plot_mask, c_pmdec])
        target_pmra = df.loc[pred | true, c_pmra].to_numpy()
        target_pmdec = df.loc[pred | true, c_pmdec].to_numpy()
        if len(target_pmra) > 0:
            px_min = min(px_min, float(np.min(target_pmra)) - 1.0)
            px_max = max(px_max, float(np.max(target_pmra)) + 1.0)
            py_min = min(py_min, float(np.min(target_pmdec)) - 1.0)
            py_max = max(py_max, float(np.max(target_pmdec)) + 1.0)
        ax_pm.set_xlim(px_min, px_max); ax_pm.set_ylim(py_min, py_max); ax_pm.legend(loc="upper left")

    # --- 2. Spatial Panel ---
    if c_ra and c_dec:
        x_a, y_a = compute_tangent_plane(bg[c_ra].to_numpy(), bg[c_dec].to_numpy(), ra0, dec0)
        x_tp, y_tp = compute_tangent_plane(df.loc[mask_tp, c_ra].to_numpy(), df.loc[mask_tp, c_dec].to_numpy(), ra0, dec0)
        x_fn, y_fn = compute_tangent_plane(df.loc[mask_fn, c_ra].to_numpy(), df.loc[mask_fn, c_dec].to_numpy(), ra0, dec0)
        x_fp, y_fp = compute_tangent_plane(df.loc[mask_fp, c_ra].to_numpy(), df.loc[mask_fp, c_dec].to_numpy(), ra0, dec0)
        
        ax_sky.scatter(x_a, y_a, s=cfg.s_all, alpha=cfg.alpha_all, c='lightgray', zorder=1)
        if mask_fn.sum() > 0: ax_sky.scatter(x_fn, y_fn, s=cfg.s_fn, alpha=cfg.alpha_fn, c='dodgerblue', marker='x', linewidths=1.5, zorder=4)
        if mask_fp.sum() > 0: ax_sky.scatter(x_fp, y_fp, s=cfg.s_fp, alpha=cfg.alpha_fp, c='crimson', marker='*', edgecolor='white', linewidths=0.5, zorder=5)
        if mask_tp.sum() > 0: ax_sky.scatter(x_tp, y_tp, s=cfg.s_tp, alpha=cfg.alpha_tp, c='darkorange', marker='o', edgecolor='black', linewidths=0.5, zorder=6)
        
        ax_sky.set_xlabel(r"$\Delta \alpha \cdot \cos(\delta)$ (deg)"); ax_sky.set_ylabel(r"$\Delta \delta$ (deg)")
        
        x_f, y_f = compute_tangent_plane(df.loc[plot_mask, c_ra].to_numpy(), df.loc[plot_mask, c_dec].to_numpy(), ra0, dec0)
        x_min, x_max = format_limits_apjs(x_f)
        y_min, y_max = format_limits_apjs(y_f)
        t_x, t_y = compute_tangent_plane(df.loc[pred | true, c_ra].to_numpy(), df.loc[pred | true, c_dec].to_numpy(), ra0, dec0)
        if len(t_x) > 0:
            x_min = min(x_min, float(np.min(t_x)) - 0.1)
            x_max = max(x_max, float(np.max(t_x)) + 0.1)
            y_min = min(y_min, float(np.min(t_y)) - 0.1)
            y_max = max(y_max, float(np.max(t_y)) + 0.1)
        ax_sky.set_xlim(x_min, x_max); ax_sky.set_ylim(y_min, y_max); ax_sky.invert_xaxis()

    # --- 3. CMD Panel ---
    c_g = resolve_column_name(df, ["phot_g_mean_mag", "Gmag"])
    c_bp = resolve_column_name(df, ["phot_bp_mean_mag", "BPmag"])
    c_rp = resolve_column_name(df, ["phot_rp_mean_mag", "RPmag"])
    c_plx = resolve_column_name(df, ['parallax', 'plx'])

    C_tp, C_fn, C_fp = None, None, None
    if c_bp and c_rp:
        C_tp = df.loc[mask_tp, c_bp].to_numpy(float) - df.loc[mask_tp, c_rp].to_numpy(float)
        C_fn = df.loc[mask_fn, c_bp].to_numpy(float) - df.loc[mask_fn, c_rp].to_numpy(float)
        C_fp = df.loc[mask_fp, c_bp].to_numpy(float) - df.loc[mask_fp, c_rp].to_numpy(float)

    if c_g and C_tp is not None:
        if c_plx:
            M_tp = transform_to_absolute_mag(df.loc[mask_tp, c_g], df.loc[mask_tp, c_plx])
            M_fn = transform_to_absolute_mag(df.loc[mask_fn, c_g], df.loc[mask_fn, c_plx])
            M_fp = transform_to_absolute_mag(df.loc[mask_fp, c_g], df.loc[mask_fp, c_plx])
            ylab = "Absolute $M_G$ (mag)"
        else:
            M_tp, M_fn, M_fp = df.loc[mask_tp, c_g].to_numpy(float), df.loc[mask_fn, c_g].to_numpy(float), df.loc[mask_fp, c_g].to_numpy(float)
            ylab = "Apparent G (mag)"
            
        if mask_fn.sum() > 0: ax_cmd.scatter(C_fn, M_fn, s=cfg.s_fn, alpha=cfg.alpha_fn, c='dodgerblue', marker='x', linewidths=1.5, zorder=4)
        if mask_fp.sum() > 0: ax_cmd.scatter(C_fp, M_fp, s=cfg.s_fp, alpha=cfg.alpha_fp, c='crimson', marker='*', edgecolor='white', linewidths=0.5, zorder=5)
        if mask_tp.sum() > 0: ax_cmd.scatter(C_tp, M_tp, s=cfg.s_tp, alpha=cfg.alpha_tp, c='darkorange', marker='o', edgecolor='black', linewidths=0.5, zorder=6)
        
        ax_cmd.set_xlabel(r"Color $(G_{BP} - G_{RP})$ (mag)"); ax_cmd.set_ylabel(ylab); ax_cmd.invert_yaxis()
        valid_c = np.concatenate([C_tp, C_fn, C_fp]); valid_c = valid_c[np.isfinite(valid_c)]
        valid_m = np.concatenate([M_tp, M_fn, M_fp]); valid_m = valid_m[np.isfinite(valid_m)]
        if len(valid_c) > 0: 
            cx_min, cx_max = format_limits_apjs(valid_c)
            ax_cmd.set_xlim(cx_min, cx_max)
        if len(valid_m) > 0: 
            cy_min, cy_max = format_limits_apjs(valid_m)
            ax_cmd.set_ylim(cy_max, cy_min) 
    else:
        ax_cmd.set_axis_off(); ax_cmd.text(0.5, 0.5, "CMD Unavailable", ha='center', va='center', fontsize=14)

    # --- 4. Parallax Panel ---
    if c_plx:
        p_a = df[c_plx].to_numpy(float)
        p_v = p_a[np.isfinite(p_a)]
        bins = np.linspace(np.quantile(p_v, 0.01), np.quantile(p_v, 0.99), 40) if len(p_v)>0 else 40
        ax_plx.hist(p_v, bins=bins, alpha=0.35, color='lightgray', label='Field', zorder=1)
        if mask_fn.sum() > 0: ax_plx.hist(df.loc[mask_fn, c_plx].dropna(), bins=bins, alpha=0.80, color='dodgerblue', label='FN', zorder=4)
        if mask_fp.sum() > 0: ax_plx.hist(df.loc[mask_fp, c_plx].dropna(), bins=bins, alpha=0.85, color='crimson', label='FP', zorder=5)
        if mask_tp.sum() > 0: ax_plx.hist(df.loc[mask_tp, c_plx].dropna(), bins=bins, alpha=0.90, color='darkorange', label='TP', zorder=6)
        ax_plx.set_xlabel("Parallax (mas)"); ax_plx.set_ylabel("Count"); ax_plx.legend(loc="upper right")

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.95)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def build_global_attribution_matrix(audit_results: List[dict], out_path: Path):
    """Compiles the systematic physical attributions for recall loss into a global summary."""
    if not audit_results: return
    
    categories = {"Faint Only": 0, "Halo Only": 0, "Faint + Halo": 0, "Other": 0, "Fatal/Suspect Truth": 0}
    for r in audit_results:
        if r.get("label_suspect_flag", 0) == 1:
            categories["Fatal/Suspect Truth"] += 1
            continue
        faint, halo = r.get("faint_flag", False), r.get("halo_flag", False)
        if faint and halo: categories["Faint + Halo"] += 1
        elif faint: categories["Faint Only"] += 1
        elif halo: categories["Halo Only"] += 1
        else: categories["Other"] += 1

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    labels = list(categories.keys())
    counts = list(categories.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = ax.bar(labels, counts, color=colors, edgecolor='black')
    ax.set_title("Global Attribution of FN Losses (Conservative Extraction)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of Clusters")
    ax.set_xticklabels(labels, rotation=15, ha='right')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontsize=12)
        
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# [Main Execution Flow]
# =============================================================================
def main():
    global DISABLE_ONLINE_FETCH
    parser = argparse.ArgumentParser(description="M-CTNC Cluster Extraction Pipeline")
    parser.add_argument("--base_dir", type=str, default=None, help="Root directory of the project.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing CSV inputs.")
    parser.add_argument("--offline", action="store_true", help="Force disable ESA Gaia photometric queries.")
    args = parser.parse_args()

    if args.offline: DISABLE_ONLINE_FETCH = True

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (base_dir / "data")
    plots_dir = base_dir / "plots"
    cache_file = data_dir / "mctnc_photometry_cache.csv"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\n[SYSTEM] Booting {PIPELINE_NAME} (Mode: {EXTRACTION_MODE})...")
    print(f"         Strategy: High-purity conservative core extractor" if EXTRACTION_MODE == "CORE_BENCHMARK" else "         Strategy: Halo & tidal structure exploration")
    
    t1_path, t2_path = data_dir / "ocfinder_table1.csv", data_dir / "ocfinder_table2.csv"
    if not t1_path.exists() or not t2_path.exists():
        print(f"[ERROR] Required table1 or table2 not found in {data_dir}.")
        return

    table1, table2 = pd.read_csv(t1_path), pd.read_csv(t2_path)
    
    c1_cl = resolve_column_name(table1, ["Cluster", "cluster", "Name", "name"])
    t1 = table1.copy()
    t1["cluster"] = t1[c1_cl].astype(str).str.strip()
    for col, alts in [("ra0", ["RA_ICRS", "ra", "RA"]), ("dec0", ["DE_ICRS", "dec", "DE"]),
                      ("radius", ["r_deg", "radius", "r"]), ("plx0", ["plx", "parallax", "Plx"]),
                      ("pmra0", ["pmra", "pmRA"]), ("pmdec0", ["pmdec", "pmDE"])]:
        col_name = resolve_column_name(t1, alts)
        t1[col] = pd.to_numeric(t1[col_name], errors="coerce") if col_name else np.nan
    t1_idx = {str(r["cluster"]): r for _, r in t1.iterrows()}

    c2_cl = resolve_column_name(table2, ["Cluster", "cluster", "Name", "name"])
    c2_sid = resolve_column_name(table2, ["source_id", "GaiaEDR3", "gaiaedr3", "SOURCE_ID"])
    t2 = table2.copy()
    sid2 = pd.to_numeric(t2[c2_sid], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    t2["sid"] = np.where(sid2 > 0, sid2, -1)
    true_map = {str(cl): np.unique(g["sid"].to_numpy(np.int64)[g["sid"].to_numpy(np.int64) > 0]) for cl, g in t2.groupby(c2_cl)}

    cone_files = discover_cone_files(base_dir, data_dir)
    clusters_to_run = [(infer_cluster_name(fp), fp) for fp in cone_files if infer_cluster_name(fp) in t1_idx]
    
    print(f"\n=========================================================================")
    print(f" PHASE 1: STRICT BENCHMARK EVALUATION")
    print(f" Executing '{EXTRACTION_MODE}' across {len(clusters_to_run)} targets.")
    print(f"=========================================================================\n")

    benchmark_results = []
    anomaly_targets = []
    f1_scores = []
    global_cache = {}
    global_sids_to_fetch = set()

    for idx, (cluster_name, fp) in enumerate(clusters_to_run, 1):
        df_raw = pd.read_csv(fp, usecols=lambda c: c.strip().lower() in ["source_id", "gaiaedr3", "ra", "ra_icrs", "dec", "de_icrs", "de", "parallax", "plx", "pmra", "pmdec", "pmde", "ruwe", "parallax_error", "e_plx", "pmra_error", "e_pmra", "pmdec_error", "e_pmdec"])
        df_raw.columns = [c.strip() for c in df_raw.columns]
        colmap = {c.lower(): c for c in df_raw.columns}
        for std, alts in [("source_id", ["source_id", "GaiaEDR3", "SOURCE_ID"]), 
                          ("ra", ["ra", "RA_ICRS", "RA"]), ("dec", ["dec", "DE_ICRS", "DE"]), 
                          ("parallax", ["parallax", "plx"]), ("pmra", ["pmra", "pmRA"]), ("pmdec", ["pmdec", "pmDE"]),
                          ("parallax_error", ["parallax_error", "e_plx"]), ("pmra_error", ["pmra_error", "e_pmra"]), ("pmdec_error", ["pmdec_error", "e_pmdec"])]:
            for a in alts:
                if a.lower() in colmap:
                    df_raw.rename(columns={colmap[a.lower()]: std}, inplace=True)
                    break
        
        res, p_cache = process_single_cluster(cluster_name, df_raw, t1_idx[cluster_name], true_map.get(cluster_name, np.array([], dtype=np.int64)))
        f1, prec, rec, contam, exec_time = res["f1"], res["precision"], res["recall"], res["contam"], res["runtime_s"]
        n_pred, n_true = res["n_pred"], res["n_true_in_cone"]
        
        f1_scores.append(f1)
        benchmark_results.append(res)
        global_cache[cluster_name] = p_cache
        
        if (f1 >= 0.999) and (contam <= 0.001) and (rec >= 0.999):
            tier = 0; tier_label = "Perfect Match"
        elif (f1 >= 0.98) or (contam <= 0.02 and rec >= 0.95):
            tier = 1; tier_label = "Tier 1 (Near-perfect)"
        elif prec >= 0.95 and rec < 0.95: 
            tier = 2; tier_label = "Tier 2 (Conservative Core)"
        elif contam > 0.15 or (n_true > 0 and n_pred / n_true >= 1.5):
            tier = 3; tier_label = "Tier 3 (Topological Over-expansion)"
        elif contam > 0.02 and contam <= 0.15:
            tier = 4; tier_label = "Tier 4 (Borderline Extension)"
        else:
            tier = 5; tier_label = "Tier 5 (Mild Mixed Discrepancy)"
            
        print(f"[{idx:03d}/{len(clusters_to_run):03d}] {cluster_name:<10} | Exec: {exec_time:.3f}s | F1: {f1:.3f} | P: {prec:.3f} | R: {rec:.3f} | {tier_label}")
        
        res["tier"] = tier
        res["tier_label"] = tier_label
        
        if tier > 1:
            anomaly_targets.append((cluster_name, fp, res, p_cache, tier))
            global_sids_to_fetch.update(map(str, p_cache["pred_ids"]))
            global_sids_to_fetch.update(map(str, p_cache["true_ids"]))

    print(f"\n[PHASE 1 COMPLETE] Median F1: {np.median(f1_scores):.3f} | Mean F1: {np.mean(f1_scores):.3f}")

    print(f"\n=========================================================================")
    print(f" PHASE 1.5: GLOBAL PHOTOMETRY PREFETCH (CIRCUIT BREAKER PROTECTED)")
    print(f"=========================================================================\n")
    global_prefetch_photometry(global_sids_to_fetch, cache_file)

    print(f"\n=========================================================================")
    print(f" PHASE 2: POST-BENCHMARK ASTROPHYSICAL AUDIT (RECALL ATTRIBUTION)")
    print(f" Executing Method-Agnostic Label Audit for {len(anomaly_targets)} targets.")
    print(f"=========================================================================\n")

    audit_results = []

    for idx, (cluster_name, fp, res, p_cache, tier) in enumerate(anomaly_targets, 1):
        df_raw = pd.read_csv(fp) 
        df_raw.columns = [c.strip() for c in df_raw.columns]
        colmap = {c.lower(): c for c in df_raw.columns}
        for std, alts in [("source_id", ["source_id", "GaiaEDR3", "SOURCE_ID"]), 
                          ("ra", ["ra", "RA_ICRS", "RA"]), ("dec", ["dec", "DE_ICRS", "DE"]), 
                          ("parallax", ["parallax", "plx"]), ("pmra", ["pmra", "pmRA"]), ("pmdec", ["pmdec", "pmDE"])]:
            for a in alts:
                if a.lower() in colmap:
                    df_raw.rename(columns={colmap[a.lower()]: std}, inplace=True)
                    break
        
        df_raw = apply_local_photometry_cache(df_raw, cache_file)
        
        sid_col = resolve_column_name(df_raw, ["source_id", "GaiaEDR3", "gaiaedr3", "SOURCE_ID"])
        is_member = df_raw[sid_col].isin(p_cache["pred_ids"]).to_numpy()
        true_mask = df_raw[sid_col].isin(p_cache["true_ids"]).to_numpy()
        
        phys = derive_astrophysical_properties(df_raw, is_member, true_mask, p_cache["mctnc_ra0"], p_cache["mctnc_dec0"])
        n_cat_only, n_pred_only = phys["n_catalog_only"], phys["n_pred_only"]
        f1, prec, rec = res["f1"], res["precision"], res["recall"]
        
        audit = execute_method_agnostic_audit(df_raw, true_mask, p_cache["lit_ra0"], p_cache["lit_dec0"])
        
        faint_flag, halo_flag = False, False
        mask_tp = is_member & true_mask
        mask_fn = (~is_member) & true_mask
        fn_mag, tp_mag, fn_r, tp_r = 0.0, 0.0, 0.0, 0.0
        
        if n_cat_only > 0 and mask_tp.sum() > 0:
            c_g = resolve_column_name(df_raw, ["phot_g_mean_mag", "Gmag"])
            tp_r = float(np.median(np.hypot(phys["x_deg"][mask_tp], phys["y_deg"][mask_tp])))
            fn_r = float(np.median(np.hypot(phys["x_deg"][mask_fn], phys["y_deg"][mask_fn])))
            if c_g:
                tp_mag = float(df_raw.loc[mask_tp, c_g].median())
                fn_mag = float(df_raw.loc[mask_fn, c_g].median())
                if fn_mag > tp_mag + 0.3: faint_flag = True
            if fn_r > tp_r * 1.3: halo_flag = True
        
        audit_results.append({
            "cluster": cluster_name, "tier": tier, "r50": round(phys["r_50_deg"], 4), "r90": round(phys["r_90_deg"], 4),
            "mean_plx": round(phys["mean_plx"], 4), "n_catalog_only": n_cat_only, "n_pred_only": n_pred_only,
            "label_suspect_flag": int(audit["suspect"]), "fatal_disruption_flag": int(audit["fatal"]),
            "faint_flag": faint_flag, "halo_flag": halo_flag,
            "audit_r_median_deg": round(audit["r_median"], 4),
            "audit_pm_disp_masyr": round(audit["pm_disp"], 4),
            "audit_plx_mad_mas": round(audit["plx_mad"], 4),
            "audit_reasons": audit["reasons"],
            "diagnostic_png_path": f"plots/{cluster_name}_MCTNC_Diagnostic.png"
        })
        
        print(f"[{idx:03d}/{len(anomaly_targets):03d}] AUDIT: {cluster_name} [Tier {tier}]")
        if res["n_pred"] < 1:
            print(f"   => [RESULT] No topological consensus found.\n" + "-"*85)
            continue
            
        print(f"   => [METRICS] F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}")
        
        if audit["fatal"]:
            print(f"   => [CRITICAL WARNING] Fatal Spatial Disruption detected in literature truth (R_med = {audit['r_median']:.2f}°).")
            print(f"      This strongly implies a mapping error or severe catalog corruption. Manual verification required.")
        
        if n_cat_only > 0 and mask_tp.sum() > 0:
            if c_g:
                print(f"   => [RECALL ATTRIBUTION] FN vs TP -> Med G_mag: {fn_mag:.2f} vs {tp_mag:.2f} | Med Radius: {fn_r:.3f}° vs {tp_r:.3f}°")
            else:
                print(f"   => [RECALL ATTRIBUTION] FN vs TP -> Med Radius: {fn_r:.3f}° vs {tp_r:.3f}°")
            if faint_flag: print("      * Physical Evidence: Missed catalog members are systematically fainter (Low SNR).")
            if halo_flag: print("      * Physical Evidence: Missed catalog members are systematically further from core (Diffuse Halo).")

        if audit["suspect"]:
            print(f"   => [METHOD-AGNOSTIC AUDIT] Literature truth flagged as SUSPECT ({audit['reasons']})")
            if n_cat_only > 0:
                print(f"   => [LABEL TENSION] The {n_cat_only} unrecovered catalog members (FN) are consistent with label contamination.")
        else:
            if n_cat_only > 0:
                print(f"   => [LABEL TENSION] Trade-off recognized. {n_cat_only} catalog members outside the extracted conservative core not recovered.")
                
        if n_pred_only > 0:
            print(f"   => [EXTENSIONS] Identified {n_pred_only} extended co-moving candidates (FP).")
            
        title = f"Astrophysical Audit: {cluster_name} | F1: {f1:.3f} | Tier {tier}"
        if audit["suspect"]: title += " [LITERATURE_LABEL_SUSPECT]"
        if audit["fatal"]: title += " [FATAL_DISRUPTION]"
        
        out_png = plots_dir / f"{cluster_name}_MCTNC_Diagnostic.png"
        render_apjs_diagnostic_panel(cluster_name, df_raw, is_member, true_mask, p_cache["mctnc_ra0"], p_cache["mctnc_dec0"], out_png, title, cache_file)
        print(f"   => [RENDER] CMD Diagnostic saved to: plots/{cluster_name}_MCTNC_Diagnostic.png")
        print("-" * 85)

    print(f"\n=========================================================================")
    print(f" PHASE 3: COMPILING APJS DELIVERABLES")
    print(f"=========================================================================\n")

    build_global_attribution_matrix(audit_results, base_dir / "Global_Attribution_Summary.png")
    print("  -> [SUCCESS] Global Attribution Summary Plot generated.")

    master_catalog_rows = []
    
    for cluster_name, fp in clusters_to_run:
        p_cache = global_cache[cluster_name]
        if not p_cache["pred_ids"]: continue 
        
        df_raw = pd.read_csv(fp)
        colmap = {c.strip().lower(): c.strip() for c in df_raw.columns}
        for std, alts in [("source_id", ["source_id", "GaiaEDR3", "SOURCE_ID"]), 
                          ("ra", ["ra", "RA_ICRS", "RA"]), ("dec", ["dec", "DE_ICRS", "DE"]), 
                          ("parallax", ["parallax", "plx"]), ("pmra", ["pmra", "pmRA"]), ("pmdec", ["pmdec", "pmDE"])]:
            for a in alts:
                if a.lower() in colmap:
                    df_raw.rename(columns={colmap[a.lower()]: std}, inplace=True)
                    break
        
        df_raw = apply_local_photometry_cache(df_raw, cache_file)
                    
        sid_col = resolve_column_name(df_raw, ["source_id", "GaiaEDR3", "gaiaedr3", "SOURCE_ID"])
        is_member = df_raw[sid_col].isin(p_cache["pred_ids"]).to_numpy()
        
        mem_df = df_raw[is_member].copy()
        mem_df["Cluster_Name"] = cluster_name
        mem_df["Extraction_Mode"] = EXTRACTION_MODE
        
        cols_to_keep = ["Cluster_Name", "Extraction_Mode", "source_id", "ra", "dec", "parallax", "pmra", "pmdec"]
        phot_cols = [c for c in df_raw.columns if "phot" in c.lower() or "mag" in c.lower() or "color" in c.lower() or "bp" in c.lower() or "rp" in c.lower()]
        cols_to_keep.extend(phot_cols)
        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        
        master_catalog_rows.append(mem_df[[c for c in cols_to_keep if c in mem_df.columns]])

    df_tableA = pd.DataFrame(benchmark_results)[["cluster", "tier", "tier_label", "n_cone", "n_true_in_cone", "n_pred", "precision", "recall", "f1", "contam", "runtime_s", "center_mode", "center_shift_arcmin", "objective"]]
    df_tableA.insert(1, "mode", EXTRACTION_MODE)
    df_tableA.to_csv(base_dir / f"ApJS_TableA_Full_Benchmark_{EXTRACTION_MODE}.csv", index=False)
    print(f"  -> [SUCCESS] Table A: ApJS_TableA_Full_Benchmark_{EXTRACTION_MODE}.csv")

    if audit_results:
        df_tableB = pd.DataFrame(audit_results)
        df_tableB.to_csv(base_dir / f"ApJS_TableB_Anomaly_Audit_{EXTRACTION_MODE}.csv", index=False)
        print(f"  -> [SUCCESS] Table B: ApJS_TableB_Anomaly_Audit_{EXTRACTION_MODE}.csv")

    if master_catalog_rows:
        pd.concat(master_catalog_rows, ignore_index=True).to_csv(base_dir / f"ApJS_TableC_Master_Catalog_{EXTRACTION_MODE}.csv", index=False)
        print(f"  -> [SUCCESS] Table C: ApJS_TableC_Master_Catalog_{EXTRACTION_MODE}.csv")
        
    print("\n" + "="*85)
    print(" [PAPER WRITING WORKFLOW INSTRUCTIONS]")
    print(f" 1. The current run generated Deliverables for '{EXTRACTION_MODE}'.")
    print(f" 2. Change line 55 to EXTRACTION_MODE = 'HALO_EXPLORATION' and run again.")
    print(f" 3. Use Pandas to merge the two Table A's on 'cluster' to create Table C: Trade-off Analysis.")
    print("=====================================================================================")

if __name__ == "__main__":
    main()
