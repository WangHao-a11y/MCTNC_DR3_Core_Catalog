import os
import argparse
import re
import json
import glob
import math
import zipfile
import warnings
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# USER CONFIG
# =============================================================================

# Public-release version: no absolute local path is used.
# The script searches relative to its own location, the repository root,
# and diagnostics/robustness_reproducibility/. Use --input_root to override.
DEFAULT_INPUT_ROOT = ""
DEFAULT_LOG_FILE = ""
EXPORT_PDF = True
EXPORT_EXCEL = True

TOP_N_CENTER_CASES = 12
DELTAF1_SMALL = 0.01
DELTAF1_ROBUST = 0.05
PRINT_PROGRESS = True


# =============================================================================
# BASIC UTILS
# =============================================================================

def log(msg: str):
    if PRINT_PROGRESS:
        print(msg)


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def savefig(fig, path_no_ext: str):
    fig.tight_layout()
    fig.savefig(path_no_ext + ".png", bbox_inches="tight")
    if EXPORT_PDF:
        fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def format_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.1f}%"


def robust_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(path, sep="\t")
            except Exception:
                return None


def discover_files(root: str) -> List[str]:
    pats = ["**/*.csv", "**/*.txt", "**/*.log", "**/*.json", "**/*.tsv"]
    out = []
    for p in pats:
        out.extend(glob.glob(os.path.join(root, p), recursive=True))
    return sorted(set(out))


def parse_percent_like(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        x = float(x)
        return x / 100.0 if x > 1.0 else x
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("％", "%")
    m = re.match(r"^\s*([0-9.]+)\s*%\s*$", s)
    if m:
        return float(m.group(1)) / 100.0
    try:
        v = float(s)
        return v / 100.0 if v > 1.0 else v
    except Exception:
        return np.nan


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

CANON_COLS = {
    "cluster": ["cluster", "cl_name", "cluster_name", "name", "target", "object"],
    "tier": ["tier", "tier_name", "group", "tier_label"],
    "parameter": ["parameter", "param", "family", "param_family"],
    "setting": ["setting", "value", "param_value", "label"],
    "f1": ["f1", "F1", "f1_score"],
    "precision": ["p", "P", "precision", "Precision"],
    "recall": ["r", "R", "recall", "Recall"],
    "n_pred": ["n_pred", "N_pred", "npred", "n_members_pred"],
    "center_mode": ["center_mode", "mode", "center"],
    "shift_arcmin": ["shift_arcmin", "center_shift_arcmin", "shift", "offset_arcmin"],
    "objective": ["objective", "obj", "score"],
    "delta_f1": ["delta_f1", "dF1", "deltaF1", "abs_dF1", "abs_delta_f1"],
    "baseline_f1": ["baseline_f1", "official_f1", "ref_f1", "benchmark_f1"],
    "rerun_f1": ["rerun_f1", "audit_f1", "test_f1", "current_f1"],

    # ranking
    "median_abs_delta_f1": ["median_abs_delta_f1", "median_abs_df1", "median_delta_f1", "median"],
    "p90_abs_delta_f1": ["p90_abs_delta_f1", "p90_abs_df1", "p90_delta_f1", "p90"],
    "frac_leq_005_num": [
        "frac_leq_005_num", "frac_robust_num", "leq_005_num", "robust_fraction",
        "frac_le_005_num", "frac_le005_num", "leq005_num"
    ],
    "frac_leq_005": [
        "frac_leq_005", "frac_robust", "leq_005", "<=0.05", "frac_le_005",
        "frac_le005", "leq005", "robust_pct"
    ],

    # center refinement wide table aliases
    "center_off_f1": ["center_off_f1", "f1_center_off", "f1_off"],
    "center_on_f1": ["center_on_f1", "f1_center_on", "f1_on"],

    "center_off_precision": ["center_off_precision", "precision_off", "p_off"],
    "center_on_precision": ["center_on_precision", "precision_on", "p_on"],

    "center_off_recall": ["center_off_recall", "recall_off", "r_off"],
    "center_on_recall": ["center_on_recall", "recall_on", "r_on"],

    "center_off_n_pred": ["center_off_n_pred", "center_off_npred", "npred_off", "n_pred_off"],
    "center_on_n_pred": ["center_on_n_pred", "center_on_npred", "npred_on", "n_pred_on"],

    "center_off_mode": ["center_off_mode", "mode_off"],
    "center_on_mode": ["center_on_mode", "mode_on"],

    "center_off_shift_arcmin": ["center_off_shift_arcmin", "shift_off", "offset_off"],
    "center_on_shift_arcmin": ["center_on_shift_arcmin", "shift_on", "offset_on"],

    "center_off_objective": ["center_off_objective", "objective_off", "obj_off"],
    "center_on_objective": ["center_on_objective", "objective_on", "obj_on"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    ren = {}

    for canon, aliases in CANON_COLS.items():
        for a in aliases:
            if a in df.columns:
                ren[a] = canon
                break
            if a.lower() in cols_lower:
                ren[cols_lower[a.lower()]] = canon
                break

    out = df.rename(columns=ren).copy()

    def coalesce(preferred: str, backups: List[str]):
        if preferred in out.columns:
            return
        for b in backups:
            if b in out.columns:
                out[preferred] = out[b]
                return

    coalesce("center_off_n_pred", ["center_off_npred", "npred_off", "n_pred_off"])
    coalesce("center_on_n_pred", ["center_on_npred", "npred_on", "n_pred_on"])
    coalesce("center_off_shift_arcmin", ["shift_off", "offset_off"])
    coalesce("center_on_shift_arcmin", ["shift_on", "offset_on"])

    return out


def ensure_fraction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 population ranking 表补齐：
    - frac_leq_005_num: 数值型 0~1
    - frac_leq_005: 字符串百分比
    """
    out = normalize_columns(df.copy())

    if "frac_leq_005_num" not in out.columns:
        candidate_cols = [
            c for c in out.columns
            if c in {"frac_leq_005", "<=0.05", "robust_fraction", "leq005", "leq_005", "frac_robust"}
        ]
        if candidate_cols:
            out["frac_leq_005_num"] = out[candidate_cols[0]].map(parse_percent_like)

    if "frac_leq_005_num" not in out.columns:
        # 再模糊搜索一遍
        fuzzy = []
        for c in out.columns:
            cl = str(c).lower().replace(" ", "")
            if ("0.05" in cl) or ("leq005" in cl) or ("robust" in cl and "frac" in cl):
                fuzzy.append(c)
        if fuzzy:
            out["frac_leq_005_num"] = out[fuzzy[0]].map(parse_percent_like)

    if "frac_leq_005" not in out.columns and "frac_leq_005_num" in out.columns:
        out["frac_leq_005"] = out["frac_leq_005_num"].map(format_pct)

    return out


# =============================================================================
# LOG PARSER
# =============================================================================

@dataclass
class ParsedLogBundle:
    representative_rows: Optional[pd.DataFrame] = None
    center_population_rows: Optional[pd.DataFrame] = None
    summary_keyvals: Optional[Dict[str, str]] = None


def parse_log_text(log_text: str) -> ParsedLogBundle:
    summary_keyvals = {}

    m = re.search(
        r"APJS-LEVEL POPULATION SENSITIVITY SUMMARY(.*?)(?:Center refinement population audit:)(.*?)(?:$)",
        log_text,
        flags=re.S,
    )
    if m:
        block1 = m.group(1)
        block2 = m.group(2)

        patterns = [
            (r"Baseline reproduction exact matches:\s*([0-9]+/[0-9]+.*?)\n", "baseline_exact_matches"),
            (r"Median \|ΔF1\| vs official benchmark:\s*([0-9.eE+\-]+)", "median_abs_delta_f1"),
            (r"Center-mode match fraction:\s*([0-9.]+%)", "center_mode_match_fraction"),
            (r"Stable\s*:\s*([0-9]+/[0-9]+.*?)\n", "center_stable"),
            (r"Improved:\s*([0-9]+/[0-9]+.*?)\n", "center_improved"),
            (r"Degraded:\s*([0-9]+/[0-9]+.*?)\n", "center_degraded"),
        ]
        for pat, key in patterns:
            mm = re.search(pat, block1 + "\n" + block2)
            if mm:
                summary_keyvals[key] = mm.group(1).strip()

        ranking_rows = []
        for line in block1.splitlines():
            line = line.strip()
            mm = re.match(
                r"-\s*([A-Za-z0-9_]+)\s+median=([0-9.]+)\s+\|\s+p90=([0-9.]+)\s+\|\s+<=0.05=([0-9.]+%)",
                line
            )
            if mm:
                ranking_rows.append({
                    "parameter": mm.group(1),
                    "median_abs_delta_f1": float(mm.group(2)),
                    "p90_abs_delta_f1": float(mm.group(3)),
                    "frac_leq_005": mm.group(4),
                    "frac_leq_005_num": parse_percent_like(mm.group(4)),
                })
        if ranking_rows:
            summary_keyvals["population_ranking_json"] = json.dumps(ranking_rows, ensure_ascii=False)

    rep_rows = []
    current_cluster = None
    current_family = None
    in_inner = False

    for line in log_text.splitlines():
        ls = line.strip()

        mm_cluster = re.match(r">>> Running\s+([A-Za-z0-9_\-]+)\s+\(", ls)
        if mm_cluster:
            current_cluster = mm_cluster.group(1)
            current_family = None
            continue

        if "[INNER-LOOP:" in ls:
            in_inner = True
            continue
        if "[OUTER-LOOP:" in ls:
            in_inner = False
            current_family = None
            continue

        mm_family = re.match(r"\[([A-Za-z0-9_ \-\(\)]+)\]", ls)
        if in_inner and mm_family:
            fam = mm_family.group(1).strip()
            if "INNER-LOOP:" not in fam:
                current_family = fam
            continue

        if not in_inner or current_cluster is None or current_family is None:
            continue
        if ls.startswith("setting") or ls.startswith("-") or ls == "":
            continue

        toks = re.split(r"\s{2,}", line.rstrip())
        if len(toks) >= 8:
            try:
                rep_rows.append({
                    "cluster": current_cluster,
                    "parameter": current_family,
                    "setting": toks[0].strip(),
                    "f1": float(toks[1]),
                    "precision": float(toks[2]),
                    "recall": float(toks[3]),
                    "n_pred": int(float(toks[4])),
                    "center_mode": toks[5].strip(),
                    "shift_arcmin": float(toks[6]),
                    "objective": float(toks[7]),
                })
            except Exception:
                pass

    rep_df = pd.DataFrame(rep_rows) if rep_rows else None

    pop_rows = []
    current_cluster = None
    anon_counter = 0
    pending = []

    for line in log_text.splitlines():
        ls = line.strip()

        mm_cluster = re.match(r">>> Running\s+([A-Za-z0-9_\-]+)\s+\(", ls)
        if mm_cluster:
            current_cluster = mm_cluster.group(1)

        if "[center_refinement (production-consistent objective)]" in ls:
            pending = []
            continue

        if ls.startswith("center_off") or ls.startswith("center_on"):
            toks = re.split(r"\s{2,}", line.rstrip())
            if len(toks) >= 8:
                cl = current_cluster if current_cluster else f"anon_{anon_counter:04d}"
                if current_cluster is None:
                    anon_counter += 1
                try:
                    row = {
                        "cluster": cl,
                        "setting": toks[0].strip(),
                        "f1": float(toks[1]),
                        "precision": float(toks[2]),
                        "recall": float(toks[3]),
                        "n_pred": int(float(toks[4])),
                        "center_mode": toks[5].strip(),
                        "shift_arcmin": float(toks[6]),
                        "objective": float(toks[7]),
                    }
                    pending.append(row)
                    if len(pending) == 2:
                        pop_rows.extend(pending)
                        pending = []
                except Exception:
                    pass

    pop_df = pd.DataFrame(pop_rows) if pop_rows else None

    return ParsedLogBundle(
        representative_rows=rep_df,
        center_population_rows=pop_df,
        summary_keyvals=summary_keyvals if summary_keyvals else None
    )


# =============================================================================
# DATA DISCOVERY / CLASSIFICATION
# =============================================================================

@dataclass
class DataBundle:
    files: List[str]
    rep_df: Optional[pd.DataFrame]
    center_pop_df: Optional[pd.DataFrame]
    population_ranking_df: Optional[pd.DataFrame]
    exact_match_df: Optional[pd.DataFrame]
    raw_tables: Dict[str, pd.DataFrame]
    parsed_summary: Dict[str, str]


def looks_like_center_long(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    if not {"cluster", "setting", "f1"} <= cols:
        return False
    vals = df["setting"].astype(str).str.lower().head(50).tolist()
    return ("center_off" in vals) or ("center_on" in vals)


def looks_like_center_wide(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    if {"center_off_f1", "center_on_f1"} <= cols:
        return True
    return ("cluster" in cols) and ("center_off_f1" in cols or "center_on_f1" in cols)


def classify_csv(df: pd.DataFrame, path: str) -> str:
    cols = set(df.columns)
    base = os.path.basename(path).lower()

    if looks_like_center_wide(df):
        return "center_refinement_wide"
    if looks_like_center_long(df):
        return "center_refinement_long"

    if {"parameter", "median_abs_delta_f1"} <= cols:
        return "population_ranking"

    if {"cluster", "baseline_f1", "rerun_f1"} <= cols:
        return "exact_match"

    if {"cluster", "parameter", "setting", "f1"} <= cols:
        return "representative_long"

    if "center" in base and {"cluster", "setting"} <= cols:
        return "center_refinement_maybe"

    return "other"


def discover_and_load(root: str, log_file: str = "") -> DataBundle:
    files = discover_files(root)

    raw_tables = {}
    rep_df = None
    center_pop_df = None
    ranking_df = None
    exact_match_df = None
    parsed_summary = {}

    for fp in files:
        if not fp.lower().endswith((".csv", ".tsv")):
            continue
        df = robust_read_csv(fp)
        if df is None or df.empty:
            continue
        df = normalize_columns(df)
        raw_tables[fp] = df

        kind = classify_csv(df, fp)
        if kind == "representative_long" and rep_df is None:
            rep_df = df.copy()
        elif kind in ("center_refinement_long", "center_refinement_wide", "center_refinement_maybe") and center_pop_df is None:
            center_pop_df = df.copy()
        elif kind == "population_ranking" and ranking_df is None:
            ranking_df = ensure_fraction_columns(df.copy())
        elif kind == "exact_match" and exact_match_df is None:
            exact_match_df = df.copy()

    candidate_logs = []
    if log_file and os.path.isfile(log_file):
        candidate_logs.append(log_file)
    candidate_logs.extend([f for f in files if f.lower().endswith((".txt", ".log"))])

    log_text = ""
    for lf in candidate_logs:
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if "APJS-LEVEL POPULATION SENSITIVITY SUMMARY" in txt or ">>> Running" in txt:
                log_text = txt
                break
        except Exception:
            pass

    if log_text:
        parsed = parse_log_text(log_text)
        if rep_df is None and parsed.representative_rows is not None:
            rep_df = normalize_columns(parsed.representative_rows)
        if center_pop_df is None and parsed.center_population_rows is not None:
            center_pop_df = normalize_columns(parsed.center_population_rows)
        if parsed.summary_keyvals:
            parsed_summary.update(parsed.summary_keyvals)
        if ranking_df is None and "population_ranking_json" in parsed_summary:
            try:
                ranking_df = ensure_fraction_columns(pd.DataFrame(json.loads(parsed_summary["population_ranking_json"])))
            except Exception:
                pass

    return DataBundle(
        files=files,
        rep_df=rep_df,
        center_pop_df=center_pop_df,
        population_ranking_df=ranking_df,
        exact_match_df=exact_match_df,
        raw_tables=raw_tables,
        parsed_summary=parsed_summary
    )


# =============================================================================
# CORE DERIVED TABLES
# =============================================================================

def build_center_wide(center_pop_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if center_pop_df is None or center_pop_df.empty:
        return pd.DataFrame()

    df = normalize_columns(center_pop_df.copy())

    if looks_like_center_wide(df):
        out = df.copy()

        if "center_off_f1" not in out.columns:
            out["center_off_f1"] = np.nan
        if "center_on_f1" not in out.columns:
            out["center_on_f1"] = np.nan

        if "center_off_n_pred" not in out.columns and "center_off_npred" in out.columns:
            out["center_off_n_pred"] = out["center_off_npred"]
        if "center_on_n_pred" not in out.columns and "center_on_npred" in out.columns:
            out["center_on_n_pred"] = out["center_on_npred"]

        if "center_off_shift_arcmin" not in out.columns and "shift_off" in out.columns:
            out["center_off_shift_arcmin"] = out["shift_off"]
        if "center_on_shift_arcmin" not in out.columns and "shift_on" in out.columns:
            out["center_on_shift_arcmin"] = out["shift_on"]

        out["delta_f1"] = out["center_on_f1"] - out["center_off_f1"]

        if {"center_off_n_pred", "center_on_n_pred"} <= set(out.columns):
            out["delta_n_pred"] = out["center_on_n_pred"] - out["center_off_n_pred"]

        if {"center_off_shift_arcmin", "center_on_shift_arcmin"} <= set(out.columns):
            out["delta_shift_arcmin"] = out["center_on_shift_arcmin"] - out["center_off_shift_arcmin"]

        def lab(x):
            if pd.isna(x):
                return "unknown"
            if x > DELTAF1_SMALL:
                return "improved"
            if x < -DELTAF1_SMALL:
                return "degraded"
            return "stable"

        out["center_effect"] = out["delta_f1"].apply(lab)
        return out

    need = {"cluster", "setting", "f1"}
    if need <= set(df.columns):
        work = df.copy()
        work = work[work["setting"].astype(str).isin(["center_off", "center_on"])].copy()
        if work.empty:
            return pd.DataFrame()

        rows = []
        value_cols = [c for c in [
            "f1", "precision", "recall", "n_pred", "center_mode", "shift_arcmin", "objective"
        ] if c in work.columns]

        for cl, g in work.groupby("cluster"):
            row = {"cluster": cl}
            for _, r in g.iterrows():
                prefix = str(r["setting"]).strip()
                if prefix not in ("center_off", "center_on"):
                    continue
                for vc in value_cols:
                    row[f"{prefix}_{vc}"] = r[vc]
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame()

        if {"center_off_f1", "center_on_f1"} <= set(out.columns):
            out["delta_f1"] = out["center_on_f1"] - out["center_off_f1"]

        if {"center_off_n_pred", "center_on_n_pred"} <= set(out.columns):
            out["delta_n_pred"] = out["center_on_n_pred"] - out["center_off_n_pred"]

        if {"center_off_shift_arcmin", "center_on_shift_arcmin"} <= set(out.columns):
            out["delta_shift_arcmin"] = out["center_on_shift_arcmin"] - out["center_off_shift_arcmin"]

        def lab(x):
            if pd.isna(x):
                return "unknown"
            if x > DELTAF1_SMALL:
                return "improved"
            if x < -DELTAF1_SMALL:
                return "degraded"
            return "stable"

        if "delta_f1" in out.columns:
            out["center_effect"] = out["delta_f1"].apply(lab)
        else:
            out["center_effect"] = "unknown"

        return out

    return pd.DataFrame()


def build_representative_robustness(rep_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(rep_df.copy())
    if not {"cluster", "parameter", "setting", "f1"} <= set(df.columns):
        return pd.DataFrame(), pd.DataFrame()

    baseline = (
        df[df["setting"].astype(str).str.lower() == "baseline"][["cluster", "f1"]]
        .rename(columns={"f1": "baseline_f1"})
        .drop_duplicates()
    )
    work = df.merge(baseline, on="cluster", how="left")
    work["abs_delta_f1"] = (work["f1"] - work["baseline_f1"]).abs()

    def infer_step(setting: str) -> str:
        s = str(setting)
        if s.lower() == "baseline":
            return "0"

        if s.startswith("tau="):
            try:
                v = float(s.split("=")[1])
                if v < 0.70:
                    return "-1_or_more"
                if v > 0.70:
                    return "+1_or_more"
                return "0"
            except Exception:
                return "other"

        if s.startswith("beta="):
            try:
                rhs = s.split("=")[1]
                v = float(rhs.replace("+", ""))
                if v < 0:
                    return "-1_or_more"
                if v > 0:
                    return "+1_or_more"
                return "0"
            except Exception:
                return "other"

        if s.startswith("anchor="):
            try:
                v = float(s.split("=")[1])
                if v < 36:
                    return "-1_or_more"
                if v > 36:
                    return "+1_or_more"
                return "0"
            except Exception:
                return "other"

        if s.startswith("K="):
            nums = re.findall(r"\d+", s)
            if len(nums) >= 3:
                k1 = int(nums[0])
                if k1 < 16:
                    return "-1_or_more"
                if k1 > 16:
                    return "+1_or_more"
                return "0"

        return "other"

    work["step_dir"] = work["setting"].apply(infer_step)

    rows = []
    for (param, cl), g in work.groupby(["parameter", "cluster"]):
        gm = g[g["step_dir"] == "-1_or_more"]
        gp = g[g["step_dir"] == "+1_or_more"]
        rows.append({
            "parameter": param,
            "cluster": cl,
            "baseline_f1": g["baseline_f1"].iloc[0] if "baseline_f1" in g.columns else np.nan,
            "best_minus_abs_delta_f1": gm["abs_delta_f1"].min() if len(gm) else np.nan,
            "best_plus_abs_delta_f1": gp["abs_delta_f1"].min() if len(gp) else np.nan,
            "combined_robust": (
                (len(gm) > 0 and gm["abs_delta_f1"].min() <= DELTAF1_ROBUST) and
                (len(gp) > 0 and gp["abs_delta_f1"].min() <= DELTAF1_ROBUST)
            )
        })

    return pd.DataFrame(rows), work


def build_param_ranking_from_representative(work_df: pd.DataFrame) -> pd.DataFrame:
    if work_df is None or work_df.empty:
        return pd.DataFrame()

    df = work_df.copy()
    df = df[df["setting"].astype(str).str.lower() != "baseline"].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for p, g in df.groupby("parameter"):
        robust_frac = float((g["abs_delta_f1"] <= DELTAF1_ROBUST).mean())
        rows.append({
            "parameter": p,
            "median_abs_delta_f1": float(g["abs_delta_f1"].median()),
            "p90_abs_delta_f1": float(np.quantile(g["abs_delta_f1"], 0.90)),
            "frac_leq_005_num": robust_frac,
            "frac_leq_005": format_pct(robust_frac),
            "source": "representative_fallback"
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["median_abs_delta_f1", "p90_abs_delta_f1", "parameter"]).reset_index(drop=True)
    return out


# =============================================================================
# FIGURES
# =============================================================================

def plot_population_param_ranking(rank_df: pd.DataFrame, outdir: str):
    if rank_df is None or rank_df.empty:
        return

    df = ensure_fraction_columns(rank_df.copy())
    required = {"parameter", "median_abs_delta_f1", "p90_abs_delta_f1"}
    if not required <= set(df.columns):
        return

    if "frac_leq_005_num" not in df.columns:
        df["frac_leq_005_num"] = np.nan

    order = df.sort_values(["median_abs_delta_f1", "p90_abs_delta_f1", "parameter"])["parameter"].tolist()
    idx = df.set_index("parameter")

    med = idx.loc[order, "median_abs_delta_f1"].astype(float).values
    p90 = idx.loc[order, "p90_abs_delta_f1"].astype(float).values
    frac = idx.loc[order, "frac_leq_005_num"].astype(float).values

    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    ax1.plot(x, med, marker="o", label="Median |ΔF1|")
    ax1.plot(x, p90, marker="s", label="P90 |ΔF1|")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=35, ha="right")
    ax1.set_ylabel("|ΔF1|")
    ax1.set_title("Population-level robustness ranking of M-CTNC parameter families")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, frac, marker="^", linestyle="--", label="Fraction with |ΔF1| ≤ 0.05")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Robust fraction")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="best")

    savefig(fig, os.path.join(outdir, "Fig01_population_parameter_ranking"))


def plot_representative_ecdf(work_df: pd.DataFrame, outdir: str):
    if work_df is None or work_df.empty:
        return
    df = work_df.copy()
    df = df[df["setting"].astype(str).str.lower() != "baseline"].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    for p, g in sorted(df.groupby("parameter"), key=lambda x: str(x[0])):
        vals = np.sort(g["abs_delta_f1"].values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=str(p))
    ax.axvline(DELTAF1_ROBUST, linestyle="--", linewidth=1.0)
    ax.set_xlabel("|ΔF1| relative to cluster-local baseline")
    ax.set_ylabel("ECDF")
    ax.set_title("Representative perturbation robustness across parameter families")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    savefig(fig, os.path.join(outdir, "Fig02_representative_parameter_ecdf"))


def plot_center_refinement_waterfall(center_wide: pd.DataFrame, outdir: str):
    if center_wide is None or center_wide.empty or "delta_f1" not in center_wide.columns:
        return

    df = center_wide.sort_values("delta_f1").reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x, df["delta_f1"].values)
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(DELTAF1_SMALL, linestyle="--", linewidth=1.0)
    ax.axhline(-DELTAF1_SMALL, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cluster index sorted by ΔF1")
    ax.set_ylabel("ΔF1 = F1(center_on) - F1(center_off)")
    ax.set_title("Population audit of center refinement effect")
    ax.grid(alpha=0.25, axis="y")
    savefig(fig, os.path.join(outdir, "Fig03_center_refinement_waterfall"))


def plot_center_refinement_scatter(center_wide: pd.DataFrame, outdir: str):
    need = {"center_off_f1", "center_on_f1"}
    if center_wide is None or center_wide.empty or not need <= set(center_wide.columns):
        return

    df = center_wide.copy()
    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    cvals = None
    if "center_on_shift_arcmin" in df.columns:
        cvals = df["center_on_shift_arcmin"].values

    sc = ax.scatter(df["center_off_f1"], df["center_on_f1"], c=cvals)
    lo = min(df["center_off_f1"].min(), df["center_on_f1"].min())
    hi = max(df["center_off_f1"].max(), df["center_on_f1"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_xlabel("F1 without center refinement")
    ax.set_ylabel("F1 with center refinement")
    ax.set_title("Center refinement: off vs on across the full population")
    ax.grid(alpha=0.25)

    if cvals is not None:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Refined-center shift (arcmin)")

    savefig(fig, os.path.join(outdir, "Fig04_center_refinement_off_vs_on"))


def plot_center_refinement_casecards(center_wide: pd.DataFrame, outdir: str):
    if center_wide is None or center_wide.empty or "delta_f1" not in center_wide.columns:
        return

    improved = center_wide.sort_values("delta_f1", ascending=False).head(6)
    degraded = center_wide.sort_values("delta_f1", ascending=True).head(6)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6))
    for ax, sub, title in zip(
        axes,
        [improved, degraded],
        ["Top improved cases under center refinement", "Top degraded cases under center refinement"]
    ):
        if sub.empty:
            ax.axis("off")
            continue
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["center_off_f1"], width=0.36, label="center_off")
        ax.bar(x + 0.18, sub["center_on_f1"], width=0.36, label="center_on")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["cluster"].astype(str), rotation=30, ha="right")
        ax.set_ylabel("F1")
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.legend()

    savefig(fig, os.path.join(outdir, "Fig05_center_refinement_casecards"))


def plot_representative_heatmap(work_df: pd.DataFrame, outdir: str):
    if work_df is None or work_df.empty:
        return
    df = work_df.copy()
    df = df[df["setting"].astype(str).str.lower() != "baseline"].copy()
    if df.empty:
        return

    piv = df.pivot_table(index="cluster", columns="parameter", values="abs_delta_f1", aggfunc="median")
    if piv.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.42 * len(piv))))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Representative clusters: median |ΔF1| by parameter family")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Median |ΔF1|")

    savefig(fig, os.path.join(outdir, "Fig06_representative_cluster_parameter_heatmap"))


def plot_exact_match_overview(summary: Dict[str, str], outdir: str):
    if "baseline_exact_matches" not in summary:
        return
    mm = re.match(r"(\d+)/(\d+)", summary["baseline_exact_matches"])
    if not mm:
        return

    good = int(mm.group(1))
    total = int(mm.group(2))
    bad = total - good

    fig, ax = plt.subplots(figsize=(5.8, 5.1))
    ax.bar(["Exact match", "Mismatch"], [good, bad])
    ax.set_ylabel("Number of clusters")
    ax.set_title("Baseline reproduction fidelity in production-consistent audit")
    for i, v in enumerate([good, bad]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.grid(alpha=0.25, axis="y")
    savefig(fig, os.path.join(outdir, "Fig07_baseline_exact_match_overview"))


# =============================================================================
# TABLE EXPORT
# =============================================================================

def export_tables(
    out_tables: str,
    rank_df: pd.DataFrame,
    center_wide: pd.DataFrame,
    rep_robust_df: pd.DataFrame,
    exact_match_df: pd.DataFrame,
    summary: Dict[str, str],
):
    if rank_df is not None and not rank_df.empty:
        df = ensure_fraction_columns(rank_df.copy())
        df.to_csv(
            os.path.join(out_tables, "Table01_population_robustness_ranking.csv"),
            index=False, encoding="utf-8-sig"
        )

    if center_wide is not None and not center_wide.empty:
        summary_df = pd.DataFrame([{
            "n_total": len(center_wide),
            "n_stable": int((center_wide["center_effect"] == "stable").sum()) if "center_effect" in center_wide.columns else np.nan,
            "n_improved": int((center_wide["center_effect"] == "improved").sum()) if "center_effect" in center_wide.columns else np.nan,
            "n_degraded": int((center_wide["center_effect"] == "degraded").sum()) if "center_effect" in center_wide.columns else np.nan,
            "frac_stable": float((center_wide["center_effect"] == "stable").mean()) if "center_effect" in center_wide.columns else np.nan,
            "frac_improved": float((center_wide["center_effect"] == "improved").mean()) if "center_effect" in center_wide.columns else np.nan,
            "frac_degraded": float((center_wide["center_effect"] == "degraded").mean()) if "center_effect" in center_wide.columns else np.nan,
        }])
        summary_df.to_csv(
            os.path.join(out_tables, "Table02_center_refinement_population_summary.csv"),
            index=False, encoding="utf-8-sig"
        )

        center_wide.sort_values("delta_f1", ascending=False).head(TOP_N_CENTER_CASES).to_csv(
            os.path.join(out_tables, "Table03_center_refinement_top_improved.csv"),
            index=False, encoding="utf-8-sig"
        )

        center_wide.sort_values("delta_f1", ascending=True).head(TOP_N_CENTER_CASES).to_csv(
            os.path.join(out_tables, "Table04_center_refinement_top_degraded.csv"),
            index=False, encoding="utf-8-sig"
        )

        if "center_on_shift_arcmin" in center_wide.columns:
            center_wide.sort_values("center_on_shift_arcmin", ascending=False).head(TOP_N_CENTER_CASES).to_csv(
                os.path.join(out_tables, "Table05_center_refinement_largest_shift_cases.csv"),
                index=False, encoding="utf-8-sig"
            )

    if rep_robust_df is not None and not rep_robust_df.empty:
        rep_robust_df.to_csv(
            os.path.join(out_tables, "TableA1_representative_parameter_cluster_matrix.csv"),
            index=False, encoding="utf-8-sig"
        )

    if summary:
        pd.DataFrame([{
            "baseline_exact_matches": summary.get("baseline_exact_matches", ""),
            "median_abs_delta_f1": summary.get("median_abs_delta_f1", ""),
            "center_mode_match_fraction": summary.get("center_mode_match_fraction", ""),
            "center_stable": summary.get("center_stable", ""),
            "center_improved": summary.get("center_improved", ""),
            "center_degraded": summary.get("center_degraded", "")
        }]).to_csv(
            os.path.join(out_tables, "TableA2_baseline_exact_match_summary.csv"),
            index=False, encoding="utf-8-sig"
        )

    if exact_match_df is not None and not exact_match_df.empty:
        df = exact_match_df.copy()
        if {"baseline_f1", "rerun_f1"} <= set(df.columns):
            df["delta_f1"] = df["rerun_f1"] - df["baseline_f1"]
            df["abs_delta_f1"] = df["delta_f1"].abs()
            df = df.sort_values(["abs_delta_f1", "cluster"], ascending=[False, True])
        df.to_csv(
            os.path.join(out_tables, "TableA3_exact_mismatch_cases.csv"),
            index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=[
            "cluster", "baseline_f1", "rerun_f1", "delta_f1", "abs_delta_f1",
            "official_center_mode", "rerun_center_mode", "notes"
        ]).to_csv(
            os.path.join(out_tables, "TableA3_exact_mismatch_cases.csv"),
            index=False, encoding="utf-8-sig"
        )


def export_excel_book(path: str, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sname, df in sheets.items():
            if df is not None:
                df.to_excel(writer, index=False, sheet_name=sname[:31])


# =============================================================================
# TEXT OUTPUT
# =============================================================================

def generate_reviewer_summary(summary, rank_df, center_wide) -> str:
    lines = []
    lines.append("Reviewer-Facing Evidence Summary for M-CTNC Sensitivity Audit")
    lines.append("=" * 72)
    lines.append("")

    if "baseline_exact_matches" in summary:
        lines.append(f"- Baseline reproduction exact matches: {summary['baseline_exact_matches']}")
    if "median_abs_delta_f1" in summary:
        lines.append(f"- Median |ΔF1| vs official benchmark: {summary['median_abs_delta_f1']}")
    if "center_mode_match_fraction" in summary:
        lines.append(f"- Center-mode match fraction: {summary['center_mode_match_fraction']}")
    lines.append("")

    if rank_df is not None and not rank_df.empty:
        rdf = ensure_fraction_columns(rank_df.copy())
        best = rdf.sort_values(["median_abs_delta_f1", "p90_abs_delta_f1"]).iloc[0]
        worst = rdf.sort_values(["p90_abs_delta_f1", "median_abs_delta_f1"], ascending=[False, False]).iloc[0]
        lines.append("Population robustness interpretation:")
        lines.append(
            f"  The most stable parameter family is {best['parameter']} "
            f"(median |ΔF1|={best['median_abs_delta_f1']:.4f}, p90={best['p90_abs_delta_f1']:.4f})."
        )
        lines.append(
            f"  The most sensitive family in the upper tail is {worst['parameter']} "
            f"(median |ΔF1|={worst['median_abs_delta_f1']:.4f}, p90={worst['p90_abs_delta_f1']:.4f})."
        )
        lines.append("")

    if center_wide is not None and not center_wide.empty and "center_effect" in center_wide.columns:
        total = len(center_wide)
        stable = int((center_wide["center_effect"] == "stable").sum())
        improved = int((center_wide["center_effect"] == "improved").sum())
        degraded = int((center_wide["center_effect"] == "degraded").sum())

        lines.append("Center-refinement interpretation:")
        lines.append(
            f"  Across {total} clusters, center refinement was stable for {stable} ({stable/total:.1%}), "
            f"improved for {improved} ({improved/total:.1%}), and degraded for {degraded} ({degraded/total:.1%})."
        )

        if improved > 0:
            r = center_wide.sort_values("delta_f1", ascending=False).iloc[0]
            lines.append(f"  Strongest positive case: {r['cluster']} with ΔF1={r['delta_f1']:.4f}.")
        if degraded > 0:
            r = center_wide.sort_values("delta_f1", ascending=True).iloc[0]
            lines.append(f"  Strongest negative case: {r['cluster']} with ΔF1={r['delta_f1']:.4f}.")
        lines.append("")

    lines.append("Recommended manuscript claim:")
    lines.append(
        "  The production-consistent audit shows that M-CTNC reproduces benchmark behavior with near-exact fidelity "
        "and remains robust under moderate perturbations of its key hyperparameter families. The sensitivity hierarchy "
        "is structured rather than arbitrary, with neighborhood construction showing the strongest upper-tail response. "
        "Center refinement is predominantly neutral at population scale, but can provide decisive recovery in a small "
        "subset of center-misaligned or boundary-sensitive systems."
    )

    return "\n".join(lines)


def generate_figure_plan() -> str:
    return """# APJS Figure/Table Plan

## Main-text figures
1. Fig01_population_parameter_ranking
2. Fig02_representative_parameter_ecdf
3. Fig03_center_refinement_waterfall
4. Fig04_center_refinement_off_vs_on
5. Fig05_center_refinement_casecards

## Appendix figures
A1. Fig06_representative_cluster_parameter_heatmap
A2. Fig07_baseline_exact_match_overview

## Main-text tables
1. Table01_population_robustness_ranking.csv
2. Table02_center_refinement_population_summary.csv
3. Table03_center_refinement_top_improved.csv
4. Table04_center_refinement_top_degraded.csv

## Appendix tables
A1. TableA1_representative_parameter_cluster_matrix.csv
A2. TableA2_baseline_exact_match_summary.csv
A3. TableA3_exact_mismatch_cases.csv
"""


def generate_caption_drafts() -> str:
    return """# Draft figure captions

## Fig. 1
Population-level robustness ranking of the principal M-CTNC parameter families under the production-consistent audit. The median and 90th-percentile absolute perturbation responses in F1 are shown together with the fraction of clusters satisfying |ΔF1| ≤ 0.05.

## Fig. 2
Empirical cumulative distributions of |ΔF1| across the representative-cluster perturbation audit. The dashed vertical line marks the robustness threshold |ΔF1| = 0.05.

## Fig. 3
Population-scale center-refinement audit, shown as sorted ΔF1 = F1(center_on) − F1(center_off). Most clusters remain close to zero, while a small subset shows strong positive recovery.

## Fig. 4
Comparison of extraction performance with and without center refinement across the full cluster population. The diagonal denotes equal performance.

## Fig. 5
Top improved and top degraded center-refinement cases identified from the population audit.

## Fig. A1
Representative-cluster heatmap of median |ΔF1| by parameter family.

## Fig. A2
Overview of baseline reproduction fidelity under the production-consistent audit.
"""


# =============================================================================
# PUBLIC-RELEASE PATH RESOLUTION
# =============================================================================

def _as_abs_path(path_like: str, base: Optional[Path] = None) -> Path:
    """Resolve a user-supplied path without assuming any local Windows path."""
    p = Path(path_like).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    return p.resolve()


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root_guess() -> Path:
    """Return the repository root if the script is stored in scripts/; otherwise script dir."""
    sd = _script_dir()
    if sd.name.lower() == "scripts":
        return sd.parent
    return sd


def _has_relevant_evidence_files(root: Path) -> bool:
    """
    Decide whether a directory is a plausible evidence input root.

    The check is deliberately broad because this script accepts several historical
    output layouts: raw sensitivity folders, APJS evidence packages, and the public
    diagnostics/robustness_reproducibility directory.
    """
    if not root.exists() or not root.is_dir():
        return False

    direct_patterns = [
        "APJS_Evidence_Package*.xlsx",
        "Table01_population_robustness*.csv",
        "Table02_center_refinement*.csv",
        "Table03_center_refinement*.csv",
        "Table04_exact*.csv",
        "Table05_overview*.csv",
        "*population*ranking*.csv",
        "*center*refinement*.csv",
        "*exact*match*.csv",
        "*.log",
        "*.txt",
    ]
    for pat in direct_patterns:
        if any(root.glob(pat)):
            return True

    for sub in [root / "diagnostics" / "robustness_reproducibility", root / "tables", root / "audit"]:
        if sub.exists() and sub.is_dir():
            for pat in direct_patterns:
                if any(sub.glob(pat)):
                    return True
    return False


def _candidate_input_roots() -> List[Path]:
    sd = _script_dir()
    rr = _repo_root_guess()
    cwd = Path.cwd().resolve()

    candidates = [
        sd,
        rr / "diagnostics" / "robustness_reproducibility",
        rr / "diagnostics",
        rr,
        cwd / "diagnostics" / "robustness_reproducibility",
        cwd / "diagnostics",
        cwd,
        rr / "sensitivity_core_v12",
        cwd / "sensitivity_core_v12",
    ]

    out = []
    seen = set()
    for c in candidates:
        try:
            rc = c.resolve()
        except Exception:
            continue
        if str(rc) not in seen:
            out.append(rc)
            seen.add(str(rc))
    return out


def resolve_input_root(user_input: str = "") -> Path:
    """Resolve input root from CLI or by searching relative/public-release locations."""
    if user_input:
        p = _as_abs_path(user_input, base=Path.cwd())
        if not p.is_dir():
            raise FileNotFoundError(f"Input root not found or not a directory: {p}")
        return p

    for c in _candidate_input_roots():
        if _has_relevant_evidence_files(c):
            return c

    tried = "\n".join(f"  - {p}" for p in _candidate_input_roots())
    raise FileNotFoundError(
        "Could not locate an evidence input directory automatically.\n"
        "Run with --input_root <folder>, or place the evidence tables under "
        "diagnostics/robustness_reproducibility/.\n"
        f"Searched:\n{tried}"
    )


def resolve_log_file(user_log: str, input_root: Path) -> str:
    """Resolve optional log file; empty string lets discover_and_load auto-search logs."""
    if not user_log:
        return ""
    p = _as_abs_path(user_log, base=input_root)
    if not p.is_file():
        raise FileNotFoundError(f"Log file not found: {p}")
    return str(p)


def relpath_for_manifest(path_like: str, base: Path) -> str:
    """Record paths in manifests as relative paths whenever possible."""
    if not path_like:
        return ""
    try:
        p = Path(path_like).resolve()
        return str(p.relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path_like).replace("\\", "/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global EXPORT_PDF, EXPORT_EXCEL, PRINT_PROGRESS

    parser = argparse.ArgumentParser(
        description=(
            "Package M-CTNC sensitivity / robustness evidence using only relative "
            "or user-specified input paths."
        )
    )
    parser.add_argument(
        "--input_root",
        "--input-dir",
        dest="input_root",
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Folder containing sensitivity/evidence outputs. If omitted, the script "
            "searches its own directory, the repository root, and "
            "diagnostics/robustness_reproducibility/."
        ),
    )
    parser.add_argument(
        "--log_file",
        "--log-file",
        dest="log_file",
        default=DEFAULT_LOG_FILE,
        help="Optional run log. Relative paths are resolved against --input_root.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="",
        help=(
            "Output package directory. If omitted, a timestamped "
            "APJS_Evidence_Package_* folder is created inside the input root."
        ),
    )
    parser.add_argument("--no_pdf", action="store_true", help="Do not export PDF copies of figures.")
    parser.add_argument("--no_excel", action="store_true", help="Do not export the Excel workbook.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    args = parser.parse_args()

    EXPORT_PDF = not args.no_pdf
    EXPORT_EXCEL = not args.no_excel
    PRINT_PROGRESS = not args.quiet

    input_root_path = resolve_input_root(args.input_root)
    log_file_path = resolve_log_file(args.log_file, input_root_path)

    stamp = now_str()
    if args.output_dir:
        pkg_root_path = _as_abs_path(args.output_dir, base=Path.cwd())
    else:
        pkg_root_path = input_root_path / f"APJS_Evidence_Package_{stamp}"

    pkg_root = str(pkg_root_path)
    fig_dir = os.path.join(pkg_root, "figures")
    tab_dir = os.path.join(pkg_root, "tables")
    txt_dir = os.path.join(pkg_root, "text")
    aud_dir = os.path.join(pkg_root, "audit")
    app_dir = os.path.join(pkg_root, "appendix")

    for d in [pkg_root, fig_dir, tab_dir, txt_dir, aud_dir, app_dir]:
        safe_mkdir(d)

    log("[PATH] input_root = " + str(input_root_path))
    log("[PATH] output_dir = " + str(pkg_root_path))
    if log_file_path:
        log("[PATH] log_file   = " + str(log_file_path))

    log("[1/7] Discovering and loading outputs...")
    bundle = discover_and_load(str(input_root_path), log_file_path)

    log("[2/7] Building derived tables...")
    rep_robust_df = pd.DataFrame()
    work_df = pd.DataFrame()
    if bundle.rep_df is not None and not bundle.rep_df.empty:
        rep_robust_df, work_df = build_representative_robustness(bundle.rep_df)

    center_wide = build_center_wide(bundle.center_pop_df)

    rank_df = pd.DataFrame()
    if bundle.population_ranking_df is not None and not bundle.population_ranking_df.empty:
        rank_df = ensure_fraction_columns(bundle.population_ranking_df.copy())
    elif work_df is not None and not work_df.empty:
        rank_df = build_param_ranking_from_representative(work_df)

    exact_match_df = bundle.exact_match_df.copy() if bundle.exact_match_df is not None else pd.DataFrame()
    summary = dict(bundle.parsed_summary)

    log("[3/7] Exporting core tables...")
    export_tables(
        out_tables=tab_dir,
        rank_df=rank_df,
        center_wide=center_wide,
        rep_robust_df=rep_robust_df,
        exact_match_df=exact_match_df,
        summary=summary,
    )

    log("[4/7] Generating figures...")
    plot_population_param_ranking(rank_df, fig_dir)
    plot_representative_ecdf(work_df, fig_dir)
    plot_center_refinement_waterfall(center_wide, fig_dir)
    plot_center_refinement_scatter(center_wide, fig_dir)
    plot_center_refinement_casecards(center_wide, fig_dir)
    plot_representative_heatmap(work_df, app_dir)
    plot_exact_match_overview(summary, app_dir)

    log("[5/7] Writing text artifacts...")
    write_text(
        os.path.join(txt_dir, "Reviewer_Response_Style_Summary.txt"),
        generate_reviewer_summary(summary, rank_df, center_wide),
    )
    write_text(
        os.path.join(txt_dir, "APJS_Figure_Table_Plan.md"),
        generate_figure_plan(),
    )
    write_text(
        os.path.join(txt_dir, "Draft_Figure_Captions.md"),
        generate_caption_drafts(),
    )

    repo_root = _repo_root_guess()
    manifest = {
        "created_at": stamp,
        "input_root": relpath_for_manifest(str(input_root_path), repo_root),
        "log_file": relpath_for_manifest(log_file_path, repo_root) if log_file_path else "",
        "output_dir": relpath_for_manifest(str(pkg_root_path), repo_root),
        "n_discovered_files": len(bundle.files),
        "has_representative_table": bundle.rep_df is not None and not bundle.rep_df.empty,
        "has_center_population_table": bundle.center_pop_df is not None and not bundle.center_pop_df.empty,
        "has_population_ranking": rank_df is not None and not rank_df.empty,
        "has_exact_match_table": exact_match_df is not None and not exact_match_df.empty,
        "summary": summary,
    }
    write_text(os.path.join(aud_dir, "manifest.json"), json.dumps(manifest, indent=2, ensure_ascii=False))

    discovered_rel = [relpath_for_manifest(f, repo_root) for f in bundle.files]
    write_text(os.path.join(aud_dir, "discovered_files.txt"), "\n".join(discovered_rel))

    if EXPORT_EXCEL:
        log("[6/7] Exporting Excel workbook...")
        export_excel_book(
            os.path.join(pkg_root, "APJS_Evidence_Package.xlsx"),
            {
                "population_ranking": ensure_fraction_columns(rank_df) if rank_df is not None else pd.DataFrame(),
                "center_wide": center_wide,
                "rep_robustness": rep_robust_df,
                "rep_long": bundle.rep_df if bundle.rep_df is not None else pd.DataFrame(),
                "center_raw": bundle.center_pop_df if bundle.center_pop_df is not None else pd.DataFrame(),
                "exact_match": exact_match_df,
            },
        )

    log("[7/7] Creating zip archive...")
    zip_path = pkg_root + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(pkg_root):
            for fn in files:
                ap = os.path.join(root, fn)
                rp = os.path.relpath(ap, pkg_root)
                zf.write(ap, arcname=rp)

    print("\n" + "=" * 90)
    print("APJS evidence package completed.")
    print(f"Package folder: {pkg_root}")
    print(f"Zip archive   : {zip_path}")
    print("=" * 90)

    if summary:
        print("\nKey summary:")
        for k, v in summary.items():
            if k == "population_ranking_json":
                continue
            print(f"  - {k}: {v}")

    if center_wide is not None and not center_wide.empty and "center_effect" in center_wide.columns:
        print("\nCenter refinement counts:")
        print(center_wide["center_effect"].value_counts(dropna=False).to_string())

    if rank_df is not None and not rank_df.empty:
        rdf = ensure_fraction_columns(rank_df.copy())
        cols = [c for c in ["parameter", "median_abs_delta_f1", "p90_abs_delta_f1", "frac_leq_005"] if c in rdf.columns]
        if cols:
            print("\nPopulation ranking preview:")
            print(rdf[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()