from __future__ import annotations

import argparse
import re
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter


# ============================================================
# Global constants
# ============================================================

EXACT_TOL = 1e-12
MATERIAL_DELTA_F1 = 0.01

COLORS = {
    "navy": "#123B5D",
    "blue": "#2F6C99",
    "sky": "#6FA8DC",
    "teal": "#1B9E77",
    "green": "#4DAF4A",
    "gold": "#D8A31A",
    "orange": "#F28E2B",
    "red": "#C73E1D",
    "crimson": "#A61C3C",
    "purple": "#6A51A3",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
    "dark": "#1A1A1A",
    "stable": "#8C8C8C",
    "panel_bg": "#FBFCFE",
    "panel_edge": "#D3D9E2",
}

TIER_COLOR_MAP = {
    0: "#4C78A8",
    1: "#72B7B2",
    2: "#F58518",
    3: "#E45756",
    4: "#B279A2",
}

PARAM_PRETTY = {
    "anchor_n": "Anchor number",
    "beta_shift": r"Beta shift $\beta$",
    "candidate_mix": "Candidate mix",
    "cap_backoff_ratio": "Cap backoff ratio",
    "center_blend": "Center blend",
    "center_limit": "Center limit",
    "k_set": r"Neighborhood set $K$",
    "objective_weights": "Objective weights",
    "ruwe_max": "RUWE max",
    "support_tau": r"Support threshold $\tau$",
}

CENTER_EFFECT_ORDER = ["improved", "stable", "degraded"]
CENTER_EFFECT_COLORS = {
    "improved": COLORS["teal"],
    "stable": COLORS["stable"],
    "degraded": COLORS["crimson"],
}


# ============================================================
# Data bundle
# ============================================================

@dataclass
class EvidenceBundle:
    base_dir: Path
    package_xlsx: Path
    population_df: pd.DataFrame
    center_df: pd.DataFrame
    exact_df: pd.DataFrame
    population_csv: Optional[pd.DataFrame] = None
    center_improved_csv: Optional[pd.DataFrame] = None
    center_degraded_csv: Optional[pd.DataFrame] = None
    mismatch_csv: Optional[pd.DataFrame] = None


# ============================================================
# Utilities
# ============================================================

def setup_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 17,
        "axes.titleweight": "semibold",
        "axes.labelsize": 12.5,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#BBBBBB",
        "grid.alpha": 0.32,
        "grid.linewidth": 0.8,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.dpi": 150,
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def savefig(fig: mpl.figure.Figure, path: Path) -> None:
    fig.savefig(path, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pct_str(x: float, digits: int = 1) -> str:
    return f"{100.0 * float(x):.{digits}f}%"


def clean_tier_label(label: object, tier: Optional[int] = None) -> str:
    if isinstance(label, str) and label.strip():
        return label.strip()
    if tier is None or (isinstance(tier, float) and np.isnan(tier)):
        return "Unknown tier"
    tier = int(tier)
    mapping = {
        0: "Perfect Match",
        1: "Tier 1 (Near-perfect)",
        2: "Tier 2 (Conservative Core)",
        3: "Tier 3 (Topological Over-expansion)",
        4: "Tier 4 (Borderline Extension)",
    }
    return mapping.get(tier, f"Tier {tier}")


def wrapped(text: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def exact_match_mask(exact_df: pd.DataFrame) -> pd.Series:
    return exact_df["abs_delta_f1"].astype(float) <= EXACT_TOL


def material_center_mask(center_df: pd.DataFrame) -> pd.Series:
    return center_df["abs_delta_f1"].astype(float) > MATERIAL_DELTA_F1


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols_lower.get(cand.lower())
        if c is not None:
            return c
    return None


def rename_if_exists(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    out = sanitize_columns(df)
    rename_map = {}
    lower_map = {c.lower(): c for c in out.columns}
    for target, aliases in mapping.items():
        if target in out.columns:
            continue
        for a in aliases:
            src = lower_map.get(a.lower())
            if src is not None:
                rename_map[src] = target
                break
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def coerce_fraction_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").astype(float)
        if vals.max(skipna=True) > 1.0 + 1e-12:
            vals = vals / 100.0
        return vals
    s = series.astype(str).str.strip()
    pct_mask = s.str.endswith("%")
    s = s.str.rstrip("%")
    vals = pd.to_numeric(s, errors="coerce").astype(float)
    if pct_mask.any() or vals.max(skipna=True) > 1.0 + 1e-12:
        vals = vals / 100.0
    return vals


# ============================================================
# File discovery
# ============================================================

def _valid_xlsx(path: Path) -> bool:
    return (
        path.exists()
        and path.is_file()
        and path.suffix.lower() == ".xlsx"
        and not path.name.startswith("~$")
    )


def _score_candidate(path: Path) -> tuple:
    name = path.name.lower()
    parent = str(path.parent).lower()

    exact_std = 1 if name == "apjs_evidence_package.xlsx" else 0
    exact_polished = 1 if name == "apjs_evidence_package_polished.xlsx" else 0
    contains_apjs = 1 if "apjs" in name else 0
    contains_evidence = 1 if "evidence" in name and "package" in name else 0
    under_package_folder = 1 if "apjs_evidence_package" in parent else 0
    under_polished_folder = 1 if "polished" in parent else 0
    mtime = path.stat().st_mtime

    # Higher is better for the first six, newer is better for mtime
    return (
        exact_polished,
        exact_std,
        contains_apjs,
        contains_evidence,
        under_polished_folder,
        under_package_folder,
        mtime,
    )


def collect_search_roots(explicit_input: Optional[Path]) -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    roots: List[Path] = []
    if explicit_input is not None:
        roots.append(explicit_input.resolve())

    roots.extend([
        script_dir,
        cwd,
        script_dir.parent,
        cwd.parent if cwd.parent != cwd else cwd,
    ])

    # 去重且只保留存在路径
    uniq = []
    seen = set()
    for p in roots:
        try:
            rp = p.resolve()
        except Exception:
            continue
        if rp.exists() and str(rp) not in seen:
            uniq.append(rp)
            seen.add(str(rp))
    return uniq


def locate_package(explicit_input: Optional[Path]) -> Path:
    # 1) 如果用户直接传入 xlsx 文件
    if explicit_input is not None and explicit_input.exists() and explicit_input.is_file():
        if _valid_xlsx(explicit_input):
            return explicit_input.resolve()
        raise FileNotFoundError(f"Provided file is not a valid .xlsx workbook: {explicit_input}")

    roots = collect_search_roots(explicit_input)

    patterns = [
        "APJS_Evidence_Package.xlsx",
        "APJS_Evidence_Package_POLISHED.xlsx",
        "*APJS*Evidence*Package*.xlsx",
        "*Evidence*Package*.xlsx",
        "*.xlsx",
    ]

    candidates: List[Path] = []
    for root in roots:
        if root.is_file():
            if _valid_xlsx(root):
                candidates.append(root.resolve())
            continue

        # 先查根目录
        for pat in patterns:
            for p in root.glob(pat):
                if _valid_xlsx(p):
                    low = p.name.lower()
                    if ("evidence" in low and "package" in low) or low.startswith("apjs_"):
                        candidates.append(p.resolve())

        # 再递归查子目录
        for pat in patterns:
            for p in root.rglob(pat):
                if _valid_xlsx(p):
                    low = p.name.lower()
                    parent_low = str(p.parent).lower()
                    if (
                        ("evidence" in low and "package" in low)
                        or low.startswith("apjs_")
                        or "apjs_evidence_package" in parent_low
                    ):
                        candidates.append(p.resolve())

    # 去重
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        roots_str = "\n".join([f"  - {r}" for r in roots])
        raise FileNotFoundError(
            "APJS_Evidence_Package workbook was not found.\n"
            "Searched roots:\n"
            f"{roots_str}\n"
            "Please make sure the workbook exists in the script directory, the specified --input_dir, "
            "or one of their subfolders."
        )

    candidates = sorted(candidates, key=_score_candidate, reverse=True)
    return candidates[0]


# ============================================================
# Loaders
# ============================================================

def load_bundle(input_hint: Optional[Path]) -> EvidenceBundle:
    package_xlsx = locate_package(input_hint)
    print(f"  -> Using workbook: {package_xlsx}")

    xls = pd.ExcelFile(package_xlsx)
    sheet_names = set(xls.sheet_names)

    def read_sheet(name: str) -> pd.DataFrame:
        return sanitize_columns(pd.read_excel(package_xlsx, sheet_name=name))

    if "population_ranking" not in sheet_names:
        raise ValueError("Workbook lacks required sheet: population_ranking")
    if "exact_match" not in sheet_names:
        raise ValueError("Workbook lacks required sheet: exact_match")

    population_df = read_sheet("population_ranking")

    if "center_wide" in sheet_names:
        center_df = read_sheet("center_wide")
    elif "center_raw" in sheet_names:
        center_df = read_sheet("center_raw")
    else:
        raise ValueError("Workbook lacks center_wide or center_raw sheet.")

    exact_df = read_sheet("exact_match")

    data_root = package_xlsx.parent

    optional = {}
    optional["population_csv"] = (
        pd.read_csv(data_root / "Table01_population_robustness_ranking.csv")
        if (data_root / "Table01_population_robustness_ranking.csv").exists()
        else None
    )
    optional["center_improved_csv"] = (
        pd.read_csv(data_root / "Table03_center_refinement_top_improved.csv")
        if (data_root / "Table03_center_refinement_top_improved.csv").exists()
        else None
    )
    optional["center_degraded_csv"] = (
        pd.read_csv(data_root / "Table04_center_refinement_top_degraded.csv")
        if (data_root / "Table04_center_refinement_top_degraded.csv").exists()
        else None
    )
    optional["mismatch_csv"] = (
        pd.read_csv(data_root / "TableA3_exact_mismatch_cases.csv")
        if (data_root / "TableA3_exact_mismatch_cases.csv").exists()
        else None
    )

    return EvidenceBundle(
        base_dir=data_root,
        package_xlsx=package_xlsx,
        population_df=population_df.copy(),
        center_df=center_df.copy(),
        exact_df=exact_df.copy(),
        **optional,
    )


# ============================================================
# Derivations
# ============================================================

def derive_population_table(df: pd.DataFrame) -> pd.DataFrame:
    out = sanitize_columns(df)

    out = rename_if_exists(out, {
        "parameter": ["parameter", "param", "family"],
        "median_abs_delta_f1": ["median_abs_delta_f1", "median_abs_delta", "median_delta"],
        "mean_abs_delta_f1": ["mean_abs_delta_f1", "mean_abs_delta", "mean_delta"],
        "p90_abs_delta_f1": ["p90_abs_delta_f1", "p90_abs_delta", "p90_delta"],
        "frac_le_0.05_num": [
            "frac_le_0.05_num", "frac_leq_005_num", "frac_le_005_num",
            "frac_leq_005", "frac_le_0.05", "frac_leq_05"
        ],
        "frac_le_0.03": ["frac_le_0.03", "frac_leq_003", "frac_le_003"],
        "frac_le_0.01": ["frac_le_0.01", "frac_leq_001", "frac_le_001"],
    })

    if "parameter" not in out.columns:
        raise ValueError("population_ranking sheet lacks parameter column.")
    if "p90_abs_delta_f1" not in out.columns:
        raise ValueError("population_ranking sheet lacks p90_abs_delta_f1 column.")

    # fraction 0.05
    if "frac_le_0.05_num" not in out.columns:
        raise ValueError("population_ranking sheet lacks a usable robustness-fraction column for |ΔF1|<=0.05.")
    out["frac_le_0.05_num"] = coerce_fraction_column(out["frac_le_0.05_num"])

    # optional fractions
    for c in ["frac_le_0.03", "frac_le_0.01"]:
        if c in out.columns:
            out[c] = coerce_fraction_column(out[c])

    # numeric coercion
    out["p90_abs_delta_f1"] = pd.to_numeric(out["p90_abs_delta_f1"], errors="coerce").fillna(0.0)

    if "mean_abs_delta_f1" in out.columns:
        out["mean_abs_delta_f1"] = pd.to_numeric(out["mean_abs_delta_f1"], errors="coerce")
    elif "median_abs_delta_f1" in out.columns:
        out["mean_abs_delta_f1"] = pd.to_numeric(out["median_abs_delta_f1"], errors="coerce")
    else:
        out["mean_abs_delta_f1"] = 0.0
    out["mean_abs_delta_f1"] = out["mean_abs_delta_f1"].fillna(0.0)

    if "median_abs_delta_f1" in out.columns:
        out["median_abs_delta_f1"] = pd.to_numeric(out["median_abs_delta_f1"], errors="coerce").fillna(0.0)
    else:
        out["median_abs_delta_f1"] = 0.0

    out["parameter_pretty"] = out["parameter"].map(PARAM_PRETTY).fillna(out["parameter"])

    out["robustness_floor_group"] = np.where(
        out["p90_abs_delta_f1"] <= 1e-12,
        "Stability floor",
        np.where(out["p90_abs_delta_f1"] <= 0.015, "Mild tail sensitivity", "Pronounced tail sensitivity"),
    )

    out = out.sort_values(
        ["p90_abs_delta_f1", "mean_abs_delta_f1", "parameter_pretty"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return out


def derive_center_table(df: pd.DataFrame) -> pd.DataFrame:
    out = sanitize_columns(df)

    out = rename_if_exists(out, {
        "cluster": ["cluster", "cluster_name", "name"],
        "tier": ["tier", "tier_id"],
        "tier_label": ["tier_label", "tier_name"],
        "center_off_f1": ["center_off_f1", "f1_center_off", "off_f1", "f1_off"],
        "center_on_f1": ["center_on_f1", "f1_center_on", "on_f1", "f1_on"],
        "delta_f1": ["delta_f1", "f1_delta", "delta"],
        "abs_delta_f1": ["abs_delta_f1", "absolute_delta_f1"],
        "center_shift_arcmin_on": ["center_shift_arcmin_on", "shift_arcmin_on", "shift_arcmin"],
        "selected_center_mode_on": ["selected_center_mode_on", "center_mode_on", "selected_mode_on"],
        "center_effect": ["center_effect", "effect"],
    })

    required = ["cluster", "center_off_f1", "center_on_f1"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"center table missing required columns: {missing}")

    out["center_off_f1"] = pd.to_numeric(out["center_off_f1"], errors="coerce").fillna(0.0)
    out["center_on_f1"] = pd.to_numeric(out["center_on_f1"], errors="coerce").fillna(0.0)

    if "delta_f1" not in out.columns:
        out["delta_f1"] = out["center_on_f1"] - out["center_off_f1"]
    else:
        out["delta_f1"] = pd.to_numeric(out["delta_f1"], errors="coerce").fillna(
            out["center_on_f1"] - out["center_off_f1"]
        )

    if "abs_delta_f1" not in out.columns:
        out["abs_delta_f1"] = out["delta_f1"].abs()
    else:
        out["abs_delta_f1"] = pd.to_numeric(out["abs_delta_f1"], errors="coerce").fillna(out["delta_f1"].abs())

    if "tier" not in out.columns:
        out["tier"] = -1
    out["tier"] = pd.to_numeric(out["tier"], errors="coerce").fillna(-1).astype(int)

    if "tier_label" not in out.columns:
        out["tier_label"] = [clean_tier_label(None, t) for t in out["tier"]]
    else:
        out["tier_label"] = [clean_tier_label(a, b) for a, b in zip(out["tier_label"], out["tier"])]

    if "center_shift_arcmin_on" not in out.columns:
        out["center_shift_arcmin_on"] = 0.0
    out["center_shift_arcmin_on"] = pd.to_numeric(out["center_shift_arcmin_on"], errors="coerce").fillna(0.0)

    if "selected_center_mode_on" not in out.columns:
        out["selected_center_mode_on"] = "NA"
    out["selected_center_mode_on"] = out["selected_center_mode_on"].astype(str)

    if "center_effect" not in out.columns:
        out["center_effect"] = np.where(
            out["delta_f1"] > MATERIAL_DELTA_F1,
            "improved",
            np.where(out["delta_f1"] < -MATERIAL_DELTA_F1, "degraded", "stable"),
        )
    else:
        out["center_effect"] = out["center_effect"].astype(str).str.lower().replace({
            "neutral": "stable"
        })
        mask_bad = ~out["center_effect"].isin(CENTER_EFFECT_ORDER)
        out.loc[mask_bad, "center_effect"] = np.where(
            out.loc[mask_bad, "delta_f1"] > MATERIAL_DELTA_F1,
            "improved",
            np.where(out.loc[mask_bad, "delta_f1"] < -MATERIAL_DELTA_F1, "degraded", "stable"),
        )

    out["is_material_case"] = material_center_mask(out)
    return out.reset_index(drop=True)


def derive_exact_table(df: pd.DataFrame) -> pd.DataFrame:
    out = sanitize_columns(df)

    out = rename_if_exists(out, {
        "cluster": ["cluster", "cluster_name", "name"],
        "abs_delta_f1": ["abs_delta_f1", "absolute_delta_f1"],
        "delta_f1": ["delta_f1", "f1_delta", "delta"],
        "rerun_f1": ["rerun_f1", "reproduced_f1", "current_f1"],
        "baseline_f1": ["baseline_f1", "official_f1", "benchmark_f1"],
        "center_mode_match": ["center_mode_match", "mode_match", "center_match"],
        "objective_abs_diff": ["objective_abs_diff", "abs_objective_diff", "objective_delta_abs"],
        "rerun_center_mode": ["rerun_center_mode", "current_center_mode"],
        "official_center_mode": ["official_center_mode", "baseline_center_mode", "benchmark_center_mode"],
    })

    if "cluster" not in out.columns:
        out["cluster"] = [f"case_{i+1}" for i in range(len(out))]

    if "abs_delta_f1" not in out.columns:
        if {"rerun_f1", "baseline_f1"}.issubset(out.columns):
            out["rerun_f1"] = pd.to_numeric(out["rerun_f1"], errors="coerce").fillna(0.0)
            out["baseline_f1"] = pd.to_numeric(out["baseline_f1"], errors="coerce").fillna(0.0)
            out["abs_delta_f1"] = (out["rerun_f1"] - out["baseline_f1"]).abs()
        elif "delta_f1" in out.columns:
            out["abs_delta_f1"] = pd.to_numeric(out["delta_f1"], errors="coerce").abs().fillna(0.0)
        else:
            raise ValueError("exact_match sheet lacks abs_delta_f1 and cannot infer it.")

    out["abs_delta_f1"] = pd.to_numeric(out["abs_delta_f1"], errors="coerce").fillna(0.0)

    if "center_mode_match" not in out.columns:
        out["center_mode_match"] = 1
    out["center_mode_match"] = pd.to_numeric(out["center_mode_match"], errors="coerce").fillna(1).astype(int)

    if "objective_abs_diff" not in out.columns:
        out["objective_abs_diff"] = 0.0
    out["objective_abs_diff"] = pd.to_numeric(out["objective_abs_diff"], errors="coerce").fillna(0.0)

    return out.sort_values(["abs_delta_f1", "objective_abs_diff"], ascending=[False, False]).reset_index(drop=True)


def make_summary_tables(pop_df: pd.DataFrame, center_df: pd.DataFrame, exact_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    center_counts = (
        center_df["center_effect"]
        .value_counts()
        .reindex(CENTER_EFFECT_ORDER, fill_value=0)
        .rename_axis("center_effect")
        .reset_index(name="n_cluster")
    )
    center_counts["fraction"] = center_counts["n_cluster"] / max(len(center_df), 1)
    center_counts["fraction_label"] = center_counts["fraction"].map(lambda x: pct_str(x, 1))

    material_center_cases = center_df.loc[center_df["is_material_case"]].copy()
    material_center_cases = material_center_cases.sort_values("delta_f1", ascending=False).reset_index(drop=True)

    mismatch_cases = exact_df.loc[(~exact_match_mask(exact_df)) | (exact_df["center_mode_match"] == 0)].copy()
    mismatch_cases = mismatch_cases.sort_values(["abs_delta_f1", "objective_abs_diff"], ascending=[False, False]).reset_index(drop=True)

    exact_match_count = int(exact_match_mask(exact_df).sum())
    exact_match_fraction = exact_match_count / len(exact_df)
    center_mode_match_fraction = float(exact_df["center_mode_match"].mean())
    stable_fraction = float((center_df["center_effect"] == "stable").mean())

    overview = pd.DataFrame(
        {
            "metric": [
                "Population size",
                "Exact benchmark reproduction fraction",
                "Exact benchmark reproduction count",
                "Median |ΔF1| vs official benchmark",
                "Center-mode match fraction",
                "Stable center-refinement fraction",
                "Material center-refinement cases",
                "Appendix mismatch cases",
            ],
            "value": [
                len(exact_df),
                exact_match_fraction,
                exact_match_count,
                float(exact_df["abs_delta_f1"].median()),
                center_mode_match_fraction,
                stable_fraction,
                int(center_df["is_material_case"].sum()),
                int(len(mismatch_cases)),
            ],
        }
    )

    return {
        "population_ranking": pop_df,
        "center_summary": center_counts,
        "center_material_cases": material_center_cases,
        "exact_mismatches": mismatch_cases,
        "overview": overview,
    }


# ============================================================
# Figure helpers
# ============================================================

def draw_kpi_box(ax, xy, width, height, title, value, subtitle="", face=None, edge=None):
    face = COLORS["panel_bg"] if face is None else face
    edge = COLORS["panel_edge"] if edge is None else edge

    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.15, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(patch)

    ax.text(x + 0.05 * width, y + 0.72 * height, title, transform=ax.transAxes,
            ha="left", va="center", fontsize=11, color="#555555", weight="semibold")
    ax.text(x + 0.05 * width, y + 0.40 * height, value, transform=ax.transAxes,
            ha="left", va="center", fontsize=24, color="#111111", weight="bold")
    if subtitle:
        ax.text(x + 0.05 * width, y + 0.14 * height, subtitle, transform=ax.transAxes,
                ha="left", va="center", fontsize=9.7, color="#666666")


# ============================================================
# Figures
# ============================================================

def plot_parameter_story(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values(["p90_abs_delta_f1", "mean_abs_delta_f1"], ascending=[True, True]).copy()
    y = np.arange(len(d))
    labels = [PARAM_PRETTY.get(p, p) for p in d["parameter"]]

    fig, ax = plt.subplots(figsize=(13.2, 7.1))
    ax.barh(y, d["p90_abs_delta_f1"], color=COLORS["navy"], alpha=0.92, label=r"P90 $|\Delta F_1|$")
    ax.barh(y, d["mean_abs_delta_f1"], color=COLORS["sky"], alpha=0.88, height=0.46, label=r"Mean $|\Delta F_1|$")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Sensitivity magnitude ($|\Delta F_1|$)")
    ax.set_title("Population-level sensitivity structure of M-CTNC parameter families")
    ax.set_xlim(0, max(0.05, float(d["p90_abs_delta_f1"].max()) * 1.15))

    ax2 = ax.twiny()
    ax2.plot(
        d["frac_le_0.05_num"], y,
        marker="o", markersize=8, linewidth=2.6,
        color=COLORS["teal"],
        label=r"Robust fraction ($|\Delta F_1| \leq 0.05$)"
    )
    ax2.set_xlim(0.88, 1.005)
    ax2.set_xlabel(r"Robust fraction ($|\Delta F_1| \leq 0.05$)")
    ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.axvspan(0, 0.015, color="#EAF4EA", alpha=0.70, zorder=0)
    ax.text(0.001, -0.85, "stability floor / mild-tail region",
            color=COLORS["green"], fontsize=11, weight="semibold")

    for i, (_, row) in enumerate(d.iterrows()):
        p90_val = float(row["p90_abs_delta_f1"])
        robust_val = float(row["frac_le_0.05_num"])
        ax.text(p90_val + 0.0012, i, f"{p90_val:.3f}",
                va="center", ha="left", fontsize=10, color=COLORS["navy"])
        ax2.text(robust_val + 0.0022, i + 0.18, pct_str(robust_val, 1),
                 va="center", ha="left", fontsize=9.3, color=COLORS["teal"])

    if "k_set" in d["parameter"].values:
        idx = d.index[d["parameter"] == "k_set"][0]
        ax.annotate(
            "largest upper-tail response",
            xy=(float(d.loc[idx, "p90_abs_delta_f1"]), d.index.get_loc(idx)),
            xytext=(0.027, d.index.get_loc(idx) + 0.78),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=1.2, color=COLORS["crimson"]),
            fontsize=10.5,
            color=COLORS["crimson"],
            weight="semibold"
        )

    if "beta_shift" in d["parameter"].values:
        idx = d.index[d["parameter"] == "beta_shift"][0]
        ax.annotate(
            "secondary tail sensitivity",
            xy=(float(d.loc[idx, "p90_abs_delta_f1"]), d.index.get_loc(idx)),
            xytext=(0.018, d.index.get_loc(idx) + 1.15),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=1.1, color=COLORS["orange"]),
            fontsize=10,
            color=COLORS["orange"],
            weight="semibold"
        )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_parameter_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()
    d["parameter_pretty"] = d["parameter"].map(PARAM_PRETTY).fillna(d["parameter"])
    d = d.sort_values(["p90_abs_delta_f1", "mean_abs_delta_f1"], ascending=[True, True])

    available_cols = []
    for c in ["frac_le_0.01", "frac_le_0.03", "frac_le_0.05_num"]:
        if c in d.columns:
            available_cols.append(c)

    if not available_cols:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No heatmap-ready robustness fraction columns were found.",
                ha="center", va="center", fontsize=14, weight="semibold")
        savefig(fig, out_path)
        return

    heat = d[available_cols].to_numpy()

    label_map = {
        "frac_le_0.01": r"$|\Delta F_1| \leq 0.01$",
        "frac_le_0.03": r"$|\Delta F_1| \leq 0.03$",
        "frac_le_0.05_num": r"$|\Delta F_1| \leq 0.05$",
    }

    fig, ax = plt.subplots(figsize=(9.8, 6.7))
    im = ax.imshow(heat, aspect="auto", cmap=mpl.cm.YlGnBu, vmin=0.6, vmax=1.0)

    ax.set_xticks(range(len(available_cols)))
    ax.set_xticklabels([label_map[c] for c in available_cols])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d["parameter_pretty"]))
    ax.set_title("Robust-fraction heatmap across perturbation tolerances")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = float(heat[i, j])
            ax.text(j, i, f"{100 * val:.1f}%", ha="center", va="center",
                    color="white" if val < 0.84 else "#111111",
                    fontsize=9.2, weight="semibold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Population fraction")

    fig.tight_layout()
    savefig(fig, out_path)


def plot_center_waterfall(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values("delta_f1").reset_index(drop=True).copy()
    x = np.arange(len(d))
    colors = [CENTER_EFFECT_COLORS.get(v, COLORS["stable"]) for v in d["center_effect"]]

    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.bar(x, d["delta_f1"], color=colors, width=0.88, edgecolor="none")
    ax.axhline(0, color="#222222", lw=1.2)
    ax.axhline(MATERIAL_DELTA_F1, color=COLORS["teal"], lw=1.15, ls="--")
    ax.axhline(-MATERIAL_DELTA_F1, color=COLORS["crimson"], lw=1.15, ls="--")
    ax.fill_between(
        [-3, len(d) + 3], -MATERIAL_DELTA_F1, MATERIAL_DELTA_F1,
        color="#ECECEC", alpha=0.58, zorder=0
    )

    ax.set_xlim(-1.5, len(d) + 1.5)
    ax.set_xlabel(r"Cluster index sorted by $\Delta F_1$")
    ax.set_ylabel(r"$\Delta F_1 = F_1(\mathrm{center\ on}) - F_1(\mathrm{center\ off})$")
    ax.set_title("Population audit of center refinement: overwhelmingly neutral, occasionally decisive")

    material = d.loc[d["is_material_case"]].copy()
    for _, row in material.iterrows():
        idx = int(row.name)
        yv = float(row["delta_f1"])
        offset = 0.02 if yv >= 0 else -0.03
        va = "bottom" if yv >= 0 else "top"
        ax.text(idx, yv + offset, str(row["cluster"]), rotation=70, ha="center", va=va,
                fontsize=9.4, color=CENTER_EFFECT_COLORS[row["center_effect"]], weight="semibold")

    stable_n = int((d["center_effect"] == "stable").sum())
    imp_n = int((d["center_effect"] == "improved").sum())
    deg_n = int((d["center_effect"] == "degraded").sum())

    ax.text(0.015, 0.965, f"Stable: {stable_n}/{len(d)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.3, color=COLORS["stable"], weight="semibold")
    ax.text(0.20, 0.965, f"Improved: {imp_n}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.3, color=COLORS["teal"], weight="semibold")
    ax.text(0.33, 0.965, f"Degraded: {deg_n}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.3, color=COLORS["crimson"], weight="semibold")

    fig.tight_layout()
    savefig(fig, out_path)


def plot_center_scatter(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()
    sizes = 48 + 18 * np.sqrt(np.clip(d["center_shift_arcmin_on"].astype(float).values, 0, None))
    colors = [TIER_COLOR_MAP.get(int(t), COLORS["gray"]) for t in d["tier"]]

    fig, ax = plt.subplots(figsize=(8.7, 8.2))
    ax.scatter(
        d["center_off_f1"], d["center_on_f1"],
        s=sizes, c=colors, alpha=0.84,
        edgecolors="white", linewidths=0.65
    )
    ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color=COLORS["navy"])
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$F_1$ without center refinement")
    ax.set_ylabel(r"$F_1$ with center refinement")
    ax.set_title("Center refinement across the full cluster population")

    material = d.loc[d["is_material_case"]].sort_values("abs_delta_f1", ascending=False)
    for _, row in material.iterrows():
        ax.annotate(
            str(row["cluster"]),
            xy=(float(row["center_off_f1"]), float(row["center_on_f1"])),
            xytext=(8, 8 if row["delta_f1"] >= 0 else -12),
            textcoords="offset points",
            fontsize=9.6,
            color=CENTER_EFFECT_COLORS[row["center_effect"]],
            weight="semibold",
        )

    handles = []
    labels = []
    unique_tiers = sorted(t for t in d["tier"].unique() if t >= 0)
    for tier in unique_tiers:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                                  markerfacecolor=TIER_COLOR_MAP.get(int(tier), COLORS["gray"]),
                                  markeredgecolor="white"))
        labels.append(clean_tier_label(None, tier))
    if handles:
        ax.legend(handles, labels, title="Tier", loc="lower right", frameon=True)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_center_material_cases(df: pd.DataFrame, out_path: Path) -> None:
    material = df.loc[df["is_material_case"]].sort_values("delta_f1", ascending=False).copy()

    if material.empty:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.axis("off")
        ax.text(0.5, 0.55, "No material center-refinement cases (|ΔF1| > 0.01) were found.",
                ha="center", va="center", fontsize=14, weight="semibold")
        savefig(fig, out_path)
        return

    n = len(material)
    fig_h = 2.75 * n + 0.7
    fig, axes = plt.subplots(n, 1, figsize=(11.8, fig_h), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, material.iterrows()):
        vals = [float(row["center_off_f1"]), float(row["center_on_f1"])]
        labels = ["center off", "center on"]
        cols = [COLORS["navy"], COLORS["orange"] if row["delta_f1"] >= 0 else COLORS["crimson"]]

        bars = ax.bar(labels, vals, color=cols, width=0.6)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(r"$F_1$")
        ax.grid(axis="y", alpha=0.28)
        ax.set_axisbelow(True)

        title = f'{row["cluster"]}  |  {clean_tier_label(row.get("tier_label"), row.get("tier"))}'
        ax.set_title(title, loc="left", fontsize=12.8, weight="semibold")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10.2, weight="semibold")

        desc = (
            f'ΔF1 = {row["delta_f1"]:+.3f};  '
            f'shift = {float(row.get("center_shift_arcmin_on", 0.0)):.2f} arcmin;  '
            f'mode = {row.get("selected_center_mode_on", "NA")}'
        )
        ax.text(0.98, 0.90, desc, transform=ax.transAxes,
                ha="right", va="top", fontsize=10.2,
                color=CENTER_EFFECT_COLORS[row["center_effect"]], weight="semibold")

    fig.suptitle("Only the material center-refinement cases retained for the manuscript narrative",
                 fontsize=17, weight="semibold")
    savefig(fig, out_path)


def plot_exact_reproducibility(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()
    exact_mask = exact_match_mask(d)
    exact_fraction = float(exact_mask.mean())
    exact_count = int(exact_mask.sum())
    mismatch_case_mask = (~exact_mask) | (d["center_mode_match"] == 0)
    mismatch_count = int(mismatch_case_mask.sum())

    fig = plt.figure(figsize=(12.1, 6.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.42], height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.30)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.axis("off")

    draw_kpi_box(ax0, (0.05, 0.72), 0.90, 0.20, "Exact benchmark reproduction",
                 pct_str(exact_fraction, 1), f"{exact_count}/{len(d)} clusters within |ΔF1| ≤ {EXACT_TOL:.0e}")
    draw_kpi_box(ax0, (0.05, 0.46), 0.90, 0.20, "Center-mode agreement",
                 pct_str(float(d['center_mode_match'].mean()), 1), "rerun center mode matches benchmark mode")
    draw_kpi_box(ax0, (0.05, 0.20), 0.90, 0.20, "Appendix-level mismatch cases",
                 str(mismatch_count), "non-zero F1 mismatch and/or center-mode mismatch")

    ax1 = fig.add_subplot(gs[0, 1])
    sorted_abs = np.sort(d["abs_delta_f1"].to_numpy())[::-1]
    xvals = np.arange(1, len(sorted_abs) + 1)
    ax1.plot(xvals, sorted_abs, color=COLORS["navy"], lw=2.2)
    ax1.fill_between(xvals, 0, sorted_abs, color=COLORS["sky"], alpha=0.35)
    ax1.set_yscale("symlog", linthresh=1e-12)
    ax1.set_xlabel("Cluster rank by absolute mismatch")
    ax1.set_ylabel(r"$|\Delta F_1|$")
    ax1.set_title("Reproducibility audit: mismatch tail")
    ax1.axhline(MATERIAL_DELTA_F1, ls="--", lw=1.0, color=COLORS["crimson"])

    if len(d) > 0 and len(sorted_abs) > 0:
        top = d.iloc[0]
        ax1.annotate(
            str(top["cluster"]),
            xy=(1, float(sorted_abs[0])),
            xytext=(18, 12),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=1.1, color=COLORS["crimson"]),
            fontsize=9.7,
            color=COLORS["crimson"],
            weight="semibold"
        )

    ax2 = fig.add_subplot(gs[1, 1])
    bins = np.array([0, EXACT_TOL, 1e-9, 1e-6, 1e-4, 1e-2, 1.0])
    counts, _ = np.histogram(d["abs_delta_f1"].to_numpy(), bins=bins)
    xpos = np.arange(len(counts))
    labels = ["≤1e-12", "(1e-12,1e-9]", "(1e-9,1e-6]", "(1e-6,1e-4]", "(1e-4,1e-2]", ">1e-2"]

    ax2.bar(xpos, counts, color=COLORS["blue"], alpha=0.92)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, rotation=22, ha="right")
    ax2.set_ylabel("Cluster count")
    ax2.set_title("Mismatch distribution by scale")

    ymax = max(counts.max(), 1)
    for i, c in enumerate(counts):
        ax2.text(i, c + ymax * 0.02, str(int(c)), ha="center", va="bottom", fontsize=9.2, weight="semibold")

    fig.suptitle("Production-consistent reproducibility audit of the M-CTNC evidence package",
                 fontsize=16.5, weight="semibold")
    savefig(fig, out_path)


def plot_overview_dashboard(pop_df: pd.DataFrame, center_df: pd.DataFrame, exact_df: pd.DataFrame, out_path: Path) -> None:
    exact_mask = exact_match_mask(exact_df)
    exact_fraction = float(exact_mask.mean())
    exact_count = int(exact_mask.sum())

    center_match = float(exact_df["center_mode_match"].mean())
    stable_fraction = float((center_df["center_effect"] == "stable").mean())
    material_cases = int(center_df["is_material_case"].sum())

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.88, 1.55, 1.62], hspace=0.40, wspace=0.33)

    axk = fig.add_subplot(gs[0, :])
    axk.axis("off")
    draw_kpi_box(axk, (0.00, 0.12), 0.23, 0.76, "Exact reproduction", pct_str(exact_fraction, 1),
                 f"{exact_count}/{len(exact_df)} exact F1 matches expected")
    draw_kpi_box(axk, (0.255, 0.12), 0.23, 0.76, "Center-mode match", pct_str(center_match, 1),
                 "population-scale implementation consistency")
    draw_kpi_box(axk, (0.51, 0.12), 0.23, 0.76, "Stable center refinement", pct_str(stable_fraction, 1),
                 "population-level neutrality")
    draw_kpi_box(axk, (0.765, 0.12), 0.23, 0.76, "Material center cases", str(material_cases),
                 r"cases with $|\Delta F_1| > 0.01$")

    ax1 = fig.add_subplot(gs[1:, :2])
    d = pop_df.sort_values(["p90_abs_delta_f1", "mean_abs_delta_f1"]).copy()
    y = np.arange(len(d))
    ax1.barh(y, d["p90_abs_delta_f1"], color=COLORS["navy"], alpha=0.88)
    ax1.scatter(d["mean_abs_delta_f1"], y, s=55, color=COLORS["orange"], zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([PARAM_PRETTY.get(p, p) for p in d["parameter"]])
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Sensitivity magnitude ($|\Delta F_1|$)")
    ax1.set_title("Parameter-family sensitivity envelope")

    for i, (_, row) in enumerate(d.iterrows()):
        ax1.text(float(row["p90_abs_delta_f1"]) + 0.001, i, f'{float(row["p90_abs_delta_f1"]):.3f}',
                 va="center", ha="left", fontsize=8.8, color=COLORS["navy"])

    ax2 = fig.add_subplot(gs[1, 2:])
    counts = center_df["center_effect"].value_counts().reindex(CENTER_EFFECT_ORDER, fill_value=0)
    ax2.bar(
        counts.index, counts.values,
        color=[CENTER_EFFECT_COLORS[k] for k in counts.index], width=0.58
    )
    ax2.set_ylabel("Cluster count")
    ax2.set_title("Center-refinement outcome mix")
    ymax = max(int(counts.max()), 1)
    for i, v in enumerate(counts.values):
        ax2.text(i, v + ymax * 0.02, f"{int(v)}", ha="center", va="bottom", fontsize=10.2, weight="semibold")

    ax3 = fig.add_subplot(gs[2, 2:])
    ax3.scatter(
        center_df["center_off_f1"], center_df["center_on_f1"],
        c=[CENTER_EFFECT_COLORS[e] for e in center_df["center_effect"]],
        s=40, alpha=0.82, edgecolors="white", linewidths=0.55
    )
    ax3.plot([0, 1], [0, 1], ls="--", color=COLORS["navy"], lw=1.25)
    ax3.set_xlim(-0.02, 1.03)
    ax3.set_ylim(-0.02, 1.03)
    ax3.set_xlabel(r"$F_1$ without center refinement")
    ax3.set_ylabel(r"$F_1$ with center refinement")
    ax3.set_title("Center-off vs center-on population map")

    fig.suptitle("APJS evidence dashboard for the M-CTNC sensitivity and center-refinement audit",
                 fontsize=18.5, weight="semibold")
    savefig(fig, out_path)


def plot_center_shift_vs_delta(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy()

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for effect in CENTER_EFFECT_ORDER:
        sub = d.loc[d["center_effect"] == effect]
        ax.scatter(
            sub["center_shift_arcmin_on"], sub["delta_f1"],
            s=70,
            color=CENTER_EFFECT_COLORS[effect],
            alpha=0.84,
            edgecolors="white",
            linewidths=0.6,
            label=effect
        )

    ax.axhline(0, color="#222222", lw=1.15)
    ax.axhline(MATERIAL_DELTA_F1, color=COLORS["teal"], lw=1.0, ls="--")
    ax.axhline(-MATERIAL_DELTA_F1, color=COLORS["crimson"], lw=1.0, ls="--")
    ax.set_xlabel("Selected center shift under refinement (arcmin)")
    ax.set_ylabel(r"$\Delta F_1$")
    ax.set_title("Center-refinement response as a function of selected center shift")

    material = d.loc[d["is_material_case"]].sort_values("abs_delta_f1", ascending=False)
    for _, row in material.iterrows():
        ax.annotate(
            str(row["cluster"]),
            xy=(float(row["center_shift_arcmin_on"]), float(row["delta_f1"])),
            xytext=(7, 7 if row["delta_f1"] >= 0 else -12),
            textcoords="offset points",
            fontsize=9.5,
            color=CENTER_EFFECT_COLORS[row["center_effect"]],
            weight="semibold"
        )

    ax.legend(title="Center effect", frameon=True, loc="upper right")
    fig.tight_layout()
    savefig(fig, out_path)


def plot_tier_stratified_center_effect(df: pd.DataFrame, out_path: Path) -> None:
    tab = pd.crosstab(df["tier"], df["center_effect"]).reindex(columns=CENTER_EFFECT_ORDER, fill_value=0)
    tier_labels = [clean_tier_label(None, t) for t in tab.index]

    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    bottom = np.zeros(len(tab))
    for effect in CENTER_EFFECT_ORDER:
        vals = tab[effect].to_numpy()
        ax.bar(
            np.arange(len(tab)), vals, bottom=bottom,
            color=CENTER_EFFECT_COLORS[effect], width=0.62, label=effect
        )
        bottom += vals

    ax.set_xticks(np.arange(len(tab)))
    ax.set_xticklabels([wrapped(t, 20) for t in tier_labels])
    ax.set_ylabel("Cluster count")
    ax.set_title("Tier-stratified distribution of center-refinement outcomes")

    for i in range(len(tab)):
        total = int(tab.iloc[i].sum())
        ax.text(i, total + max(bottom) * 0.02, str(total), ha="center", va="bottom", fontsize=9.8, weight="semibold")

    ax.legend(title="Center effect", frameon=True, loc="upper right")
    fig.tight_layout()
    savefig(fig, out_path)


# ============================================================
# Exports
# ============================================================

def write_readme(out_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    center_summary = tables["center_summary"]
    material_cases = tables["center_material_cases"]
    mismatches = tables["exact_mismatches"]

    lines = []
    lines.append("# APJS Evidence Package – Final Polished Figure Set")
    lines.append("")
    lines.append("This package is a final figure-and-table polish layer for the M-CTNC sensitivity audit.")
    lines.append("It does not alter the main algorithm. It reorganizes the evidence into manuscript-ready, reviewer-facing artifacts.")
    lines.append("")
    lines.append("## Manuscript-level messages")
    lines.append("")
    lines.append("1. The production-consistent rerun reproduces the official benchmark at near-perfect fidelity.")
    lines.append("2. Population-level perturbation sensitivity is concentrated in a small subset of parameter families, rather than being globally fragile.")
    lines.append("3. Center refinement is overwhelmingly neutral at population scale, but decisive for a very small subset of materially mis-centered systems.")
    lines.append("4. All remaining anomalies are explicitly extracted for appendix-level discussion rather than being hidden inside aggregate statistics.")
    lines.append("")
    lines.append("## Center refinement summary")
    lines.append("")
    for _, row in center_summary.iterrows():
        lines.append(f'- {row["center_effect"]}: {int(row["n_cluster"])} clusters ({row["fraction_label"]})')
    lines.append("")
    lines.append("## Material center-refinement cases (|ΔF1| > 0.01)")
    lines.append("")
    if material_cases.empty:
        lines.append("- None")
    else:
        for _, row in material_cases.iterrows():
            lines.append(
                f'- {row["cluster"]}: ΔF1 = {row["delta_f1"]:+.4f}, '
                f'shift = {float(row.get("center_shift_arcmin_on", 0.0)):.2f} arcmin, '
                f'tier = {row.get("tier_label", "NA")}'
            )
    lines.append("")
    lines.append("## Appendix mismatch note")
    lines.append("")
    lines.append(f"- Number of appendix-level mismatch cases: {len(mismatches)}")
    if len(mismatches) > 0:
        top = mismatches.iloc[0]
        lines.append(
            f'- Largest mismatch case: {top["cluster"]} '
            f'(abs ΔF1 = {float(top["abs_delta_f1"]):.4f}, '
            f'rerun mode = {top.get("rerun_center_mode", "NA")}, '
            f'official mode = {top.get("official_center_mode", "NA")})'
        )

    (out_dir / "README_APJS_Final_Polished_Package.md").write_text("\n".join(lines), encoding="utf-8")


def export_tables(out_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    table_dir = out_dir / "tables"
    ensure_dir(table_dir)

    tables["population_ranking"].to_csv(table_dir / "Table01_population_robustness_ranking_polished.csv", index=False)
    tables["center_summary"].to_csv(table_dir / "Table02_center_refinement_population_summary_polished.csv", index=False)
    tables["center_material_cases"].to_csv(table_dir / "Table03_center_refinement_material_cases_only.csv", index=False)
    tables["exact_mismatches"].to_csv(table_dir / "Table04_exact_mismatch_cases_only.csv", index=False)
    tables["overview"].to_csv(table_dir / "Table05_overview_metrics.csv", index=False)

    with pd.ExcelWriter(out_dir / "APJS_Evidence_Package_POLISHED.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet_name = name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# ============================================================
# Builder
# ============================================================

def build_polished_package(bundle: EvidenceBundle, out_root: Optional[Path] = None) -> Path:
    pop_df = derive_population_table(bundle.population_df)
    center_df = derive_center_table(bundle.center_df)
    exact_df = derive_exact_table(bundle.exact_df)
    tables = make_summary_tables(pop_df, center_df, exact_df)

    if out_root is None:
        out_root = bundle.base_dir

    out_dir = out_root / f"APJS_Evidence_Package_POLISHED_{timestamp_now()}"
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    export_tables(out_dir, tables)

    plot_parameter_story(pop_df, fig_dir / "Fig01_parameter_family_story.png")
    plot_parameter_heatmap(pop_df, fig_dir / "Fig02_parameter_family_heatmap.png")
    plot_center_waterfall(center_df, fig_dir / "Fig03_center_refinement_waterfall_polished.png")
    plot_center_scatter(center_df, fig_dir / "Fig04_center_refinement_population_map.png")
    plot_center_material_cases(center_df, fig_dir / "Fig05_center_refinement_material_cases.png")
    plot_exact_reproducibility(exact_df, fig_dir / "Fig06_reproducibility_audit.png")
    plot_overview_dashboard(pop_df, center_df, exact_df, fig_dir / "Fig07_apjs_evidence_dashboard.png")
    plot_center_shift_vs_delta(center_df, fig_dir / "Fig08_center_shift_vs_delta.png")
    plot_tier_stratified_center_effect(center_df, fig_dir / "Fig09_tier_stratified_center_effect.png")

    write_readme(out_dir, tables)
    shutil.make_archive(str(out_dir), "zip", root_dir=str(out_dir))
    return out_dir


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final APJS-oriented figure polishing and evidence packaging for the M-CTNC sensitivity audit."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory containing APJS_Evidence_Package workbook, or a direct .xlsx path. "
             "If omitted, the script will search from the script directory automatically."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional parent directory for the polished package. Defaults to the workbook directory."
    )
    return parser.parse_args()


def main() -> None:
    setup_matplotlib()
    args = parse_args()

    # 核心修补：默认从脚本所在目录搜索，而不是 PowerShell 当前目录
    script_dir = Path(__file__).resolve().parent

    input_hint: Optional[Path]
    if args.input_dir and str(args.input_dir).strip():
        input_hint = Path(args.input_dir).expanduser().resolve()
    else:
        input_hint = script_dir

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    print("[1/5] Loading evidence package ...")
    bundle = load_bundle(input_hint)

    print("[2/5] Building derived tables ...")
    # built inside builder

    print("[3/5] Generating polished manuscript figures ...")
    out_dir = build_polished_package(bundle, output_dir)

    print("[4/5] Writing polished workbook and README ...")
    print("[5/5] Creating zip archive ...")

    print("\n" + "=" * 96)
    print("APJS final polished evidence package completed.")
    print(f"Package folder: {out_dir}")
    print(f"Zip archive   : {str(out_dir)}.zip")
    print("=" * 96)


if __name__ == "__main__":
    main()