#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sht0606_official_full.py

SH/T 0606—2005 full calculation implementation with explicit dynamic carbon-number selection.

Run:
python sht0606_official_full.py --input sample.xlsx --output result.xlsx
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple

ALKYLBENZENE_TABLE1 = {
    10: {"m": 134, "K1": 0.1101, "K2": 85}, 11: {"m": 148, "K1": 0.1212, "K2": 63},
    12: {"m": 162, "K1": 0.1323, "K2": 60}, 13: {"m": 176, "K1": 0.1434, "K2": 57},
    14: {"m": 190, "K1": 0.1545, "K2": 54}, 15: {"m": 204, "K1": 0.1656, "K2": 51},
    16: {"m": 218, "K1": 0.1767, "K2": 48}, 17: {"m": 232, "K1": 0.1878, "K2": 45},
    18: {"m": 246, "K1": 0.1989, "K2": 42},
}
NAPHTHALENE_TABLE1 = {
    11: {"m": 142, "L1": 0.1201, "L2": 194}, 12: {"m": 156, "L1": 0.1314, "L2": 166},
    13: {"m": 170, "L1": 0.1425, "L2": 150}, 14: {"m": 184, "L1": 0.1536, "L2": 150},
    15: {"m": 198, "L1": 0.1647, "L2": 150}, 16: {"m": 212, "L1": 0.1758, "L2": 150},
    17: {"m": 226, "L1": 0.1871, "L2": 150}, 18: {"m": 240, "L1": 0.1982, "L2": 150},
}
SIGMAS_ALL = ["Sigma71", "Sigma67", "Sigma123", "Sigma149", "Sigma91", "Sigma103", "Sigma115", "Sigma128", "Sigma141", "Sigma153", "Sigma151", "Sigma177"]

# Table 3 coefficients. Dots in the standard are encoded as 0.
TABLE3 = {
    "paraffins": {
        12:   {"coef": [100,19,0,0,0.4,0,0.5,0,9,0,0,0], "mass":87},
        13:   {"coef": [100,21,0,0,0.4,0,0,0,9,0,0,0], "mass":92},
        14.5: {"coef": [100,23,0.1,0,0.4,0,0,0,10,0,0,0], "mass":97},
        15.5: {"coef": [100,26,0.2,0,0.4,0,0,0,12,0,0,0], "mass":105},
    },
    "monocycloparaffins": {
        12:   {"coef": [4,100,1,0,0,0,1,0,0,1,1,0], "mass":180},
        13:   {"coef": [4,100,1,0,0,0,1,0,0,2,5,0], "mass":191},
        14.5: {"coef": [6,100,1,0,0.2,0,1,0,2,2,7,2], "mass":204},
        15.5: {"coef": [6,100,3,0,3,0,1,0,0.3,2,10,2], "mass":209},
    },
    "dicycloparaffins": {
        13:   {"coef": [2,160,100,0.2,4,0,0.5,0,0.2,0,0,0], "mass":122},
        14.5: {"coef": [1.1,130,100,5,4,0,0,0,0,0,0,0], "mass":156},
        15.5: {"coef": [1.5,150,100,8,5,0,0,0,0,0,0,0], "mass":134},
    },
    "tricycloparaffins": {
        13:   {"coef": [1,175,26,100,15,1,0,0,0.1,0,0,0], "mass":124},
        14.5: {"coef": [1,170,10,100,15,0,0,0,0.1,0,0,0], "mass":158},
        15.5: {"coef": [2,150,20,100,20,3,0,0,0.4,0,0,0], "mass":135},
    },
    "alkylbenzenes": {
        11: {"coef": [0.3,0.7,0.1,1.3,100,9,4.4,0.7,0,0,0,0], "mass":304},
        12: {"coef": [0.3,0.7,0.1,1,100,10,4.5,1,0,0,0,0], "mass":278},
        13: {"coef": [0.4,2,0.2,1.5,100,10,5,1,0,0,0,0], "mass":256},
        14: {"coef": [0.5,3,0.3,2,100,9,5,1,0,0,0,0], "mass":237},
    },
    "indans_tetralins": {
        10: {"coef": [0.2,0.6,0,0,15,100,20,3,0,0,0,0], "mass":288},
        11: {"coef": [0.4,1,0.1,0.1,18,100,28,5.4,1,0,0,0], "mass":288},
        12: {"coef": [0.4,1,1,0.2,17,100,25,7,2.5,0,0,0], "mass":263},
        13: {"coef": [1,2,2,0.3,15,100,25,0,0,0,0,0], "mass":241},
    },
    "indenes_CnH2n_10": {
        10: {"coef": [0.3,0.3,0.4,0,0.6,1.5,100,15,0,0,0,0], "mass":315},
        13: {"coef": [1.7,6.0,4.8,0.9,6.2,20.3,100,13,28,6.1,4.5,0.6], "mass":200},
    },
    "naphthalene": {10: {"coef": [0.5,0.8,0.2,0,0.1,0.6,11.4,100,0,0,0,0], "mass":184}},
    "alkyl_naphthalenes": {
        11: {"coef": [5.2,1.2,0.5,0.1,0.9,0.1,23,0.7,100,0,0,0], "mass":254},
        12: {"coef": [1.5,1.5,7.8,0.7,1,0.1,19,5.6,100,8,7,0], "mass":244},
        13: {"coef": [2,2,4,0.5,1,0.1,18,5.6,100,10,7,0], "mass":224},
    },
    "acenaphthenes_CnH2n_14": {
        12: {"coef": [1,0.3,0,0,0.1,0,0.8,1,8,100,27,0], "mass":214},
        13: {"coef": [1,2,0,0,5,3,0.8,0.7,10,100,20,4], "mass":196},
    },
    "acenaphthylenes_CnH2n_16": {
        12: {"coef": [1,1,0,0,1,0.2,0.3,0.2,1,17,100,0], "mass":224},
        13: {"coef": [1,5,0,0,3,3,2.7,0.1,0,15,100,15], "mass":205},
    },
    "tricyclic_aromatics": {14: {"coef": [0.6,0.7,0,0,18,1.5,1.0,0.8,0.3,3.5,30,100], "mass":205}},
}
ALKYLBENZENE_TO_PARA_CYCLO = {10: 11, 11: 12, 12: 13, 13: 14.5, 14: 15.5}

def nearest(value, candidates):
    arr = np.array(list(candidates), dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(value)))])

def _canon(name):
    return str(name).strip().lower().replace(" ", "").replace("_", "")

def normalize_ms_columns(df):
    rename = {}
    for c in df.columns:
        cc = _canon(c)
        if cc in ("mz", "m/z", "me", "m/e"):
            rename[c] = "mz"
        elif cc in ("intensity", "peak", "height", "abundance"):
            rename[c] = "intensity"
        elif cc in ("sample", "samplename", "name"):
            rename[c] = "sample"
    return df.rename(columns=rename)

def normalize_fraction_columns(df):
    rename = {}
    for c in df.columns:
        cc = _canon(c)
        if cc in ("saturatewtpercent", "saturatedwtpercent", "satwtpercent"):
            rename[c] = "saturate_wt_percent"
        elif cc in ("aromaticwtpercent", "arowtpercent"):
            rename[c] = "aromatic_wt_percent"
        elif cc in ("sample", "samplename", "name"):
            rename[c] = "sample"
    return df.rename(columns=rename)

def load_input_tables(input_excel): 
    xf = pd.ExcelFile(input_excel) 
    sheet_map = {_canon(s): s for s in xf.sheet_names} 
    def find_sheet(aliases):
        for a in aliases:
            if _canon(a) in sheet_map:
                return sheet_map[_canon(a)]
        raise ValueError(f"Cannot find sheet among aliases {aliases}; found {xf.sheet_names}")
    sat_sheet = find_sheet(["Saturated_ms", "saturate_ms", "saturated", "saturate"])
    aro_sheet = find_sheet(["Aromatic_ms", "aromatic_ms", "aromatic", "aromatics"])
    ratio_sheet = find_sheet(["Ratio", "fractions", "fraction"])
    sample_name = Path(input_excel).stem 
    sat = normalize_ms_columns(pd.read_excel(input_excel, sheet_name=sat_sheet)) 
    aro = normalize_ms_columns(pd.read_excel(input_excel, sheet_name=aro_sheet)) 
    ratio = normalize_fraction_columns(pd.read_excel(input_excel, sheet_name=ratio_sheet)) 
    for name, df, cols in [
        ("saturate_ms", sat, ("mz", "intensity")),
        ("aromatic_ms", aro, ("mz", "intensity")),
        ("fractions", ratio, ("saturate_wt_percent", "aromatic_wt_percent")),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Sheet {name} missing columns: {', '.join(missing)} (found: {', '.join(map(str, df.columns))})"
            )
    if "sample" not in sat.columns: sat["sample"] = sample_name 
    if "sample" not in aro.columns: aro["sample"] = sample_name 
    if "sample" not in ratio.columns: ratio.insert(0, "sample", sample_name) 
    return sat[["sample","mz","intensity"]], aro[["sample","mz","intensity"]], ratio[["sample","saturate_wt_percent","aromatic_wt_percent"]] 

def round_half_up(x): 
    return int(np.floor(float(x) + 0.5)) 
 

def _build_spectrum_half_up_sum(df: pd.DataFrame, sample: str) -> dict[int, float]:
    sub = df[df["sample"].astype(str) == str(sample)].copy() 
    spec = {} 
    for _, row in sub.iterrows(): 
        mz = round_half_up(row["mz"]) 
        spec[mz] = spec.get(mz, 0.0) + float(row["intensity"])
    return spec 


def _build_spectrum_drop_frac_gt_0_5_floor_sum(df: pd.DataFrame, sample: str) -> dict[int, float]:
    """
    Keep peaks whose first-decimal fractional part <= 0.5, then bin to floor(mz) and sum.
    Example: 81.2/81.3 kept and summed to 81; 81.9 dropped.
    """
    sub = df[df["sample"].astype(str) == str(sample)].copy()
    spec: dict[int, float] = {}
    for _, row in sub.iterrows():
        value = float(row["mz"])
        base = int(np.floor(value))
        frac = value - base
        if frac > 0.5:
            continue
        spec[base] = spec.get(base, 0.0) + float(row["intensity"])
    return spec


def _build_spectrum_floor_max(df: pd.DataFrame, sample: str) -> dict[int, float]:
    """
    Bin to floor(mz) and take max intensity within each nominal mass.
    Example: 81.2 and 81.9 both floor->81, keep the higher intensity among them.
    """
    sub = df[df["sample"].astype(str) == str(sample)].copy()
    spec: dict[int, float] = {}
    for _, row in sub.iterrows():
        base = int(np.floor(float(row["mz"])))
        intensity = float(row["intensity"])
        prev = spec.get(base)
        spec[base] = intensity if prev is None else max(float(prev), intensity)
    return spec


def _build_spectrum_floor_sum(df: pd.DataFrame, sample: str) -> dict[int, float]:
    """
    Bin to int-truncated nominal mass floor(mz) and sum intensity.
    Example: 81.2 and 81.9 both floor->81, sum their intensities.
    """
    sub = df[df["sample"].astype(str) == str(sample)].copy()
    spec: dict[int, float] = {}
    for _, row in sub.iterrows():
        base = int(np.floor(float(row["mz"])))
        spec[base] = spec.get(base, 0.0) + float(row["intensity"])
    return spec


SPECTRUM_BUILDERS: dict[str, Callable[[pd.DataFrame, str], dict[int, float]]] = {
    "half_up_sum": _build_spectrum_half_up_sum,
    "drop_frac_gt_0_5_floor_sum": _build_spectrum_drop_frac_gt_0_5_floor_sum,
    "floor_sum": _build_spectrum_floor_sum,
    "floor_max": _build_spectrum_floor_max,
}

def p(spec, mz):
    return float(spec.get(int(mz), 0.0))

def sum_series(spec, start1, start2, nmax):
    return sum(p(spec, start1 + 14*n) + p(spec, start2 + 14*n) for n in range(nmax + 1))

def calc_sigmas_aromatic(spec):
    return {"Sigma71": p(spec,71)+p(spec,85), "Sigma67": sum(p(spec,m) for m in [67,68,69,81,82,83,96,97]),
            "Sigma91": sum_series(spec,91,92,6), "Sigma103": sum_series(spec,103,104,6),
            "Sigma115": sum_series(spec,115,116,5), "Sigma128": p(spec,128),
            "Sigma141": sum_series(spec,141,142,7), "Sigma153": sum_series(spec,153,154,7),
            "Sigma151": sum_series(spec,151,152,7), "Sigma177": sum_series(spec,177,178,5)}

def calc_sigmas_saturate(spec):
    return {"Sigma71": p(spec,71)+p(spec,85), "Sigma67": sum(p(spec,m) for m in [67,68,69,81,82,83,96,97]),
            "Sigma123": sum_series(spec,123,124,9), "Sigma149": sum_series(spec,149,150,7),
            "Sigma91": sum_series(spec,91,92,6)}

def calc_alkylbenzene_average_carbon_A(aromatic_spec):
    rows = []
    for n, info in ALKYLBENZENE_TABLE1.items():
        m = info["m"]
        mu = (p(aromatic_spec, m) - p(aromatic_spec, m-1) * info["K1"]) / info["K2"]
        rows.append({"n": n, "m": m, "value_raw": mu, "value_used": max(mu, 0.0)})
    denom = sum(r["value_used"] for r in rows)
    A = sum(r["n"]*r["value_used"] for r in rows)/denom if denom > 0 else np.nan
    return A, pd.DataFrame(rows)

def calc_naphthalene_average_carbon_B(aromatic_spec):
    rows = []
    for n, info in NAPHTHALENE_TABLE1.items():
        m = info["m"]
        x = (p(aromatic_spec, m) - p(aromatic_spec, m-1) * info["L1"]) / info["L2"]
        rows.append({"n": n, "m": m, "value_raw": x, "value_used": max(x, 0.0)})
    denom = sum(r["value_used"] for r in rows)
    B = sum(r["n"]*r["value_used"] for r in rows)/denom if denom > 0 else np.nan
    return B, pd.DataFrame(rows)

def model_from_A_for_para_cyclo(A):
    if np.isnan(A): return 15.5
    a_key = int(round(A)); a_key = min(max(a_key,10),14)
    return nearest(ALKYLBENZENE_TO_PARA_CYCLO[a_key], TABLE3["paraffins"].keys())

def model_from_A_for_alkylbenzene(A):
    return 14 if np.isnan(A) else nearest(A, TABLE3["alkylbenzenes"].keys())

def model_from_B(B, family, default=13):
    if family == "naphthalene": return 10
    return nearest(default if np.isnan(B) else B, TABLE3[family].keys())

def select_models(A, B):
    pc = model_from_A_for_para_cyclo(A)
    alk = model_from_A_for_alkylbenzene(A)
    return {"paraffins": pc, "monocycloparaffins": pc,
            "dicycloparaffins": nearest(pc, TABLE3["dicycloparaffins"].keys()),
            "tricycloparaffins": nearest(pc, TABLE3["tricycloparaffins"].keys()),
            "alkylbenzenes": alk,
            "indans_tetralins": model_from_B(B, "indans_tetralins"),
            "indenes_CnH2n_10": model_from_B(B, "indenes_CnH2n_10"),
            "naphthalene": 10,
            "alkyl_naphthalenes": model_from_B(B, "alkyl_naphthalenes"),
            "acenaphthenes_CnH2n_14": model_from_B(B, "acenaphthenes_CnH2n_14"),
            "acenaphthylenes_CnH2n_16": model_from_B(B, "acenaphthylenes_CnH2n_16"),
            "tricyclic_aromatics": 14}

def build_column(family, model, selected_rows):
    entry = TABLE3[family][model]
    all_coef = dict(zip(SIGMAS_ALL, entry["coef"]))
    return [all_coef[row] for row in selected_rows], entry["mass"]

def build_matrix(models, rows, families):
    cols, sens = [], []
    for fam in families:
        c, s = build_column(fam, models[fam], rows)
        cols.append(c); sens.append(s)
    return np.array(cols, dtype=float).T, np.array(sens, dtype=float)

def solve_matrix(matrix, b, sensitivity):
    raw = np.linalg.solve(matrix, np.array(b, dtype=float))
    return raw, raw / sensitivity

def positive_normalize(values, target_total):
    v = np.array(values, dtype=float); v[v < 0] = 0.0
    return np.zeros_like(v) if v.sum() <= 0 else v/v.sum()*float(target_total)

def calculate_file(input_excel, output_excel): 
    sat_df, aro_df, ratio_df = load_input_tables(input_excel) 
    mode_outputs: dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for mode, builder in SPECTRUM_BUILDERS.items():
        summary_rows, detail_rows, sigma_rows, model_rows = [], [], [], []
        for _, rr in ratio_df.iterrows():
            sample = str(rr["sample"])
            sat_frac, aro_frac = float(rr["saturate_wt_percent"]), float(rr["aromatic_wt_percent"])

            sat_spec = builder(sat_df, sample)
            aro_spec = builder(aro_df, sample)

            A, A_detail = calc_alkylbenzene_average_carbon_A(aro_spec)
            B, B_detail = calc_naphthalene_average_carbon_B(aro_spec)
            models = select_models(A, B)

            sat_rows = ["Sigma71","Sigma67","Sigma123","Sigma149","Sigma91"]
            sat_fams = ["paraffins","monocycloparaffins","dicycloparaffins","tricycloparaffins","alkylbenzenes"]
            aro_rows = ["Sigma71","Sigma67","Sigma91","Sigma103","Sigma115","Sigma128","Sigma141","Sigma153","Sigma151","Sigma177"]
            aro_fams = ["paraffins","monocycloparaffins","alkylbenzenes","indans_tetralins","indenes_CnH2n_10","naphthalene","alkyl_naphthalenes","acenaphthenes_CnH2n_14","acenaphthylenes_CnH2n_16","tricyclic_aromatics"]

            sat_matrix, sat_sens = build_matrix(models, sat_rows, sat_fams)
            aro_matrix, aro_sens = build_matrix(models, aro_rows, aro_fams)

            sat_sigmas, aro_sigmas = calc_sigmas_saturate(sat_spec), calc_sigmas_aromatic(aro_spec)
            sat_raw, sat_mass = solve_matrix(sat_matrix, [sat_sigmas[x] for x in sat_rows], sat_sens)
            aro_raw, aro_mass = solve_matrix(aro_matrix, [aro_sigmas[x] for x in aro_rows], aro_sens)

            for fam, raw, mass, sens in zip(sat_fams, sat_raw, sat_mass, sat_sens):
                detail_rows.append({"mode": mode, "sample":sample,"fraction":"saturate","family":fam,"model":models[fam],"raw_matrix_solution":raw,"mass_sensitivity":sens,"mass_corrected":mass})
            for fam, raw, mass, sens in zip(aro_fams, aro_raw, aro_mass, aro_sens):
                detail_rows.append({"mode": mode, "sample":sample,"fraction":"aromatic","family":fam,"model":models[fam],"raw_matrix_solution":raw,"mass_sensitivity":sens,"mass_corrected":mass})

            for k,v in sat_sigmas.items(): sigma_rows.append({"mode": mode, "sample":sample,"fraction":"saturate","sigma":k,"value":v})
            for k,v in aro_sigmas.items(): sigma_rows.append({"mode": mode, "sample":sample,"fraction":"aromatic","sigma":k,"value":v})

            model_rows.append({"mode": mode, "sample":sample,"A_alkylbenzene_avg_carbon":A,"B_naphthalene_avg_carbon":B, **{f"model_{k}":v for k,v in models.items()}})
            for _, row in A_detail.iterrows(): model_rows.append({"mode": mode, "sample":sample,"carbon_detail":"alkylbenzene",**row.to_dict()})
            for _, row in B_detail.iterrows(): model_rows.append({"mode": mode, "sample":sample,"carbon_detail":"naphthalene",**row.to_dict()})

            # Normalize within each fraction to m1/m2, then move cross-terms between fractions.
            sat_norm = positive_normalize(np.array(sat_mass, dtype=float), sat_frac)  # len=5
            aro_norm = positive_normalize(np.array(aro_mass, dtype=float), aro_frac)  # len=10

            sat_report = np.array(
                [
                    sat_norm[0] + aro_norm[0],
                    sat_norm[1] + aro_norm[1],
                    sat_norm[2],
                    sat_norm[3],
                ],
                dtype=float,
            )
            aro_report = np.array(
                [
                    aro_norm[2] + sat_norm[4],
                    aro_norm[3],
                    aro_norm[4],
                    aro_norm[5],
                    aro_norm[6],
                    aro_norm[7],
                    aro_norm[8],
                    aro_norm[9],
                ],
                dtype=float,
            )

            result = dict(zip(["paraffins","monocycloparaffins","dicycloparaffins","tricycloparaffins"], sat_report))
            result.update(dict(zip(["alkylbenzenes","indans_tetralins","indenes_CnH2n_10","naphthalene","alkyl_naphthalenes","acenaphthenes_CnH2n_14","acenaphthylenes_CnH2n_16","tricyclic_aromatics"], aro_report)))

            rows = [("链烷烃", result.get("paraffins",0)), ("一环烷烃", result.get("monocycloparaffins",0)), ("二环烷烃", result.get("dicycloparaffins",0)), ("三环烷烃", result.get("tricycloparaffins",0))]
            para,mono,di,tri = [x[1] for x in rows]
            alk,indan,indene = result.get("alkylbenzenes",0),result.get("indans_tetralins",0),result.get("indenes_CnH2n_10",0)
            naph,alknaph,acen,aceny,triar = result.get("naphthalene",0),result.get("alkyl_naphthalenes",0),result.get("acenaphthenes_CnH2n_14",0),result.get("acenaphthylenes_CnH2n_16",0),result.get("tricyclic_aromatics",0)

            out = rows + [("总环烷烃",mono+di+tri),("总饱和烃",para+mono+di+tri),("烷基苯",alk),("茚满或四氢萘",indan),("茚类和/或CnH2n-10",indene),("总单环芳烃",alk+indan+indene),("萘",naph),("萘类",alknaph),("苊类和/或CnH2n-14",acen),("芴类和/或CnH2n-16",aceny),("总双环芳烃",naph+alknaph+acen+aceny),("三环芳烃",triar),("总芳烃",alk+indan+indene+naph+alknaph+acen+aceny+triar),("总重量",para+mono+di+tri+alk+indan+indene+naph+alknaph+acen+aceny+triar)]
            for item,val in out:
                summary_rows.append({"mode": mode, "sample":sample,"item":item,"mass_percent":val})

        summary_df = pd.DataFrame(summary_rows)
        detail_df = pd.DataFrame(detail_rows)
        sigma_df = pd.DataFrame(sigma_rows)
        model_df = pd.DataFrame(model_rows)
        if "mass_percent" in summary_df.columns:
            summary_df["mass_percent"] = pd.to_numeric(summary_df["mass_percent"], errors="coerce").round(1)
        mode_outputs[mode] = (summary_df, model_df, detail_df, sigma_df)

    # Write workbook with per-mode sheets + comparison sheet(s).
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for mode, (summary_df, model_df, detail_df, sigma_df) in mode_outputs.items():
            summary_df.to_excel(writer, sheet_name=f"summary_{mode}", index=False)
            model_df.to_excel(writer, sheet_name=f"avg_models_{mode}", index=False)
            detail_df.to_excel(writer, sheet_name=f"matrix_{mode}", index=False)
            sigma_df.to_excel(writer, sheet_name=f"sigmas_{mode}", index=False)

        # Wide comparison for summary_report items.
        all_summary = pd.concat([v[0] for v in mode_outputs.values()], ignore_index=True)
        compare = (
            all_summary.pivot_table(
                index=["sample", "item"],
                columns="mode",
                values="mass_percent",
                aggfunc="first",
            )
            .reset_index()
        )
        compare.to_excel(writer, sheet_name="summary_compare", index=False)

    # Return the "current" baseline mode for compatibility.
    return mode_outputs["half_up_sum"]

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input", default=None, help="Input Excel file (omit to choose via dialog)") 
    parser.add_argument("--output", default="sht0606_official_full_result.xlsx") 
    args = parser.parse_args() 
    input_path = args.input
    if not input_path:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"Missing --input and cannot open file dialog: {exc}") from exc
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        root.lift()
        root.focus_force()
        input_path = filedialog.askopenfilename(
            title="选择样本数据 Excel 文件",
            filetypes=[("Excel files", "*.xlsx;*.xlsm;*.xls"), ("All files", "*.*")],
            parent=root,
        )
        root.destroy()
        if not input_path:
            raise SystemExit("No input file selected.")

    sample_suffix = Path(input_path).stem
    out_path = Path(args.output)
    if out_path.suffix.lower() not in (".xlsx", ".xlsm", ".xls"):
        out_path = out_path.with_suffix(".xlsx")
    suffix_token = f"_{sample_suffix}"
    if not out_path.stem.endswith(suffix_token):
        out_path = out_path.with_name(f"{out_path.stem}{suffix_token}{out_path.suffix}")

    calculate_file(input_path, str(out_path)) 
    print(f"Done: {out_path}") 

if __name__ == "__main__":
    main()
