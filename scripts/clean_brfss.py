"""brfss cleaning pipeline (xpt -> per-year parquet)."""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "cleaned"
DOC_DIR = BASE_DIR / "documentation"

XPT_FILES = {
    2020: "LLCP2020.XPT",
    2021: "LLCP2021.XPT",
    2022: "LLCP2022.XPT",
    2023: "LLCP2023.XPT",
    2024: "LLCP2024.XPT",
}

# brfss missing codes:
#   7, 77, 7777   -> dk     (NaN)
#   9, 99, 9999   -> ref    (NaN)
#   88            -> none   (0, only for the days vars)

DK_REF_1 = {7.0: np.nan, 9.0: np.nan}
VARS_1 = [
    "GENHLTH",
    "CHECKUP1", "EXERANY2", "SMOKE100", "SMOKDAY2",
    "LASTDEN4", "RMVTETH4",
    "CVDINFR4", "CVDCRHD4", "CVDSTRK3",
    "ASTHMA3", "ASTHNOW", "ADDEPEV3", "CHCKDNY2",
    "HAVARTH4", "DIABETE4",
    "PNEUVAC4", "FLUSHOT7", "HIVTST7", "SHINGLE2",
    "HADMAM", "HADHYST2",
    "DEAF", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON",
    "RENTHOM1", "VETERAN3", "PREGNANT", "CAREGIV1",
    "USENOW3", "ECIGNOW3",
    "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
    "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX",
    "SOMALE", "SOFEMALE", "HADSEX",
    "MARIJAN1", "FIREARM5", "LSATISFY",
    "CASTHDX2", "CASTHNO2",
]

DAYS_MAP = {77.0: np.nan, 88.0: 0.0, 99.0: np.nan}
VARS_DAYS = ["PHYSHLTH", "MENTHLTH", "POORHLTH"]

DK_REF_2 = {77.0: np.nan, 99.0: np.nan}
VARS_2 = [
    "INCOME3", "EDUCA", "EMPLOY1", "MARITAL",
    "HOWLONG", "LASTSMK2", "HIVRISK5",
    "MAXDRNKS", "DRNK3GE5",
    "NUMADULT", "CHILDREN", "NUMHHOL4", "NUMPHON4",
    "HPVADVC4", "TETANUS1", "STOPSMK2", "PSATEST1",
]

DK_REF_4 = {7777.0: np.nan, 9999.0: np.nan}
VARS_4 = ["ALCDAY4", "ALCDAY5", "FLSHTMY3", "HIVTSTD3", "DIABAGE4"]

NONE88 = {77.0: np.nan, 88.0: 0.0, 99.0: np.nan}
VARS_NONE88 = ["CHILDREN", "DRNK3GE5", "AVEDRNK3", "AVEDRNK4"]

CALC_BIN = {9.0: np.nan}
CALC_VARS_BIN = [
    "_RFHLTH", "_PHYS14D", "_MENT14D",
    "_HLTHPL2", "_HCVU654",
    "_TOTINDA", "_MICHD",
    "_LTASTH1", "_CASTHM1", "_DRDXAR2",
    "_RFBMI5", "_RFSMOK3",
    "_FLSHOT7", "_PNEUMO3", "_AIDTST4",
]

CALC_MULTI = {9.0: np.nan, 14.0: np.nan}
CALC_VARS_MULTI = [
    "_SMOKER3", "_ASTHMS1", "_BMI5CAT",
    "_EDUCAG", "_AGE_G", "_AGE65YR",
    "_RACE", "_IMPRACE", "_SEX",
]

INCOME_VARS = ["_INCOMG", "_INCOMG1"]  # renamed in 2021

# stuff we don't want in the ML parquet
DROP_ML = [
    "IDATE", "IMONTH", "IDAY", "IYEAR",
    "DISPCODE", "SEQNO", "FMONTH",
    "CTELENM1", "PVTRESD1", "COLGHOUS",
    "STATERE1", "CELPHON1", "LADULT1",
    "RESPSLC1", "LANDSEX3", "SAFETIME",
    "CTELNUM1", "CELLFON5", "CADULT1",
    "CELLSEX3", "CELLSEX2", "CELLSEX1", "CELLSEX",
    "PVTRESD3", "CCLGHOUS", "CSTATE1",
    "LANDLINE", "HHADULT",
    "NUMADULT", "NUMHHOL4", "NUMPHON4",
    "CPDEMO1C", "CPDEMO1B", "CPDEMO1A", "CPDEMO1",
    "QSTVER", "QSTLANG", "MSCODE",
    "_PSU", "_STSTR", "_STRWT", "_RAWRAKE",
    "_WT2RAKE", "_CLLCPWT", "_LLCPWT2",
    "_DUALUSE", "_DUALCOR",
    "_METSTAT", "_URBSTAT",
]


def setup_logging(year):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"brfss_{year}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    fh = logging.FileHandler(DOC_DIR / f"cleaning_log_{year}.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


def load_xpt(year, logger):
    fpath = DATA_DIR / XPT_FILES[year]
    logger.info(f"Loading {fpath} ...")
    t0 = time.time()
    df = pd.read_sas(str(fpath), format="xport")
    logger.info(f"  {df.shape[0]:,} x {df.shape[1]} in {time.time()-t0:.1f}s "
                f"({df.memory_usage(deep=True).sum()/1e9:.2f} GB)")
    df.insert(0, "SURVEY_YEAR", year)
    return df


def recode_missing(df, year, logger):
    logger.info("Recoding missing codes ...")
    counts = {}

    def apply_(var_list, mapping, label):
        for var in var_list:
            if var in df.columns:
                n = df[var].isin(mapping.keys()).sum()
                if n > 0:
                    df[var] = df[var].replace(mapping)
                    counts[var] = n
                    logger.debug(f"  {var}: {n:,} ({label})")

    apply_(VARS_1, DK_REF_1, "7/9 -> NaN")
    apply_(VARS_DAYS, DAYS_MAP, "77/99 -> NaN, 88 -> 0")
    apply_(VARS_2, DK_REF_2, "77/99 -> NaN")
    apply_(VARS_NONE88, NONE88, "77/99 -> NaN, 88 -> 0")

    for var in ("HEIGHT3", "WEIGHT2"):
        if var in df.columns:
            n = df[var].isin([7777.0, 9999.0]).sum()
            if n > 0:
                df[var] = df[var].replace({7777.0: np.nan, 9999.0: np.nan})
                counts[var] = n

    apply_(VARS_4, DK_REF_4, "7777/9999 -> NaN")
    apply_(CALC_VARS_BIN, CALC_BIN, "9 -> NaN")
    apply_(CALC_VARS_MULTI, CALC_MULTI, "9/14 -> NaN")

    for v in INCOME_VARS:
        if v in df.columns:
            apply_([v], CALC_MULTI, "9 -> NaN")

    logger.info(f"  recoded {sum(counts.values()):,} across {len(counts)} vars")
    return df


def validate_design(df, year, logger):
    logger.info("Checking survey design vars ...")
    stats = {}
    for var in ("_LLCPWT", "_STSTR", "_PSU"):
        if var not in df.columns:
            logger.error(f"  missing: {var}")
            continue
        nm = df[var].isna().sum()
        nz = (df[var] == 0).sum()
        nn = (df[var] < 0).sum()
        if nm:
            logger.warning(f"  {var}: {nm:,} missing")
        if var == "_LLCPWT":
            if nz: logger.warning(f"  {var}: {nz:,} zero-weight rows")
            if nn: logger.warning(f"  {var}: {nn:,} negative-weight rows")
            stats = {
                "min": float(df[var].min()),
                "max": float(df[var].max()),
                "mean": float(df[var].mean()),
                "median": float(df[var].median()),
                "std": float(df[var].std()),
                "sum": float(df[var].sum()),
                "skewness": float(df[var].skew()),
                "n_zero": int(nz),
                "n_negative": int(nn),
            }
            logger.info(f"  _LLCPWT: mean={stats['mean']:.1f} median={stats['median']:.1f} "
                        f"sum={stats['sum']:,.0f}")
    logger.info(f"  strata: {df['_STSTR'].nunique():,}, psus: {df['_PSU'].nunique():,}")
    return stats


def clean_calc(df, year, logger):
    # _BMI5 is BMI*100 in the raw file
    if "_BMI5" in df.columns:
        df["_BMI5_SCALED"] = df["_BMI5"] / 100.0
        lo = (df["_BMI5_SCALED"] < 12).sum()
        hi = (df["_BMI5_SCALED"] > 80).sum()
        if lo or hi:
            logger.warning(f"  _BMI5_SCALED extremes: {lo} <12, {hi} >80")

    if "_AGE80" in df.columns:
        logger.info(f"  _AGE80: {(df['_AGE80']==80).sum():,} top-coded at 80")

    return df


def derive_features(df, year, logger):
    n = 0

    def to_bin(src, target, pos=1.0):
        nonlocal n
        if src in df.columns:
            df[target] = np.where(df[src].isna(), np.nan,
                                  np.where(df[src] == pos, 1.0, 0.0))
            n += 1

    if "GENHLTH" in df.columns:
        df["GOOD_HEALTH"] = np.where(df["GENHLTH"].isna(), np.nan,
                                     np.where(df["GENHLTH"].isin([1.0, 2.0, 3.0]), 1.0, 0.0))
        n += 1

    to_bin("DIABETE4", "HAS_DIABETES")
    to_bin("SMOKE100", "EVER_SMOKED")
    to_bin("EXERANY2", "HAS_EXERCISE")
    to_bin("ADDEPEV3", "HAS_DEPRESSION")
    to_bin("FLUSHOT7", "GOT_FLUSHOT")
    to_bin("CHECKUP1", "RECENT_CHECKUP")
    to_bin("CVDINFR4", "HAS_HEART_ATTACK")
    to_bin("CVDSTRK3", "HAS_STROKE")

    # whichever insurance var exists this year
    for hvar in ("PRIMINS2", "HLTHPLN1"):
        if hvar in df.columns:
            df["HAS_HEALTH_PLAN"] = np.where(df[hvar].isna(), np.nan,
                                             np.where(df[hvar] == 1.0, 1.0, 0.0))
            n += 1
            break

    if "MEDCOST1" in df.columns:
        df["MEDCOST1"] = df["MEDCOST1"].replace({7.0: np.nan, 9.0: np.nan})
        df["COST_BARRIER"] = np.where(df["MEDCOST1"].isna(), np.nan,
                                      np.where(df["MEDCOST1"] == 1.0, 1.0, 0.0))
        n += 1

    for pdvar in ("PERSDOC3", "PERSDOC2"):
        if pdvar in df.columns:
            df[pdvar] = df[pdvar].replace({7.0: np.nan, 9.0: np.nan})
            df["HAS_PERSONAL_DOCTOR"] = np.where(df[pdvar].isna(), np.nan,
                                                 np.where(df[pdvar] == 1.0, 1.0, 0.0))
            n += 1
            break

    if "_STATE" in df.columns:
        df["STATE_FIPS"] = df["_STATE"].astype("Int64")
        n += 1

    logger.info(f"  derived {n} features")
    return df


def save_outputs(df, year, logger):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    info = {}

    eda = CLEAN_DIR / f"brfss_{year}_eda.parquet"
    # parquet hates mixed object dtypes
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    df.to_parquet(eda, index=False, engine="pyarrow")
    info.update(eda_path=str(eda),
                eda_size_mb=round(os.path.getsize(eda)/1e6, 1),
                eda_rows=df.shape[0], eda_cols=df.shape[1])
    logger.info(f"  eda: {info['eda_size_mb']} MB ({df.shape[0]:,} x {df.shape[1]})")

    ml = df.copy()
    if "_LLCPWT" in ml.columns:
        ml["sample_weight"] = ml["_LLCPWT"] / ml["_LLCPWT"].mean()
    drop = [c for c in DROP_ML if c in ml.columns]
    ml = ml.drop(columns=drop)
    if "_LLCPWT" in ml.columns:
        ml = ml.drop(columns=["_LLCPWT"])

    ml_path = CLEAN_DIR / f"brfss_{year}_ml.parquet"
    ml.to_parquet(ml_path, index=False, engine="pyarrow")
    info.update(ml_path=str(ml_path),
                ml_size_mb=round(os.path.getsize(ml_path)/1e6, 1),
                ml_rows=ml.shape[0], ml_cols=ml.shape[1])
    logger.info(f"  ml: {info['ml_size_mb']} MB ({ml.shape[0]:,} x {ml.shape[1]})")
    return info


def quality_report(df, year, weight_stats, info, logger):
    path = DOC_DIR / f"quality_report_{year}.md"
    L = [
        f"# brfss {year} - quality report",
        f"\n_generated: {datetime.now():%Y-%m-%d %H:%M:%S}_\n",
        "## overview\n",
        "| metric | value |",
        "|--------|-------|",
        f"| rows | {df.shape[0]:,} |",
        f"| cols | {df.shape[1]} |",
        f"| year | {year} |",
        f"| source | {XPT_FILES[year].strip()} |",
        f"| eda mb | {info.get('eda_size_mb','-')} |",
        f"| ml mb | {info.get('ml_size_mb','-')} |",
        f"| ml cols | {info.get('ml_cols','-')} |",
        "\n## weight (_LLCPWT)\n",
    ]
    if weight_stats:
        L += ["| stat | value |", "|------|-------|"]
        for k, v in weight_stats.items():
            L.append(f"| {k} | {v:,.2f} |" if isinstance(v, float) else f"| {k} | {v:,} |")

    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    L.append("\n## missing values\n")
    if len(miss):
        L.append(f"{len(miss)} columns have missing values.\n")
        L += ["| var | n | % |", "|-----|---|---|"]
        for var, n in miss.head(50).items():
            L.append(f"| {var} | {n:,} | {n/len(df)*100:.1f}% |")
        if len(miss) > 50:
            L.append(f"\n... + {len(miss)-50} more.\n")
    else:
        L.append("none.\n")

    L.append("\n## key distributions\n")
    keys = [
        ("GENHLTH", "general health",
         {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}),
        ("_SEX", "sex", {1: "Male", 2: "Female"}),
        ("_IMPRACE", "race/ethnicity (imputed)",
         {1: "White NH", 2: "Black NH", 3: "Asian NH", 4: "AI/AN NH", 5: "Hispanic", 6: "Other NH"}),
        ("_AGE_G", "age group",
         {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+"}),
        ("_EDUCAG", "education",
         {1: "<HS", 2: "HS", 3: "Some college", 4: "College+"}),
        ("_BMI5CAT", "bmi cat",
         {1: "Under", 2: "Normal", 3: "Over", 4: "Obese"}),
        ("DIABETE4", "diabetes",
         {1: "Yes", 2: "Yes-Pregnancy", 3: "No", 4: "Pre-diabetes"}),
        ("EXERANY2", "exercise (30d)", {1: "Yes", 2: "No"}),
    ]
    for var, label, vlabels in keys:
        if var in df.columns:
            L.append(f"### {label} ({var})\n")
            vc = df[var].value_counts(dropna=False).sort_index()
            L += ["| value | label | n | % |", "|-------|-------|---|---|"]
            for v, n in vc.items():
                lab = vlabels.get(v, "Missing" if pd.isna(v) else str(v))
                vstr = "NaN" if pd.isna(v) else f"{v:.0f}"
                L.append(f"| {vstr} | {lab} | {n:,} | {n/len(df)*100:.1f}% |")
            L.append("")

    derived = ["GOOD_HEALTH", "HAS_DIABETES", "EVER_SMOKED", "HAS_EXERCISE",
               "HAS_DEPRESSION", "GOT_FLUSHOT", "RECENT_CHECKUP",
               "HAS_HEART_ATTACK", "HAS_STROKE", "HAS_HEALTH_PLAN",
               "COST_BARRIER", "HAS_PERSONAL_DOCTOR"]
    L.append("\n## derived flags\n")
    L += ["| flag | 1 | 0 | NaN |", "|------|---|---|-----|"]
    for f in derived:
        if f in df.columns:
            p = (df[f] == 1.0).sum()
            n = (df[f] == 0.0).sum()
            m = df[f].isna().sum()
            L.append(f"| {f} | {p:,} | {n:,} | {m:,} |")

    path.write_text("\n".join(L))
    return str(path)


def process_year(year):
    log = setup_logging(year)
    log.info("=" * 60)
    log.info(f"brfss {year}")
    log.info("=" * 60)
    t0 = time.time()
    s = {"year": year}

    df = load_xpt(year, log)
    s["raw_rows"] = df.shape[0]
    s["raw_cols"] = df.shape[1] - 1  # don't count SURVEY_YEAR

    df = recode_missing(df, year, log)
    s["weight_stats"] = validate_design(df, year, log)
    df = clean_calc(df, year, log)
    df = derive_features(df, year, log)
    df = df.copy()  # defragment

    info = save_outputs(df, year, log)
    s.update(info)
    s["quality_report"] = quality_report(df, year, s["weight_stats"], info, log)

    s["processing_time_seconds"] = round(time.time() - t0, 1)
    log.info(f"\nbrfss {year} done in {s['processing_time_seconds']:.1f}s\n")
    del df
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, choices=[2020, 2021, 2022, 2023, 2024])
    p.add_argument("--all", action="store_true", default=True)
    args = p.parse_args()

    years = [args.year] if args.year else [2020, 2021, 2022, 2023, 2024]
    print(f"\nrunning years: {years}\n")

    results = []
    for y in years:
        try:
            results.append(process_year(y))
        except Exception as e:
            print(f"ERROR {y}: {e}")
            import traceback; traceback.print_exc()

    summary = DOC_DIR / "pipeline_summary.json"
    summary.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsummary -> {summary}\n")

    print(f"{'year':<6} {'rows':>10} {'cols':>5} {'eda':>7} {'ml':>7} {'time':>7}")
    for s in results:
        print(f"{s['year']:<6} {s['raw_rows']:>10,} {s['raw_cols']:>5} "
              f"{s.get('eda_size_mb',0):>7.1f} {s.get('ml_size_mb',0):>7.1f} "
              f"{s.get('processing_time_seconds',0):>6.1f}s")


if __name__ == "__main__":
    main()
