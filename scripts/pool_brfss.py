"""pool the per-year cleaned parquets into one (2020-2024)."""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "cleaned"
DOC_DIR = BASE_DIR / "documentation"

YEARS = [2020, 2021, 2022, 2023, 2024]
N = len(YEARS)

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


def setup_logging():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("brfss_pool")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%H:%M:%S"))
    log.addHandler(ch)

    fh = logging.FileHandler(DOC_DIR / "pooling_log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)
    return log


def main():
    log = setup_logging()
    log.info("brfss pool 2020-2024")
    t0 = time.time()

    cols_per_year = {}
    rows_per_year = {}
    for y in YEARS:
        p = CLEAN_DIR / f"brfss_{y}_eda.parquet"
        if not p.exists():
            log.error(f"missing: {p} - run clean_brfss.py --year {y} first")
            sys.exit(1)
        rows_per_year[y] = len(pd.read_parquet(p, columns=["SURVEY_YEAR"]))
        cols_per_year[y] = set(pd.read_parquet(p, columns=None).columns.tolist())
        log.info(f"  {y}: {len(cols_per_year[y])} cols, {rows_per_year[y]:,} rows")

    common = cols_per_year[YEARS[0]]
    for y in YEARS[1:]:
        common &= cols_per_year[y]
    common.add("SURVEY_YEAR")
    common = sorted(common)
    log.info(f"common cols: {len(common)}, total rows: {sum(rows_per_year.values()):,}")

    frames = []
    for y in YEARS:
        log.info(f"loading {y}")
        frames.append(pd.read_parquet(CLEAN_DIR / f"brfss_{y}_eda.parquet", columns=common))
    pooled = pd.concat(frames, ignore_index=True)
    del frames
    log.info(f"pooled: {pooled.shape[0]:,} x {pooled.shape[1]}")

    # cdc says divide by N when pooling years
    if "_LLCPWT" in pooled.columns:
        orig_sum = pooled["_LLCPWT"].sum()
        pooled["_LLCPWT_POOLED"] = pooled["_LLCPWT"] / N
        new_sum = pooled["_LLCPWT_POOLED"].sum()
        log.info(f"weights: orig sum {orig_sum:,.0f} -> pooled sum {new_sum:,.0f} "
                 f"(ratio {new_sum/orig_sum:.4f}, expected {1/N:.4f})")

    yc = pooled["SURVEY_YEAR"].value_counts().sort_index()
    for y, n in yc.items():
        log.info(f"  {int(y)}: {n:,}")

    eda_path = CLEAN_DIR / "brfss_2020_2024_pooled_eda.parquet"
    pooled.to_parquet(eda_path, index=False, engine="pyarrow")
    eda_mb = os.path.getsize(eda_path) / 1e6
    log.info(f"eda: {eda_path} ({eda_mb:.1f} MB)")

    ml = pooled.copy()
    if "_LLCPWT_POOLED" in ml.columns:
        ml["sample_weight"] = ml["_LLCPWT_POOLED"] / ml["_LLCPWT_POOLED"].mean()
    drop = [c for c in DROP_ML if c in ml.columns]
    for w in ("_LLCPWT", "_LLCPWT_POOLED"):
        if w in ml.columns and w not in drop:
            drop.append(w)
    ml = ml.drop(columns=drop)
    ml_path = CLEAN_DIR / "brfss_2020_2024_pooled_ml.parquet"
    ml.to_parquet(ml_path, index=False, engine="pyarrow")
    ml_mb = os.path.getsize(ml_path) / 1e6
    log.info(f"ml:  {ml_path} ({ml_mb:.1f} MB, {ml.shape[1]} cols)")
    del ml

    L = [
        "# brfss 2020-2024 pooled - quality report",
        f"\n_generated: {datetime.now():%Y-%m-%d %H:%M:%S}_\n",
        "## overview\n",
        "| metric | value |", "|--------|-------|",
        f"| years | {', '.join(map(str, YEARS))} |",
        f"| rows | {pooled.shape[0]:,} |",
        f"| common cols | {len(common)} |",
        f"| eda mb | {eda_mb:.1f} |",
        f"| ml mb | {ml_mb:.1f} |",
        "\n## rows per year\n",
        "| year | n | % |", "|------|---|---|",
    ]
    for y, n in yc.items():
        L.append(f"| {int(y)} | {n:,} | {n/pooled.shape[0]*100:.1f}% |")

    L += [
        "\n## weight pooling\n",
        f"- `_LLCPWT_POOLED = _LLCPWT / {N}` (cdc guidance)",
        f"- orig sum: {orig_sum:,.0f}",
        f"- pooled sum: {new_sum:,.0f}",
        "\n## missing values (top 30)\n",
    ]
    miss = pooled.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        L += ["| var | n | % |", "|-----|---|---|"]
        for var, n in miss.head(30).items():
            L.append(f"| {var} | {n:,} | {n/len(pooled)*100:.1f}% |")

    L.append("\n## common columns\n```")
    L.extend(common)
    L.append("```")
    (DOC_DIR / "quality_report_pooled.md").write_text("\n".join(L))

    log.info(f"\ndone in {time.time()-t0:.1f}s")
    del pooled


if __name__ == "__main__":
    main()
