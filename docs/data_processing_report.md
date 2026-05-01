# BRFSS 2020-2024: data processing report

Documentation of how the raw CDC files become the cleaned EDA / ML parquets used in the notebooks. The pipeline lives in `scripts/clean_brfss.py` (per-year cleaning) and `scripts/pool_brfss.py` (cross-year pooling).

---

## 1. Data source

The Behavioral Risk Factor Surveillance System (BRFSS) is the CDC's annual telephone health survey. It runs in all 50 states, DC, and a few territories, and it's the largest health survey of its kind in the world (~440k respondents per year).

Survey design highlights:

- Mode: telephone (landline + cell)
- Population: non-institutionalized U.S. adults (age 18+)
- Sampling: random digit dialing, stratified by sub-state geography and density, clustered by household
- Weighting: iterative proportional fitting (raking) to match population margins on age, sex, race/ethnicity, education, marital status, home ownership, and phone usage
- Annual sample size: ~400k-460k

Without weights, the raw sample is biased toward older adults, women, white respondents, and landline users. The `_LLCPWT` weight is what makes any aggregate generalize to the U.S. adult population.

## 2. Output files

Each year produces an EDA and an ML parquet. The pooled files combine 2020-2024.

| File | Description | Size | Rows | Cols |
|------|-------------|-----:|-----:|-----:|
| `cleaned/brfss_2020_eda.parquet` | 2020 EDA (full + design vars) | 29.7 MB | 401,958 | 293 |
| `cleaned/brfss_2020_ml.parquet`  | 2020 ML (admin dropped, sample_weight added) | 25.0 MB | 401,958 | 255 |
| `cleaned/brfss_2021_eda.parquet` | 2021 EDA | 36.4 MB | 438,693 | 317 |
| `cleaned/brfss_2021_ml.parquet`  | 2021 ML | 31.0 MB | 438,693 | 278 |
| `cleaned/brfss_2022_eda.parquet` | 2022 EDA | 36.2 MB | 445,132 | 342 |
| `cleaned/brfss_2022_ml.parquet`  | 2022 ML | 30.8 MB | 445,132 | 301 |
| `cleaned/brfss_2023_eda.parquet` | 2023 EDA | 40.2 MB | 433,323 | 364 |
| `cleaned/brfss_2023_ml.parquet`  | 2023 ML | 35.1 MB | 433,323 | 322 |
| `cleaned/brfss_2024_eda.parquet` | 2024 EDA | 36.1 MB | 457,670 | 316 |
| `cleaned/brfss_2024_ml.parquet`  | 2024 ML | 30.6 MB | 457,670 | 273 |
| `cleaned/brfss_2020_2024_pooled_eda.parquet` | 5-year pooled EDA | 125.7 MB | 2,176,776 | 166 |
| `cleaned/brfss_2020_2024_pooled_ml.parquet`  | 5-year pooled ML | 83.3 MB | 2,176,776 | 129 |

The cleaned parquets are not in the repo (too large for GitHub). They're posted on Kaggle (see `data/README.md`).

## 3. Variable dictionary

### Survey design variables

| Variable | What it is | Use |
|----------|-----------|-----|
| `_LLCPWT` | Final raked weight (combined landline + cell) | Use in all weighted analyses |
| `_STSTR` | Stratification id | Required for variance estimation |
| `_PSU` | Primary Sampling Unit (cluster id) | Required for variance estimation |
| `_STRWT` | Stratum weight (#records / #selected) | Intermediate component |
| `_RAWRAKE` | Raw raking factor | Intermediate component |
| `_WT2RAKE` | Design weight = `_STRWT * _RAWRAKE` | Intermediate component |

### Demographics

| Variable | Label | Valid values | Notes |
|----------|-------|-------------|-------|
| `_STATE` | State FIPS | 1-78 | All 50 states + DC + territories |
| `_SEX` | Sex | 1=Male, 2=Female | CDC-calculated |
| `_IMPRACE` | Imputed race/ethnicity | 1=White NH, 2=Black NH, 3=Asian NH, 4=AI/AN NH, 5=Hispanic, 6=Other NH | Imputed when missing |
| `_AGE_G` | Age group (6-level) | 1=18-24 ... 6=65+ | |
| `_AGE80` | Age (top-coded at 80) | 18-80 | |
| `_EDUCAG` | Education (4-level) | 1=<HS, 2=HS, 3=Some college, 4=College+ | Prefer over raw EDUCA |
| `_INCOMG1` | Income group | Varies by year | Prefer over raw INCOME3 |
| `EMPLOY1`, `MARITAL` | Employment, marital status | various | |

### Health status

| Variable | Label | Valid values |
|----------|-------|-------------|
| `GENHLTH` | General health | 1=Excellent ... 5=Poor |
| `PHYSHLTH`, `MENTHLTH`, `POORHLTH` | Bad-health days in past 30 | 0-30 (88->0, 77/99->NaN) |
| `_BMI5` | BMI x 100 | continuous |
| `_BMI5_SCALED` | BMI (`_BMI5 / 100`) | created by pipeline |
| `_BMI5CAT` | BMI category | 1=Under, 2=Normal, 3=Over, 4=Obese |

### Chronic conditions, preventive health

`DIABETE4`, `CVDINFR4`, `CVDCRHD4`, `CVDSTRK3`, `ASTHMA3`, `ADDEPEV3`, `CHCKDNY2`, `HAVARTH4`, `CHECKUP1`, `EXERANY2`, `FLUSHOT7`, `SMOKE100`, `_SMOKER3`, `HADMAM`, `SHINGLE2` - see the codebooks in `docs/` for full definitions.

### Pipeline-derived binary flags

Each is 1=Yes, 0=No, NaN=Missing.

| Flag | Source | Definition |
|------|--------|------------|
| `GOOD_HEALTH` | `GENHLTH` | 1 if Excellent/VG/Good |
| `HAS_DIABETES` | `DIABETE4` | 1 if diagnosed diabetic |
| `EVER_SMOKED` | `SMOKE100` | 1 if smoked >=100 cigs |
| `HAS_EXERCISE` | `EXERANY2` | 1 if exercised in past 30d |
| `HAS_DEPRESSION` | `ADDEPEV3` | 1 if ever diagnosed |
| `GOT_FLUSHOT` | `FLUSHOT7` | 1 if flu shot in past 12 mo |
| `RECENT_CHECKUP` | `CHECKUP1` | 1 if checkup within past year |
| `HAS_HEART_ATTACK` | `CVDINFR4` | 1 if ever had MI |
| `HAS_STROKE` | `CVDSTRK3` | 1 if ever had stroke |
| `HAS_HEALTH_PLAN` | `PRIMINS2`/`HLTHPLN1` | 1 if has any insurance |
| `COST_BARRIER` | `MEDCOST1` | 1 if skipped doctor due to cost |
| `HAS_PERSONAL_DOCTOR` | `PERSDOC3`/`PERSDOC2` | 1 if has personal doctor |
| `STATE_FIPS` | `_STATE` | Integer FIPS |

## 4. Missing-value treatment

CDC sentinel codes are not "missing" in the usual sense - they're "Don't know", "Refused", "Not asked", or substantive zeros. The pipeline recodes them like this:

| Code | Meaning | Action | Variables |
|------|---------|--------|-----------|
| 7 | DK (1-digit) | -> NaN | GENHLTH, EXERANY2, SMOKE100, DIABETE4, all chronic Yes/No |
| 9 | Refused (1-digit) | -> NaN | same as above |
| 77 | DK (2-digit) | -> NaN | INCOME3, EDUCA, MAXDRNKS, PHYSHLTH, MENTHLTH |
| 99 | Refused (2-digit) | -> NaN | same as above |
| 88 | None / zero (days vars) | -> 0 | PHYSHLTH, MENTHLTH, POORHLTH, CHILDREN, DRNK3GE5 |
| 7777 | DK (4-digit) | -> NaN | HEIGHT3, WEIGHT2, ALCDAY4 |
| 9999 | Refused (4-digit) | -> NaN | HEIGHT3, WEIGHT2, ALCDAY4 |
| blank/NaN | Not asked / module skipped | kept as NaN | module-specific |

Calculated (`_`-prefixed) variables use `9` as DK/Ref, and a few multi-level ones also use `14` as DK.

Important: for the days-in-past-30 variables, `88` is "None" - that's a substantive zero, not missing. The pipeline recodes `88 -> 0` for those.

### Recoding totals by year

| Year | Sentinel values recoded | Variables affected |
|------|------------------------:|-------------------:|
| 2020 | 1,552,045 | 75 |
| 2021 | 1,891,949 | 82 |
| 2022 | ~1,900,000 | ~84 |
| 2023 | ~1,850,000 | ~83 |
| 2024 | ~1,950,000 | ~85 |

## 5. Weighting

CDC builds the BRFSS weights in stages:

1. Base weight `_STRWT` = records-in-stratum / records-selected.
2. Raw raking factor `_RAWRAKE` = adults-in-household (cap 5) / imputed-phones (cap 3).
3. Design weight `_WT2RAKE = _STRWT * _RAWRAKE`.
4. Raking adjusts the design weight to match population margins on age-by-sex, race/ethnicity, education, marital status, home ownership, phone use, and within-state geography.
5. Final weight `_LLCPWT` is the raked weight after truncation.

### Using the weights

For descriptive estimates with proper standard errors, use a survey package:

```python
import samplics
design = samplics.SurveyDesign(strata="_STSTR", psu="_PSU",
                               weight="_LLCPWT", data=df)
```

```r
library(survey)
options(survey.lonely.psu = "adjust")
d <- svydesign(id=~`_PSU`, strata=~`_STSTR`, weights=~`_LLCPWT`,
               data=df, nest=TRUE)
svymean(~variable, design=d, na.rm=TRUE)
```

For machine learning, the pipeline writes a normalized `sample_weight` column (mean = 1) into the ML parquets. Use it with sklearn:

```python
clf.fit(X_train, y_train, sample_weight=df_train["sample_weight"])
```

Note: passing `sample_weight` adjusts for unequal selection probabilities, but does *not* correct standard errors for clustering. For inferential claims, fall back to a survey-aware estimator.

## 6. Per-year weight summary

| Year | Records | Cols (raw) | `_LLCPWT` sum (~ US adult pop) | Mean weight | Median weight | Max weight |
|------|--------:|-----------:|-------------------------------:|------------:|--------------:|-----------:|
| 2020 | 401,958 | 279 | 260,408,470 | 647.85 | 282.29 | 83,193.32 |
| 2021 | 438,693 | 303 | 246,041,640 | 560.93 | 244.47 | ~80,000 |
| 2022 | 445,132 | 328 | 264,789,594 | 594.87 | 253.14 | ~85,000 |
| 2023 | 433,323 | 350 | 254,139,829 | 586.55 | 247.54 | ~82,000 |
| 2024 | 457,670 | 301 | 263,783,390 | 576.34 | 244.95 | ~84,000 |

State coverage: all five years cover 53 FIPS codes (50 states + DC + Guam + Puerto Rico).

The column count drifts (279 -> 350 -> 301) because BRFSS rotates modules in and out each year (COVID-19, e-cigarettes, ACE expansions, firearm safety, etc.).

## 7. Multi-year pooling

When pooling years, the CDC says to divide the weight by the number of years:

```
_LLCPWT_POOLED = _LLCPWT / 5
```

If we just concatenated 5 years without adjusting, the weight sum would suggest ~1.3B adults - 5x reality. Dividing by 5 keeps the sum at one year's worth (~258M).

Only the 165 columns common to all 5 years are kept in the pooled file. That avoids structural NaN columns from year-specific modules.

| Pooled metric | Value |
|---------------|------:|
| Total records | 2,176,776 |
| Common columns (EDA / ML) | 165 / 129 |
| Pooled weight sum | 257,832,585 |
| Original sum | 1,289,162,923 |
| Ratio | 0.2000 (= 1/5) |

| Year | Records | % of total |
|------|--------:|-----------:|
| 2020 | 401,958 | 18.5% |
| 2021 | 438,693 | 20.2% |
| 2022 | 445,132 | 20.4% |
| 2023 | 433,323 | 19.9% |
| 2024 | 457,670 | 21.0% |

## 8. Year-over-year quirks

### Variable renames

Several variables changed names mid-period. The pipeline checks for both names and uses whichever exists.

| Concept | 2020 | 2021 | 2022 | 2023 | 2024 |
|---------|------|------|------|------|------|
| Health coverage | `HLTHPLN1` | `HLTHPLN1` | `PRIMINS1` | `PRIMINS1` | `PRIMINS2` |
| Personal doctor | `PERSDOC2` | `PERSDOC3` | `PERSDOC3` | `PERSDOC3` | `PERSDOC3` |
| Income (raw) | `INCOME2` | `INCOME3` | `INCOME3` | `INCOME3` | `INCOME3` |
| Alcohol days | `ALCDAY5` | `ALCDAY5` | `ALCDAY4` | `ALCDAY4` | `ALCDAY4` |
| Avg drinks | `AVEDRNK3` | `AVEDRNK3` | `AVEDRNK3` | `AVEDRNK3` | `AVEDRNK4` |
| Income group (calc) | `_INCOMG` | `_INCOMG1` | `_INCOMG1` | `_INCOMG1` | `_INCOMG1` |
| COPD | `CHCCOPD2` | `CHCCOPD2` | `CHCCOPD3` | `CHCCOPD3` | `CHCCOPD3` |
| Cell phone sex | `CELLSEX` | `CELLSEX` | `CELLSEX1` | `CELLSEX2` | `CELLSEX3` |
| Skin cancer | `CHCSCNCR` | `CHCSCNCR` | `CHCSCNC1` | `CHCSCNC1` | `CHCSCNC1` |
| Other cancer | `CHCOCNCR` | `CHCOCNCR` | `CHCOCNC1` | `CHCOCNC1` | `CHCOCNC1` |

### BMI extremes

A small number of records (54 in 2020) have `_BMI5_SCALED > 80`, which isn't physiologically plausible. The pipeline flags but doesn't drop them - the analyst can decide whether to cap or exclude.

### Income coding break in 2021

INCOME2/INCOME3 expanded from 8 to 11 categories in 2021. Don't compare raw income across the break - use `_INCOMG`/`_INCOMG1`.

### COVID-19 in 2020

The 2020 BRFSS was collected during the pandemic, which probably affected response rates and some health behaviors. Worth noting if any 2020-specific result looks odd.

### Skewed weights

All years are heavily right-skewed (skewness ~10-15). That's expected - a small number of underrepresented respondents pull weights up to ~80,000.

## 9. Output file contents

### EDA files (`brfss_{year}_eda.parquet`)

Use for descriptive stats, prevalence, cross-tabs, trend analysis. Contains:
- All original variables (after recoding)
- All `_`-prefix calculated variables
- Survey design vars (`_LLCPWT`, `_STSTR`, `_PSU`)
- Pipeline-derived flags
- `SURVEY_YEAR`, `_BMI5_SCALED`, `STATE_FIPS`

### ML files (`brfss_{year}_ml.parquet`)

Use for modeling. Contains everything in the EDA file *except*:
- Survey admin columns (`IDATE`, `SEQNO`, `DISPCODE`, ...)
- Phone/household screening variables
- Raw design columns (`_PSU`, `_STSTR`, `_STRWT`, ...)
- The original `_LLCPWT` (replaced by normalized `sample_weight`)

### Pooled files

- `brfss_2020_2024_pooled_eda.parquet`: 165 common columns, includes `_LLCPWT_POOLED`.
- `brfss_2020_2024_pooled_ml.parquet`: 129 columns, normalized `sample_weight`.

### Loading in pandas

```python
import pandas as pd

df_eda = pd.read_parquet("cleaned/brfss_2024_eda.parquet")
df_ml  = pd.read_parquet("cleaned/brfss_2024_ml.parquet")
df_pooled = pd.read_parquet("cleaned/brfss_2020_2024_pooled_ml.parquet")

# loading specific columns is way faster
df = pd.read_parquet("cleaned/brfss_2024_eda.parquet",
                     columns=["GENHLTH", "_IMPRACE", "_AGE_G", "_LLCPWT"])
```

## 10. Notes for analysis

For descriptive work:
- Always weight with `_LLCPWT` (or `_LLCPWT_POOLED` for pooled).
- Use all three design variables for proper SEs.
- In R, set `survey.lonely.psu = "adjust"` (some strata have a single PSU).
- Watch unweighted n - if a subgroup has n < 50, the estimate is unstable.
- Prefer `_`-calculated variables over raw ones.

For ML:
- Use the ML parquets, not EDA.
- Pass `sample_weight=df["sample_weight"]` to `fit()`.
- Be careful with SMOTE / oversampling - those conflict with survey weights.
- Standard k-fold CV is fine for prediction, but for survey inference you'd want folds stratified by `_STSTR`/`SURVEY_YEAR`.
- The pipeline-derived flags are ready-to-use binary targets.

Things to avoid:
- Don't ignore weights for population-level claims.
- Don't treat `88 = None` as missing for the days variables.
- Don't compare raw `INCOME2`/`INCOME3` across the 2021 break.
- Don't drop missing rows blindly - the missingness patterns can be systematic (skip logic).
- Don't read `_AGE80 = 80` as exactly 80 - it's top-coded.

## 11. Reproducibility

### Environment

```
Python 3.13
pandas 3.0.2
numpy 2.4.4
pyreadstat 1.3.3
scipy 1.17.1
scikit-learn 1.8.0
pyarrow 23.0.1
```

### Running the pipeline

```bash
# from the repo root
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# all years
./venv/bin/python3 scripts/clean_brfss.py

# one year
./venv/bin/python3 scripts/clean_brfss.py --year 2024

# pool (after all years are cleaned)
./venv/bin/python3 scripts/pool_brfss.py
```

### Approximate runtimes

| Step | Time |
|------|-----:|
| 2020 cleaning | ~46s |
| 2021 cleaning | ~52s |
| 2022 cleaning | ~56s |
| 2023 cleaning | ~56s |
| 2024 cleaning | ~53s |
| Pooling | ~31s |
| Total | ~5 min |

### Generated artifacts

Each run writes:
- `documentation/cleaning_log_{year}.txt` - per-year processing log
- `documentation/quality_report_{year}.md` - QC report with distributions
- `documentation/pipeline_summary.json` - machine-readable summary
- `documentation/pooling_log.txt` - pooling log
- `documentation/quality_report_pooled.md` - pooled QC report

### Sanity checks the pipeline runs

- Row counts preserved (no rows dropped during cleaning)
- Survey design vars have no missing/zero/negative weights
- All known sentinel codes (7, 9, 77, 99, ...) eliminated
- Pooled weight ratio = 0.2000
- `sample_weight` is normalized (mean = 1.0)
- Per-year weight sums are within range of the U.S. adult population (~250-265M)
