# DAT 490 - project style guide and shared variables

A short reference for the team so we're all using the same dataset, the same outcomes, and the same predictors when we run our individual sub-questions.

## 1. Ground rules

- Reporting figures use the pooled EDA file (`brfss_2020_2024_pooled_eda.parquet`).
- Always weight: use `_LLCPWT_POOLED` for pooled and `_LLCPWT` for single-year analyses.
- Prefer the CDC calculated variables (`_`-prefix) and the pipeline-derived flags over the raw versions - they handle skip logic and missingness more cleanly.
- Use the same anchor outcomes across the whole report. We picked them once - don't pick a different one just because it looks better.
- Every figure should have one caption and 2-4 sentences of interpretation.

## 2. Shared variables

### A. Anchor outcomes (preventive health behaviors)

These are the dependent variables for the modeling sub-questions.

1. `RECENT_CHECKUP` - routine checkup within the past year (1=Yes, 0=No)
2. `GOT_FLUSHOT` - flu shot in the past 12 months (1=Yes, 0=No)
3. `HAS_EXERCISE` - any physical activity in the past 30 days (1=Yes, 0=No)
4. `EVER_SMOKED` - smoked 100+ cigarettes ever (1=Yes, 0=No)
5. `HADMAM` (optional) - had a mammogram. Useful as a clinical preventive marker if the analysis is restricted to the appropriate demographic band.

### B. Socioeconomic predictors (SQ1, SQ3)

Hypothesis: education and income are positively associated with preventive behavior.

1. `_EDUCAG` - education, 4 levels (< HS / HS Grad / Some College / College+). Prefer over raw `EDUCA`.
2. `_INCOMG1` - income group. CDC-calculated; harmonizes the 2021 income-scale change. Prefer over raw `INCOME3`.

### C. Healthcare access predictors (SQ1, SQ3)

Hypothesis: healthcare access adds predictive value on top of socioeconomic variables.

1. `HAS_HEALTH_PLAN` - has any health insurance (1=Yes, 0=No)
2. `HAS_PERSONAL_DOCTOR` - has a personal healthcare provider (1=Yes, 0=No)
3. `COST_BARRIER` - skipped a doctor visit due to cost in the past 12 months (1=Yes, 0=No)

### D. Demographic controls

Always include these as controls when reporting variable importance (SQ2, SQ3).

1. `_AGE_G` (age group) or `_AGE80` (continuous, top-coded at 80)
2. `_IMPRACE` - imputed race/ethnicity. Prefer over `_RACE` to avoid dropping rows.
3. `_SEX`
4. `STATE_FIPS` - state identifier; useful for any regional control or random effect.
