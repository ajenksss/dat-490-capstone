# DAT 490 Capstone - Predicting preventive health behavior from BRFSS

Arizona State University - DAT 490 Data Science Capstone, Spring 2026.

We use the CDC Behavioral Risk Factor Surveillance System (BRFSS) for 2020-2024 to look at which factors best predict participation in preventive health behaviors (checkups, flu shots, exercise, smoking status). The big-picture question is whether income, education, healthcare access, or some combination of those is the strongest predictor.

## Team

- Adam Jenkins
- Jordan Sutherlin
- Jesus Montijo
- Nicholas Calip

## Research questions

**Broad question.** Which socioeconomic factors best predict participation in preventive health behaviors?

- **SQ1.** How important is income compared to education, insurance coverage, and access to care?
- **SQ2.** Can we build a model that predicts whether someone participates in preventive behaviors?
- **SQ3.** Which variable contributes the most to prediction performance?

## Data

We pull the public CDC BRFSS LLCP files for 2020-2024 (one `.XPT` per year, ~1 GB each) and produce cleaned EDA + ML parquets. The cleaned data is too large for GitHub (~500 MB total), so it lives on Kaggle:

- **Kaggle:** https://www.kaggle.com/datasets/ajenks/brfss-2020-2024-cleaned-and-weighted
- **Raw source:** https://www.cdc.gov/brfss/annual_data/annual_data.htm

See `data/README.md` for what to download and where it goes.

## Repo layout

```
dat-490-capstone/
├── README.md
├── requirements.txt
├── data/                       (raw .XPT files go here, gitignored)
│   └── README.md               where to grab the data
├── notebooks/
│   ├── 00_shared_starter.ipynb         shared loading + train/test split
│   ├── 01_eda.ipynb                    main EDA
│   ├── 02_relationships_eda.ipynb      weighted correlations + GLMs
│   ├── 03_baseline_model.ipynb         nested logistic regression (SQ2)
│   ├── 04_spatial_clustering.ipynb     state-level clustering (geographic)
│   ├── 05_nn_entity_embeddings.ipynb   neural net w/ entity embeddings (SQ3)
│   └── 06_cross_model_comparison.ipynb logistic vs RF vs XGB vs NN
├── scripts/
│   ├── clean_brfss.py          per-year cleaning + weighting
│   └── pool_brfss.py           combine 2020-2024 into one file
├── docs/
│   ├── data_processing_report.md   full pipeline writeup
│   ├── data_source_overview.md     sample sizes + figures
│   └── style_guide.md              shared variables / outcomes for the team
└── figures/                    figures referenced from the docs and report
```

## Getting started

```bash
git clone <this-repo>
cd dat-490-capstone

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Two paths from here:

**1. Use the cleaned data from Kaggle (fastest).** Download the parquets from the Kaggle link above and drop them into a `cleaned/` folder. Then jump into the notebooks.

**2. Re-run the pipeline from raw.** Download the five `.XPT` files from the CDC link and put them in `data/raw/`. Then:

```bash
python scripts/clean_brfss.py     # ~5 minutes total
python scripts/pool_brfss.py      # ~30 seconds
```

Outputs land in `cleaned/` (per-year EDA + ML parquets, plus the pooled files).

## Notebook order

We use numbered prefixes because the order matters - 00 sets up the shared train/test split everyone depends on.

| # | Notebook | Lead | What it does |
|---|----------|------|--------------|
| 00 | `00_shared_starter.ipynb` | Team | Loads the pooled ML parquet, harmonizes income / healthcare access, builds the composite preventive-health target, writes the 80/20 stratified split (`train_indices.npy`, `test_indices.npy`). Run this first. |
| 01 | `01_eda.ipynb` | Adam | Distribution of the composite preventive health score; weighted summaries by education, race, age, income; baseline figures used in the report. |
| 02 | `02_relationships_eda.ipynb` | Jordan | Weighted relationships between specific behaviors (checkup, exercise, flu shot) and demographics; weighted correlations + a logistic GLM as a sanity check. |
| 03 | `03_baseline_model.ipynb` | Nicholas | Nested logistic regressions: base demographic model vs. the model + healthcare access. Tests SQ2 with a likelihood-ratio test and AIC comparison. |
| 04 | `04_spatial_clustering.ipynb` | Jesus | State-level clustering of preventive-behavior rates and demographic profiles. |
| 05 | `05_nn_entity_embeddings.ipynb` | Adam | Dual-path neural net with entity embeddings for state, race, education, income; SHAP for variable importance; branch ablation. SQ3. |
| 06 | `06_cross_model_comparison.ipynb` | Team | Final synthesis: logistic regression, random forest, XGBoost, and the NN compared head-to-head on the same test set. |

## Key results (one-line summary)

- Healthcare access (insurance, personal doctor, cost barrier) adds significant predictive lift over demographics + income alone (LR test, p << 0.001).
- Across logistic, RF, XGB, and NN, AUCs land in a similar band (~0.74-0.78); the NN's lift is small but real, mostly from interactions across the demographic embeddings.
- Education is a stronger contributor than income alone in every model we ran, consistent with Cutler & Lleras-Muney (2010).

See `06_cross_model_comparison.ipynb` for the full numbers.

## Acknowledgments

Data: CDC Behavioral Risk Factor Surveillance System (BRFSS).
Coursework: ASU DAT 490, Spring 2026.
