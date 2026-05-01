# Data

The data files aren't in this repo - they're too big for GitHub (`.XPT` files are ~1 GB each, cleaned parquets are ~500 MB combined).

## Two ways to get the data

### Option 1: Cleaned parquets from Kaggle (recommended)

If you just want to run the notebooks, grab the cleaned + weighted parquets from Kaggle:

https://www.kaggle.com/datasets/ajenks/brfss-2020-2024-cleaned-and-weighted

Download and unzip into a `cleaned/` folder at the repo root, like:

```
dat-490-capstone/
├── cleaned/
│   ├── brfss_2020_eda.parquet
│   ├── brfss_2020_ml.parquet
│   ├── ...
│   ├── brfss_2020_2024_pooled_eda.parquet
│   └── brfss_2020_2024_pooled_ml.parquet
└── ...
```

That's all the notebooks need.

### Option 2: Raw CDC files (full pipeline)

If you want to re-run the cleaning pipeline from scratch:

1. Download the five LLCP files (one per year) from the CDC BRFSS page:
   - https://www.cdc.gov/brfss/annual_data/annual_data.htm
2. Put them in `data/raw/`:

   ```
   data/raw/
   ├── LLCP2020.XPT
   ├── LLCP2021.XPT
   ├── LLCP2022.XPT
   ├── LLCP2023.XPT
   └── LLCP2024.XPT
   ```
3. From the repo root:
   ```bash
   python scripts/clean_brfss.py    # produces cleaned/brfss_<year>_*.parquet
   python scripts/pool_brfss.py     # produces the pooled parquets
   ```

Cleaning the full 2020-2024 set takes ~5 minutes on a modern laptop; pooling is another ~30 seconds.

## Reference docs

CDC publishes codebooks, calculated-variable guides, and complex-sampling-weight notes for each year. The relevant PDFs are listed at the bottom of `docs/data_processing_report.md`.
