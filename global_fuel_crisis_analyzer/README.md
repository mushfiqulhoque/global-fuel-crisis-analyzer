# ⛽ Global Fuel Crisis Analyzer

> **An end-to-end data science system for analysing and predicting the impact of global fuel crises on oil prices and country-level fuel costs.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Problem Statement

Geopolitical conflicts, OPEC production cuts, and demand shocks routinely disrupt global crude oil supply—with cascading consequences for fuel prices, household budgets, and national inflation. Yet most publicly available analyses are retrospective, country-specific, and disconnected from real-time economic data.

This project builds a **production-quality analytical system** that:

1. Ingests live data from three institutional APIs (FRED, EIA, World Bank)
2. Engineered a rich feature set from raw price and supply time series
3. Trains and compares four forecasting models (Ridge, ARIMA, Random Forest, XGBoost)
4. Simulates arbitrary supply-shock scenarios and computes country-level fuel price and inflation impacts
5. Delivers all results through an interactive Streamlit dashboard

---

## 🏗️ Architecture

```
global_fuel_crisis_analyzer/
│
├── src/
│   ├── config.py            ← Centralised config (API keys, paths, model params)
│   ├── data_collection.py   ← FRED, EIA, World Bank API clients
│   ├── preprocessing.py     ← Cleaning, merging, feature engineering
│   ├── modeling.py          ← Ridge, ARIMA, Random Forest, XGBoost
│   ├── simulation.py        ← Supply shock engine
│   ├── visualization.py     ← Matplotlib & Plotly chart library
│   └── sentiment.py         ← News headline sentiment (bonus)
│
├── app/
│   └── app.py               ← Streamlit dashboard (4-tab interactive app)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── data/
│   ├── raw/                 ← API response CSVs
│   ├── processed/           ← Feature-engineered master dataset
│   └── cache/               ← SQLite request cache (requests-cache)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📡 Data Sources

| Source | Data | Endpoint | Auth |
|--------|------|----------|------|
| **FRED** (St. Louis Fed) | Brent & WTI crude, CPI-energy, natural gas | `api.stlouisfed.org/fred` | Free API key |
| **EIA** (U.S. Energy Info. Admin.) | World crude oil supply (TBPD) | `api.eia.gov/series` | Free API key |
| **World Bank** | GDP per capita, fuel imports %, energy use per capita, oil rents | `api.worldbank.org/v2` | None required |
| **GNews** *(optional)* | Recent oil market headlines | `gnews.io/api/v4` | Free API key |

All API responses are cached to SQLite for 24 hours — safe for development iteration.

---

## 🔬 Methodology

### Data Processing
- Monthly resampling for temporal alignment across sources
- IQR-fenced outlier removal (factor = 3.0; preserves genuine price spikes)
- Forward-fill (≤3 periods) + time-based interpolation for short gaps
- Feature engineering: rolling means/std (3m, 6m, 12m), % changes (1m, 3m, 12m), lag features (1–12m)
- Crisis period labelling (6 historical events since 2000)

### Models

| Model | Type | Notes |
|-------|------|-------|
| **Baseline (Ridge)** | Linear regression | OLS on lag + macro features; L2 regularised |
| **ARIMA/SARIMA** | Time series | `pmdarima.auto_arima` or manual `(2,1,2)×(1,1,1,12)` |
| **Random Forest** | Ensemble | 300 estimators, depth 8; feature importance |
| **XGBoost** | Gradient boosting | 300 rounds, η=0.05; typically best RMSE |

**Evaluation**: RMSE, MAE, R², MAPE on held-out test set (2022–present).

### Supply Shock Simulation

The core engine uses the **short-run price elasticity of supply** (ε ≈ −0.08, from Hamilton 2009 & Kilian 2014):

```
pct_price_change = supply_drop_fraction / elasticity
```

For shocks exceeding 10%, a Gaussian nonlinear amplifier captures the empirically observed *panic premium*. Country-level retail impacts are computed via:

```
retail_delta = base_retail × brent_delta_pct × passthrough_rate
inflation_contribution = (retail_delta / base_retail) × fuel_cpi_weight
```

Country multipliers encode crude-to-pump markups, taxation, and subsidy structures calibrated from IEA and GlobalPetrolPrices.com (2022 baseline).

---

## 📊 Key Results

| Scenario | Brent Δ | Most Impacted Country | Global CPI Proxy |
|----------|---------|-----------------------|------------------|
| 5% supply drop  | +6.25%  | Germany (+4.69%) | +0.026 pp |
| 10% supply drop | +12.50% | Germany (+9.38%) | +0.051 pp |
| 20% supply drop | +38.0%* | Germany (+28.5%) | +0.156 pp |
| 30% supply drop | +80.9%* | Germany (+60.7%) | +0.333 pp |

*Nonlinear amplification applied for >10% shocks.

**Best model**: XGBoost achieves the lowest RMSE on the 2022+ test set.  
**Most important features**: 1-month lag price, 12-month rolling mean, crisis flag.

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/global-fuel-crisis-analyzer.git
cd global-fuel-crisis-analyzer

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your FRED_API_KEY and EIA_API_KEY
```

### 3. Collect data

```bash
cd src
python data_collection.py
```

### 4. Build master dataset

```bash
python preprocessing.py
```

### 5. Train models

```bash
python modeling.py
```

### 6. Run simulation (CLI)

```bash
python simulation.py
```

### 7. Launch the dashboard

```bash
streamlit run app/app.py
```

Visit `http://localhost:8501` in your browser.

---

## 🛠️ Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data collection | `requests`, `requests-cache` |
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `xgboost` |
| Time series | `statsmodels`, `pmdarima` |
| NLP / Sentiment | `textblob`, `transformers` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Utilities | `loguru`, `python-dotenv`, `joblib`, `pydantic` |

---

## 🔮 Future Improvements

1. **Real-time pipeline** — Schedule data collection via Apache Airflow or Prefect; push to a time-series database (InfluxDB or TimescaleDB).
2. **Deep learning** — Replace XGBoost with a Temporal Fusion Transformer (TFT) for multi-horizon probabilistic forecasting.
3. **Geospatial layer** — Add choropleth maps for visualising country vulnerability scores.
4. **Commodity linkages** — Extend to natural gas, coal, and electricity prices for a full energy crisis model.
5. **LLM-augmented reasoning** — Integrate a large language model to auto-generate natural language shock summaries.
6. **FinBERT sentiment** — Replace TextBlob with a finance-domain-tuned BERT for more accurate news sentiment.
7. **Containerisation** — Dockerfile + docker-compose for reproducible, portable deployment.
8. **Unit tests** — pytest suite covering data validation, feature engineering, and simulation math.

---

## 📚 References

- Hamilton, J.D. (2009). *Causes and Consequences of the Oil Shock of 2007–08.* Brookings Papers on Economic Activity.
- Kilian, L. (2014). *Oil Price Shocks: Causes and Consequences.* Annual Review of Resource Economics.
- Baumeister, C. & Kilian, L. (2016). *Forty Years of Oil Price Fluctuations.* Journal of Economic Perspectives.
- IEA (2023). *Oil Market Report.* International Energy Agency.

---

## 👤 Author

**[Md Mushfiqul Hoque]** 
Built as a portfolio project demonstrating end-to-end data science engineering.
---

## 📝 License



This project is free and open source — created by **Md Mushfiqul Hoque**.
