# ⛽ Global Fuel Crisis Analyzer

> **An end-to-end data science system for analysing and predicting the impact of global fuel crises on oil prices and country-level fuel costs.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://global-fuel-crisis-analyzer.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🔗 **Live Demo**: [global-fuel-crisis-analyzer.streamlit.app](https://global-fuel-crisis-analyzer.streamlit.app)

---

## 📌 Problem Statement

Geopolitical conflicts, OPEC production cuts, and demand shocks routinely disrupt global crude oil supply — with cascading consequences for fuel prices, household budgets, and national inflation. Yet most publicly available analyses are retrospective, country-specific, and disconnected from real-time economic data.

This project builds a production-quality analytical system that:

1. Ingests live data from three institutional APIs (FRED, EIA, World Bank)
2. Engineers a rich feature set from raw price and supply time series
3. Trains and compares four forecasting models (Ridge, ARIMA, Random Forest, XGBoost)
4. Simulates arbitrary supply-shock scenarios and computes country-level fuel price and inflation impacts
5. Delivers all results through an interactive Streamlit dashboard

---

## 🖥️ Dashboard Preview

### Supply Shock Simulator
Simulate a global crude oil supply disruption and see real-time impact across 15 countries — including predicted Brent price, retail fuel delta, and monthly household cost increase.

![Supply Shock Simulator](screenshots/supply_shock_simulator.png)

### Multi-Scenario Sensitivity Analysis
Compare fuel price impact across multiple supply drop scenarios (5%–30%) with line charts and a country × severity heatmap.

![Sensitivity Analysis](screenshots/sensitivity_analysis.png)

### Full Impact Table
Detailed breakdown per country — base retail price, new retail price, delta, and CPI contribution.

![Impact Table](screenshots/impact_table.png)

### Historical Oil Price Explorer (2000–2026)
Interactive Brent & WTI price chart with all major crisis periods annotated — Gulf War II, GFC, Arab Spring, OPEC Price War, COVID Crash, Ukraine War, Israel–Hamas–Iran conflict, and US–Iran tensions.

![Historical Explorer](screenshots/historical_explorer.png)

### Model Performance Metrics
XGBoost achieves R² = 0.907 on the 2022–2026 test set. All three models are compared on RMSE, MAE, R², and MAPE.

![Model Metrics](screenshots/model_metrics.png)

### Predictions vs Actual
All three models tracked against real Brent crude prices on the held-out test set.

![Predictions](screenshots/predictions.png)

### XGBoost Feature Importances
Top features are `wti_crude_lag1m`, `brent_crude_roll3m_mean`, and `brent_crude_pct3m` — confirming the model is correctly learning from lagged price signals.

![Feature Importance](screenshots/feature_importance.png)

---

## 🏗️ Project Structure

```
global_fuel_crisis_analyzer/
│
├── src/
│   ├── config.py            ← Centralised config (API keys, paths, model params)
│   ├── data_collection.py   ← FRED, EIA, World Bank API clients + SQLite caching
│   ├── preprocessing.py     ← Cleaning, merging, lag/rolling feature engineering
│   ├── modeling.py          ← Ridge, ARIMA, Random Forest, XGBoost + comparison
│   ├── simulation.py        ← Supply shock engine (simulate_supply_shock)
│   ├── visualization.py     ← Matplotlib & Plotly chart library
│   └── sentiment.py         ← News headline NLP sentiment (TextBlob + HuggingFace)
│
├── app/
│   └── app.py               ← Streamlit dashboard (4-tab interactive app)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📡 Data Sources

| Source | Data | Auth |
|--------|------|------|
| **FRED** (St. Louis Fed) | Brent & WTI crude prices, CPI-energy, natural gas | Free API key |
| **EIA** (U.S. Energy Info. Admin.) | World crude oil supply (TBPD, monthly) | Free API key |
| **World Bank** | GDP per capita, fuel imports %, energy use per capita, oil rents | None required |
| **GNews** *(optional)* | Recent oil market headlines for sentiment analysis | Free API key |

All API responses are cached to SQLite for 24 hours to avoid rate limits during development.

---

## 🔬 Methodology

### Feature Engineering
- Monthly resampling for temporal alignment across all sources
- IQR-fenced outlier removal (factor = 3.0 — preserves genuine price spikes)
- Forward-fill (≤3 periods) + time-based interpolation for short gaps
- Lag features: 1m, 2m, 3m, 6m, 12m
- Rolling mean and std: 3m, 6m, 12m windows
- Percentage change features: 1m, 3m, 12m
- Crisis period labelling for 8 historical events (2000–2026)

### Models

| Model | Type | Test R² | Notes |
|-------|------|---------|-------|
| **XGBoost** ★ | Gradient boosting | **0.907** | Best performer; lag + rolling features |
| **Baseline (Ridge)** | Linear regression | 0.844 | L2 regularised; StandardScaler pipeline |
| **Random Forest** | Ensemble | 0.828 | 500 estimators, depth 8 |
| **ARIMA/SARIMA** | Time series | — | Manual order (2,1,2)×(1,1,1,12) |

**Evaluation**: RMSE, MAE, R², MAPE on held-out test set (Jan 2022 – Mar 2026).

**Key finding**: Top features are `wti_crude_lag1m`, `brent_crude_roll3m_mean`, and `brent_crude_pct3m` — confirming the model learns from recent price momentum, not from spurious variables.

### Supply Shock Simulation

The core engine uses the **short-run price elasticity of supply** (ε ≈ −0.08) calibrated from Hamilton (2009) and Kilian (2014):

```
pct_price_change = supply_drop_fraction / elasticity
```

For shocks exceeding 10%, a nonlinear amplifier captures the empirically observed panic premium. Country-level retail impacts:

```
retail_delta        = base_retail × brent_delta_pct × passthrough_rate
inflation_proxy     = (retail_delta / base_retail) × fuel_cpi_weight
```

Country multipliers encode crude-to-pump markups, taxation, and subsidy structures calibrated from IEA and GlobalPetrolPrices.com (2022 baseline).

---

## 📊 Key Results

| Scenario | Brent Δ | Brent Price | Most Impacted | Global CPI |
|----------|---------|-------------|----------------|------------|
| 5% supply drop  | +62.5%  | $138/bbl | Germany (+46.9%) | +0.026 pp |
| 10% supply drop | +125%   | $191/bbl | Germany (+93.8%) | +0.051 pp |
| 15% supply drop | +192%   | $248/bbl | Germany (+41.2%) | +0.156 pp |
| 20% supply drop | +263%   | $309/bbl | Germany (+197%)  | +1.266 pp |
| 30% supply drop | +414%   | $437/bbl | Germany (+311%)  | +0.333 pp |

*Nonlinear amplification applied for shocks > 10%.*

**Germany, Turkey, and the UK are consistently the most vulnerable** due to high import dependency and fuel tax structures. Saudi Arabia and Russia are least affected due to domestic production.

---

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/mushfiqulhoque/global-fuel-crisis-analyzer.git
cd global-fuel-crisis-analyzer
pip install -r requirements.txt
```

### 2. Add API keys (optional — app works without them using synthetic data)

```bash
cp .env.example .env
# Edit .env and add FRED_API_KEY and EIA_API_KEY (both free)
```

### 3. Launch the dashboard

```bash
python -m streamlit run global_fuel_crisis_analyzer/app/app.py
```

Visit `http://localhost:8501` in your browser.

### 4. (Optional) Run the full data pipeline

```bash
cd global_fuel_crisis_analyzer/src
python data_collection.py     # fetch from APIs
python preprocessing.py       # build master dataset
python modeling.py            # train and evaluate models
python simulation.py          # run supply shock demo
```

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
| Utilities | `loguru`, `python-dotenv`, `joblib` |

---

## 🌍 Countries Covered

Bangladesh, Brazil, China, Germany, India, Indonesia, Japan, Nigeria, Pakistan, Russia, Saudi Arabia, South Africa, Turkey, United Kingdom, United States

---

## 🔮 Future Improvements

1. **Real-time pipeline** — Apache Airflow or Prefect for scheduled data collection
2. **Deep learning** — Temporal Fusion Transformer (TFT) for probabilistic forecasting
3. **Geospatial layer** — Choropleth maps showing country vulnerability scores
4. **Commodity linkages** — Extend to natural gas, coal, and electricity prices
5. **FinBERT sentiment** — Replace TextBlob with finance-domain BERT
6. **Containerisation** — Dockerfile + docker-compose for portable deployment
7. **Unit tests** — pytest suite for data validation and simulation math

---

## 📚 References

- Hamilton, J.D. (2009). *Causes and Consequences of the Oil Shock of 2007–08.* Brookings Papers on Economic Activity.
- Kilian, L. (2014). *Oil Price Shocks: Causes and Consequences.* Annual Review of Resource Economics.
- Baumeister, C. & Kilian, L. (2016). *Forty Years of Oil Price Fluctuations.* Journal of Economic Perspectives.
- IEA (2023). *Oil Market Report.* International Energy Agency.

---

## 👤 Author

**Md Mushfiqul Hoque** — Data Analyst | Aspiring Data Scientist

Built as a portfolio project demonstrating end-to-end data science engineering.

🔗 [Live Demo](https://global-fuel-crisis-analyzer.streamlit.app) · 💻 [GitHub](https://github.com/mushfiqulhoque/global-fuel-crisis-analyzer)
