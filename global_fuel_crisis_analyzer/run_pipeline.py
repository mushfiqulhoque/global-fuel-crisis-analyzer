"""
run_pipeline.py — One-shot pipeline runner.

Runs the full pipeline in sequence:
  1. Data collection
  2. Preprocessing
  3. Model training & evaluation
  4. Simulation demo

Run from the project root:
    python run_pipeline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger

logger.add("pipeline.log", rotation="10 MB", retention="7 days")


def main():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   Global Fuel Crisis Analyzer — Pipeline     ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── Step 1: Data Collection ────────────────────────────────────────────────
    logger.info("\n[Step 1/4] Data Collection")
    from data_collection import collect_all_data
    raw = collect_all_data(start="2000-01-01")

    # ── Step 2: Preprocessing ──────────────────────────────────────────────────
    logger.info("\n[Step 2/4] Preprocessing & Feature Engineering")
    from preprocessing import build_master_dataset, get_train_test
    master = build_master_dataset(
        fred_df=raw.get("fred"),
        eia_df=raw.get("eia"),
        save=True,
    )
    train, test = get_train_test(master)

    # ── Step 3: Model Training ─────────────────────────────────────────────────
    logger.info("\n[Step 3/4] Model Training & Evaluation")
    from modeling import ModelComparison
    mc = ModelComparison()
    metrics_df, preds_df = mc.run(train, test, fit_arima=False)

    logger.info("\n── Model Results ──")
    logger.info("\n" + metrics_df.to_string())

    best_name, best_model = mc.best_model()
    logger.success(f"Best model: {best_name}  (RMSE={metrics_df.loc[best_name, 'RMSE']:.3f})")

    # ── Step 4: Simulation ─────────────────────────────────────────────────────
    logger.info("\n[Step 4/4] Supply Shock Simulation")
    from simulation import simulate_supply_shock

    scenarios = [
        ("Russia-Ukraine style conflict", 15),
        ("Major Gulf disruption",         25),
        ("Pandemic demand shock reversal", 10),
    ]
    for name, drop in scenarios:
        result = simulate_supply_shock(drop, base_price=85.0, scenario_name=name)
        logger.info("\n" + result.summary())

    logger.success("\n✓ Pipeline complete! Launch the dashboard with:")
    logger.success("    streamlit run app/app.py")


if __name__ == "__main__":
    main()
