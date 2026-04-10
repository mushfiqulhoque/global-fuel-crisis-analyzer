"""
simulation.py — Supply-shock simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import (
    COUNTRIES,
    COUNTRY_FUEL_MULTIPLIER,
    INFLATION_FUEL_WEIGHT,
    PASSTHROUGH_RATE,
    SUPPLY_ELASTICITY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Country-specific structural parameters
# (passthrough, elasticity_adj, subsidy_factor, volatility_factor)
#
# passthrough      — fraction of crude shock that reaches the pump
# elasticity_adj   — market sensitivity relative to global average
# subsidy_factor   — <1 where government subsidies absorb part of the shock
# volatility_factor— exchange-rate exposure / structural amplifier
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_PARAMS = {
    "USA": (0.72, 1.00, 1.00, 1.00),
    "GBR": (0.78, 1.05, 1.02, 1.04),
    "DEU": (0.80, 1.08, 1.00, 1.06),
    "JPN": (0.74, 1.02, 0.98, 0.95),
    "CHN": (0.55, 0.82, 0.78, 0.88),
    "IND": (0.60, 0.88, 0.72, 1.12),
    "BRA": (0.65, 0.92, 0.85, 1.18),
    "BGD": (0.62, 0.90, 0.68, 1.22),
    "PAK": (0.63, 0.91, 0.65, 1.25),
    "IDN": (0.58, 0.85, 0.70, 1.10),
    "TUR": (0.75, 1.03, 0.90, 1.30),
    "ZAF": (0.68, 0.95, 0.88, 1.15),
    "NGA": (0.45, 0.70, 0.55, 1.35),
    "SAU": (0.25, 0.40, 0.20, 0.45),
    "RUS": (0.30, 0.50, 0.35, 0.60),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountryImpact:
    iso3: str
    name: str
    region: str
    base_retail_price_usd: float
    new_retail_price_usd: float
    retail_price_delta: float
    retail_price_pct: float
    monthly_cost_increase: float
    inflation_contribution: float

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ShockResult:
    scenario_name: str
    supply_drop_pct: float
    base_brent_price: float
    shocked_brent_price: float
    brent_price_delta: float
    brent_price_pct: float
    global_inflation_proxy: float
    country_impacts: list[CountryImpact] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.country_impacts])

    def summary(self) -> str:
        lines = [
            f"{'═'*60}",
            f"  Scenario : {self.scenario_name}",
            f"  Supply drop : {self.supply_drop_pct:.1f}%",
            f"  Brent crude : ${self.base_brent_price:.2f} → "
            f"${self.shocked_brent_price:.2f} "
            f"(Δ {self.brent_price_delta:+.2f} USD/bbl, {self.brent_price_pct:+.1f}%)",
            f"  Global CPI proxy : {self.global_inflation_proxy:+.2f} pp",
            f"{'─'*60}",
            f"  {'Country':<20} {'Retail Δ ($/L)':>14} {'Δ%':>7} {'Monthly cost Δ':>16}",
            f"  {'─'*20} {'─'*14} {'─'*7} {'─'*16}",
        ]
        for c in sorted(self.country_impacts, key=lambda x: x.retail_price_pct, reverse=True):
            lines.append(
                f"  {c.name:<20} {c.retail_price_delta:>+13.4f} "
                f"{c.retail_price_pct:>+6.1f}% {c.monthly_cost_increase:>+15.2f}"
            )
        lines.append(f"{'═'*60}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_brent_shock(
    base_price: float,
    supply_drop_pct: float,
    elasticity: float = SUPPLY_ELASTICITY,
) -> float:
    """
    Nonlinear (tanh-saturating) price response to supply disruption.

    - Small shocks  → near-linear response
    - Large shocks  → diminishing marginal price increase
                      (demand destruction, SPR releases, fuel switching)

    supply_drop_pct is a positive number, e.g. 15 = 15% drop.
    """
    drop_frac    = supply_drop_pct / 100.0
    linear_pct   = drop_frac / abs(elasticity)
    # tanh saturates: at 30% drop ≈ +45%, at 50% drop ≈ +65% (not +150%)
    saturated_pct = 0.85 * np.tanh(linear_pct / 0.85)
    return max(base_price * (1.0 + saturated_pct), base_price)


def _compute_retail_impact(
    iso3: str,
    brent_shocked: float,
    base_brent_price: float,
    monthly_litres: float = 60.0,
) -> CountryImpact:
    """
    Compute country-specific retail fuel price impact.

    Dollar change and percentage change are derived from the same delta,
    so they are always internally consistent.
    """
    info       = COUNTRIES.get(iso3, {"name": iso3, "region": "Unknown"})
    multiplier = COUNTRY_FUEL_MULTIPLIER.get(iso3, 0.025)

    pt, e_adj, sub, vol = COUNTRY_PARAMS.get(iso3, (0.65, 0.95, 0.90, 1.00))

    brent_delta_frac = (brent_shocked - base_brent_price) / base_brent_price

    # Base retail USD/litre
    base_retail = base_brent_price * multiplier

    # Country-specific effective pass-through
    effective_pt = pt * sub * e_adj

    # Retail delta USD/litre  — volatility_factor applied last (exchange-rate effect)
    delta_retail = base_retail * brent_delta_frac * effective_pt * vol
    new_retail   = base_retail + delta_retail

    # Percentage — computed FROM delta so $ and % are always consistent
    retail_pct = (delta_retail / base_retail) * 100.0

    return CountryImpact(
        iso3=iso3,
        name=info["name"],
        region=info["region"],
        base_retail_price_usd=round(base_retail, 4),
        new_retail_price_usd=round(new_retail, 4),
        retail_price_delta=round(delta_retail, 4),
        retail_price_pct=round(retail_pct, 2),
        monthly_cost_increase=round(delta_retail * monthly_litres, 2),
        inflation_contribution=round(
            (delta_retail / base_retail) * INFLATION_FUEL_WEIGHT * 100.0, 4
        ),
    )


def simulate_supply_shock(
    percent_drop: float,
    base_price: float = 85.0,
    countries: Optional[list[str]] = None,
    scenario_name: str = "",
    elasticity: float = SUPPLY_ELASTICITY,
    passthrough: float = PASSTHROUGH_RATE,  # kept for API compat; overridden per-country
    monthly_litres: float = 60.0,
) -> ShockResult:

    if percent_drop <= 0:
        raise ValueError("percent_drop must be a positive number.")
    if percent_drop >= 100:
        raise ValueError("percent_drop must be < 100.")

    if countries is None:
        countries = list(COUNTRIES.keys())

    scenario_name = scenario_name or f"{percent_drop}% Supply Shock"
    logger.info(f"Simulating: {scenario_name}")

    shocked_price = _compute_brent_shock(base_price, percent_drop, elasticity)
    brent_delta   = shocked_price - base_price
    brent_pct     = (brent_delta / base_price) * 100.0

    impacts: list[CountryImpact] = []
    for iso3 in countries:
        impact = _compute_retail_impact(
            iso3=iso3,
            brent_shocked=shocked_price,
            base_brent_price=base_price,
            monthly_litres=monthly_litres,
        )
        impacts.append(impact)

    global_inflation = float(np.mean([c.inflation_contribution for c in impacts]))

    return ShockResult(
        scenario_name=scenario_name,
        supply_drop_pct=percent_drop,
        base_brent_price=base_price,
        shocked_brent_price=shocked_price,
        brent_price_delta=brent_delta,
        brent_price_pct=brent_pct,
        global_inflation_proxy=global_inflation,
        country_impacts=impacts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario_sweep(
    drops: list[float],
    base_price: float = 85.0,
) -> pd.DataFrame:
    """
    Run multiple supply shock scenarios and return combined results.

    Parameters
    ----------
    drops      : list of supply-drop percentages (e.g. [5, 10, 15, 20, 25, 30])
    base_price : baseline Brent crude price in USD/bbl

    Returns
    -------
    pd.DataFrame — one row per country per scenario, with a 'scenario' column.
    """
    results = []
    for drop in drops:
        result = simulate_supply_shock(percent_drop=drop, base_price=base_price)
        df     = result.to_dataframe()
        df["scenario"] = drop
        results.append(df)
    return pd.concat(results, ignore_index=True)
