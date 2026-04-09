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


# -----------------------------
# DATA CLASSES
# -----------------------------

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
            f"  Brent crude : ${self.base_brent_price:.2f} → ${self.shocked_brent_price:.2f} "
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


# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def _compute_brent_shock(
    base_price: float,
    supply_drop: float,
    elasticity: float = SUPPLY_ELASTICITY,
) -> float:
    pct_price_change = supply_drop / elasticity
    pct_price_change = min(pct_price_change, 0.80)
    shocked = base_price * (1 + pct_price_change)
    return max(shocked, base_price)


def _compute_retail_impact(
    iso3: str,
    brent_delta_pct: float,
    base_brent_price: float,
    passthrough: float = PASSTHROUGH_RATE,
    monthly_litres: float = 60.0,
) -> CountryImpact:
    info = COUNTRIES.get(iso3, {"name": iso3, "region": "Unknown"})
    multiplier = COUNTRY_FUEL_MULTIPLIER.get(iso3, 0.025)

    base_retail = base_brent_price * multiplier
    delta_retail = base_retail * brent_delta_pct * passthrough
    new_retail = base_retail + delta_retail

    inflation_contribution = (delta_retail / base_retail) * INFLATION_FUEL_WEIGHT * 100

    return CountryImpact(
        iso3=iso3,
        name=info["name"],
        region=info["region"],
        base_retail_price_usd=round(base_retail, 4),
        new_retail_price_usd=round(new_retail, 4),
        retail_price_delta=round(delta_retail, 4),
        retail_price_pct=round(brent_delta_pct * passthrough * 100, 2),
        monthly_cost_increase=round(delta_retail * monthly_litres, 2),
        inflation_contribution=round(inflation_contribution, 4),
    )


def simulate_supply_shock(
    percent_drop: float,
    base_price: float = 85.0,
    countries: Optional[list[str]] = None,
    scenario_name: str = "",
    elasticity: float = SUPPLY_ELASTICITY,
    passthrough: float = PASSTHROUGH_RATE,
    monthly_litres: float = 60.0,
) -> ShockResult:

    if percent_drop <= 0:
        raise ValueError("percent_drop must be a positive number.")
    if percent_drop >= 100:
        raise ValueError("percent_drop must be < 100.")

    if countries is None:
        countries = list(COUNTRIES.keys())

    supply_drop_fraction = -percent_drop / 100.0
    scenario_name = scenario_name or f"{percent_drop}% Supply Shock"

    logger.info(f"Simulating: {scenario_name}")

    shocked_price = _compute_brent_shock(base_price, supply_drop_fraction, elasticity)
    brent_delta = shocked_price - base_price
    brent_pct = brent_delta / base_price

    impacts: list[CountryImpact] = []
    for iso3 in countries:
        impact = _compute_retail_impact(
            iso3, brent_pct, base_price, passthrough, monthly_litres
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


# -----------------------------
# ADDED FIX: run_scenario_sweep
# -----------------------------

def run_scenario_sweep(
    scenarios: list[float],
    base_price: float = 85.0,
) -> pd.DataFrame:
    """
    Runs multiple supply shock scenarios and returns combined results.
    """

    results = []

    for s in scenarios:
        result = simulate_supply_shock(percent_drop=s, base_price=base_price)
        df = result.to_dataframe()
        df["scenario"] = s
        results.append(df)

    return pd.concat(results, ignore_index=True)
