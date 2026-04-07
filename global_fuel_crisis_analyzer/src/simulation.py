"""
simulation.py — Supply-shock simulation engine.

Core function: simulate_supply_shock(percent_drop, base_price, ...)

The model uses a short-run price elasticity of oil supply calibrated from
peer-reviewed literature (Hamilton 2009; Kilian 2014; Baumeister & Kilian 2016).

Outputs per simulation
-----------------------
• Predicted oil price change (USD/barrel)
• Country-wise retail fuel price impact (USD/litre equivalent)
• Inflation proxy (CPI impact in percentage points)
• Monthly household fuel cost increase per country
"""

from __future__ import annotations

import math
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
# Data classes for structured outputs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountryImpact:
    """Fuel-price and inflation impact for a single country."""
    iso3:                  str
    name:                  str
    region:                str
    base_retail_price_usd: float    # USD/litre, baseline
    new_retail_price_usd:  float    # USD/litre, post-shock
    retail_price_delta:    float    # absolute change, USD/litre
    retail_price_pct:      float    # % change
    monthly_cost_increase: float    # USD/month per 60-litre/month household
    inflation_contribution:float    # pp added to headline CPI

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ShockResult:
    """Full output bundle from simulate_supply_shock."""
    scenario_name:          str
    supply_drop_pct:        float
    base_brent_price:       float   # USD/barrel
    shocked_brent_price:    float   # USD/barrel
    brent_price_delta:      float   # USD/barrel change
    brent_price_pct:        float   # % change
    global_inflation_proxy: float   # average CPI impact across countries (pp)
    country_impacts:        list[CountryImpact] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return country impacts as a tidy DataFrame."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Price model: short-run supply elasticity
# ─────────────────────────────────────────────────────────────────────────────

def _compute_brent_shock(
    base_price:   float,
    supply_drop:  float,
    elasticity:   float = SUPPLY_ELASTICITY,
    nonlinear:    bool  = True,
) -> float:
    """
    Estimate shocked Brent crude price.

    Model
    -----
    Standard arc-elasticity formula adapted for supply shocks:
        pct_price_change = supply_drop / elasticity

    With nonlinear=True we apply a Gaussian amplification for large shocks,
    reflecting the empirical observation that markets over-react to severe
    disruptions (panic premium).

    Parameters
    ----------
    base_price   : current Brent price (USD/barrel)
    supply_drop  : supply disruption as a *negative* fraction (e.g., -0.10 = -10%)
    elasticity   : short-run price elasticity of supply (negative number)
    nonlinear    : apply shock amplification for large disruptions

    Returns
    -------
    float — new shocked price (USD/barrel)
    """
    if elasticity >= 0:
        raise ValueError("Elasticity must be negative for supply.")
    if not (-1.0 < supply_drop < 0):
        raise ValueError("supply_drop must be in (-1, 0); e.g., -0.15 for a 15% drop.")

    # Arc-elasticity: pct_price_change = pct_supply_change / elasticity
    pct_price_change = supply_drop / elasticity   # e.g., -0.10 / -0.08 = +1.25 = +125%

    if nonlinear and supply_drop < -0.10:
        # Amplification factor: exp(0.5 * excess) where excess is the drop beyond 10%
        excess = abs(supply_drop) - 0.10
        amplification = math.exp(0.5 * excess)
        pct_price_change *= amplification
        logger.debug(f"  Nonlinear amplification applied: {amplification:.3f}x")

    shocked = base_price * (1 + pct_price_change)
    return max(shocked, 0.0)


def _compute_retail_impact(
    iso3:               str,
    brent_delta_pct:    float,
    base_brent_price:   float,
    passthrough:        float = PASSTHROUGH_RATE,
    monthly_litres:     float = 60.0,
) -> CountryImpact:
    """
    Calculate country-level retail fuel price impact.

    Parameters
    ----------
    iso3             : ISO 3-letter country code
    brent_delta_pct  : % change in Brent crude price
    base_brent_price : baseline Brent price (USD/barrel)
    passthrough      : fraction of crude price change passed to retail (0-1)
    monthly_litres   : assumed household consumption (litres/month)
    """
    info       = COUNTRIES.get(iso3, {"name": iso3, "region": "Unknown"})
    multiplier = COUNTRY_FUEL_MULTIPLIER.get(iso3, 0.025)

    base_retail = base_brent_price * multiplier
    delta_retail = base_retail * brent_delta_pct * passthrough
    new_retail   = base_retail + delta_retail

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


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def simulate_supply_shock(
    percent_drop:     float,
    base_price:       float      = 85.0,
    countries:        Optional[list[str]] = None,
    scenario_name:    str        = "",
    elasticity:       float      = SUPPLY_ELASTICITY,
    passthrough:      float      = PASSTHROUGH_RATE,
    monthly_litres:   float      = 60.0,
    nonlinear:        bool       = True,
) -> ShockResult:
    """
    Simulate a global oil supply shock and compute its downstream impacts.

    Parameters
    ----------
    percent_drop   : supply reduction in percent (positive integer, e.g., 15 means -15%)
    base_price     : baseline Brent crude price in USD/barrel (default 85.0)
    countries      : list of ISO-3 codes to include (default: all in COUNTRIES)
    scenario_name  : human-readable label for the scenario
    elasticity     : short-run price elasticity of supply
    passthrough    : retail passthrough rate (0–1)
    monthly_litres : assumed household fuel consumption (litres/month)
    nonlinear      : apply nonlinear amplification for large shocks (>10%)

    Returns
    -------
    ShockResult dataclass with full impact breakdown

    Examples
    --------
    >>> result = simulate_supply_shock(15, base_price=85)
    >>> print(result.summary())
    >>> df = result.to_dataframe()
    """
    if percent_drop <= 0:
        raise ValueError("percent_drop must be a positive number (e.g., 15 for -15%).")
    if percent_drop >= 100:
        raise ValueError("percent_drop must be < 100.")

    if countries is None:
        countries = list(COUNTRIES.keys())

    supply_drop_fraction = -percent_drop / 100.0
    scenario_name = scenario_name or f"{percent_drop}% Supply Shock"

    logger.info(f"Simulating: {scenario_name}")
    logger.info(f"  Base Brent : ${base_price:.2f}")
    logger.info(f"  Supply drop: {percent_drop}%")

    # ── Oil price shock ────────────────────────────────────────────────────────
    shocked_price = _compute_brent_shock(
        base_price, supply_drop_fraction, elasticity, nonlinear
    )
    brent_delta   = shocked_price - base_price
    brent_pct     = brent_delta / base_price

    logger.info(f"  Shocked Brent : ${shocked_price:.2f}  (Δ {brent_delta:+.2f}, {brent_pct*100:+.1f}%)")

    # ── Country impacts ────────────────────────────────────────────────────────
    impacts: list[CountryImpact] = []
    for iso3 in countries:
        impact = _compute_retail_impact(
            iso3, brent_pct, base_price, passthrough, monthly_litres
        )
        impacts.append(impact)

    # ── Global inflation proxy ─────────────────────────────────────────────────
    global_inflation = float(np.mean([c.inflation_contribution for c in impacts]))

    result = ShockResult(
        scenario_name=scenario_name,
        supply_drop_pct=percent_drop,
        base_brent_price=base_price,
        shocked_brent_price=round(shocked_price, 2),
        brent_price_delta=round(brent_delta, 2),
        brent_price_pct=round(brent_pct * 100, 2),
        global_inflation_proxy=round(global_inflation, 4),
        country_impacts=impacts,
    )

    logger.success(f"Simulation complete. Global CPI impact ≈ +{global_inflation:.3f} pp")
    return result


def run_scenario_sweep(
    drops:      list[float] = [5, 10, 15, 20, 25, 30],
    base_price: float       = 85.0,
) -> pd.DataFrame:
    """
    Run simulate_supply_shock across a range of disruption severities.

    Returns
    -------
    pd.DataFrame with one row per (country × scenario)
    """
    all_rows = []
    for drop in drops:
        result = simulate_supply_shock(drop, base_price=base_price)
        df = result.to_dataframe()
        df["supply_drop_pct"]     = drop
        df["shocked_brent"]       = result.shocked_brent_price
        df["brent_delta_pct"]     = result.brent_price_pct
        all_rows.append(df)

    sweep = pd.concat(all_rows, ignore_index=True)
    logger.success(f"Scenario sweep complete. {len(sweep)} rows generated.")
    return sweep


if __name__ == "__main__":
    result = simulate_supply_shock(percent_drop=20, base_price=85)
    print(result.summary())

    sweep = run_scenario_sweep()
    print("\nScenario sweep (first 10 rows):")
    print(sweep.head(10))
