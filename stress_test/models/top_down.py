"""
Top-down macro-factor stress model.

The top-down approach uses econometric regression relationships between
macro variables and bank-level loss rates, calibrated on historical data.

Implements:
  1. Macro-factor sensitivity model: loss_rate = f(GDP, unemployment, rates, spreads)
  2. Credit loss projection from macro scenario path
  3. Revenue / NII stress from rate and volume effects
  4. Loan-loss provisioning under CECL/IFRS-9 forward-looking approach
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..scenarios.recession_scenarios import RecessionScenario


# ---------------------------------------------------------------------------
# Historical macro-to-loss regression coefficients
# Calibrated to approximate US banking sector loss rates (1990-2023)
# Loss rate = f(ΔGDP, ΔUnemployment, ΔSpread_BBB, ΔRate)
# ---------------------------------------------------------------------------

# Sector-specific macro sensitivities (loss rate per unit of factor shock)
#
# All coefficients are calibrated so that the loss RATE (a decimal fraction,
# e.g. 0.05 = 5%) increases by the stated amount per unit of each input:
#
#   gdp_elasticity     : decimal loss-rate increase per +0.01 (1pp) GDP DECLINE
#                        gdp_shock_pct is negative in recessions; code negates it
#                        e.g. 0.50 → 4.5% GDP decline → +0.045*0.50 = +2.25pp
#
#   unemployment_coeff : decimal loss-rate increase per +1 pp rise in unemployment
#                        e.g. 0.004 → +5pp → +0.020 = +2.0pp
#
#   spread_bbb_coeff   : decimal loss-rate increase per 100bps of BBB spread widening
#                        code divides raw bps by 100 before multiplying
#                        e.g. 0.004 → +500bps → +5*0.004 = +2.0pp
#
#   rate_coeff         : decimal loss-rate increase per 100bps of rate RISE (abs value)
#                        e.g. 0.001 → +200bps → +2*0.001 = +0.2pp
#
# Calibration targets (approximate historical US banking data):
#   Moderate recession : corporate ~2-4%, real-estate ~3-5%, consumer ~3-5%
#   GFC (2008)         : corporate ~5-8%, real-estate ~8-12%, consumer ~6-10%
SECTOR_SENSITIVITIES: Dict[str, Dict[str, float]] = {
    "financials": {
        "gdp_elasticity": 0.50,
        "unemployment_coeff": 0.004,
        "spread_bbb_coeff": 0.004,
        "rate_coeff": 0.001,
        "base_loss_rate": 0.010,
    },
    "real_estate": {
        "gdp_elasticity": 0.60,
        "unemployment_coeff": 0.006,
        "spread_bbb_coeff": 0.005,
        "rate_coeff": 0.0015,
        "base_loss_rate": 0.015,
    },
    "consumer": {
        "gdp_elasticity": 0.45,
        "unemployment_coeff": 0.008,
        "spread_bbb_coeff": 0.003,
        "rate_coeff": 0.0020,
        "base_loss_rate": 0.020,
    },
    "corporate": {
        "gdp_elasticity": 0.50,
        "unemployment_coeff": 0.004,
        "spread_bbb_coeff": 0.004,
        "rate_coeff": 0.001,
        "base_loss_rate": 0.012,
    },
    "energy": {
        "gdp_elasticity": 0.35,
        "unemployment_coeff": 0.003,
        "spread_bbb_coeff": 0.006,
        "rate_coeff": 0.0005,
        "base_loss_rate": 0.018,
    },
    "technology": {
        "gdp_elasticity": 0.40,
        "unemployment_coeff": 0.003,
        "spread_bbb_coeff": 0.003,
        "rate_coeff": 0.0005,
        "base_loss_rate": 0.008,
    },
    "general": {
        "gdp_elasticity": 0.45,
        "unemployment_coeff": 0.005,
        "spread_bbb_coeff": 0.004,
        "rate_coeff": 0.001,
        "base_loss_rate": 0.012,
    },
}


@dataclass
class TopDownResult:
    """Results from top-down macro factor stress model."""

    total_credit_loss: float
    total_market_loss: float
    nii_impact: float            # Net interest income impact (can be negative = loss)
    provision_increase: float    # CECL/IFRS-9 forward-looking provision build
    sector_losses: Dict[str, float]
    macro_scenario: str

    @property
    def total_loss(self) -> float:
        return self.total_credit_loss + self.total_market_loss + self.provision_increase

    @property
    def pre_tax_income_impact(self) -> float:
        """Total pre-tax P&L impact (losses + NII drag)."""
        return self.total_loss - self.nii_impact   # NII may be negative or positive

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Component": "Credit Loss (macro-projected)", "Amount (USD)": self.total_credit_loss},
            {"Component": "Market Loss", "Amount (USD)": self.total_market_loss},
            {"Component": "Provision Build (CECL/IFRS-9)", "Amount (USD)": self.provision_increase},
            {"Component": "NII Impact", "Amount (USD)": self.nii_impact},
            {"Component": "Total P&L Impact", "Amount (USD)": self.total_loss},
        ]
        for sector, loss in self.sector_losses.items():
            rows.append({"Component": f"  {sector.title()} Sector Loss", "Amount (USD)": loss})
        return pd.DataFrame(rows)


class TopDownModel:
    """
    Top-down macro-to-loss model for institutional portfolio stress testing.

    Maps macro scenario shocks onto portfolio loss rates using econometric
    sensitivities calibrated to historical bank loss data.

    Parameters
    ----------
    scenario : RecessionScenario
        The macro stress scenario.
    nii_rate_sensitivity : float
        Change in NII (as fraction of assets) per 100bps rate change.
        Positive = higher rates boost NII.
    provision_multiplier : float
        CECL/IFRS-9 lifetime loss provision = EL × this multiplier (reflects
        forward-looking provisioning beyond 12-month horizon).
    """

    def __init__(
        self,
        scenario: RecessionScenario,
        nii_rate_sensitivity: float = 0.003,
        provision_multiplier: float = 2.5,
    ) -> None:
        self.scenario = scenario
        self.nii_rate_sensitivity = nii_rate_sensitivity
        self.provision_multiplier = provision_multiplier

    # ------------------------------------------------------------------
    # Macro-factor loss projection
    # ------------------------------------------------------------------

    def _sector_loss_rate(self, sector: str) -> float:
        """
        Project stressed loss rate for a given sector.

        Returns
        -------
        float
            Stressed credit loss rate (fraction of EAD/notional).
        """
        sens = SECTOR_SENSITIVITIES.get(sector, SECTOR_SENSITIVITIES["general"])

        # GDP: gdp_shock_pct is negative in a recession (e.g. -0.045).
        # Negate it so a decline → positive loss-rate contribution.
        gdp_impact = sens["gdp_elasticity"] * (-self.scenario.gdp_shock_pct)
        unemp_impact = sens["unemployment_coeff"] * self.scenario.unemployment_rise_pp
        spread_impact = (
            sens["spread_bbb_coeff"]
            * self.scenario.credit_spread_shock.get("BBB", 0.0)
            / 100.0
        )
        rate_impact = sens["rate_coeff"] * abs(self.scenario.yield_curve_shift_bps) / 100.0

        stressed_loss_rate = (
            sens["base_loss_rate"]
            + gdp_impact
            + unemp_impact
            + spread_impact
            + rate_impact
        )
        return max(stressed_loss_rate, 0.0)

    # ------------------------------------------------------------------
    # NII projection
    # ------------------------------------------------------------------

    def _nii_impact(self, portfolio) -> float:
        """
        Estimate Net Interest Income impact from rate shock and volume effects.

        Rate rise → higher NII on floating-rate assets (short-term positive).
        Recession → lower loan volumes, higher credit costs.

        Returns signed amount (positive = income gain, negative = loss).
        """
        total_loans = sum(a.market_value for a in portfolio.loans)
        rate_nii = total_loans * self.nii_rate_sensitivity * (
            self.scenario.yield_curve_shift_bps / 100.0
        )
        # Volume effect: GDP decline → lower loan origination
        volume_effect = total_loans * abs(self.scenario.gdp_shock_pct) * 0.50 * (-0.015)
        return rate_nii + volume_effect

    # ------------------------------------------------------------------
    # Multi-period scenario path
    # ------------------------------------------------------------------

    def project_loss_path(
        self, portfolio, n_periods: int = 9, period_name: str = "quarter"
    ) -> pd.DataFrame:
        """
        Project credit losses quarter-by-quarter over a stress horizon.

        The scenario shock is assumed to materialise gradually:
          - Peaks at ~period 3-4 (severe quarters)
          - Recovers gradually thereafter

        Parameters
        ----------
        portfolio : Portfolio
        n_periods : int
            Number of periods to project (default 9 quarters for DFAST horizon).
        period_name : str
            Label for the time period axis.

        Returns
        -------
        pd.DataFrame
        """
        from ..portfolio.assets import LoanAsset

        # Severity path: ramp up, peak, taper off
        severity_path = _build_severity_path(n_periods)
        rows = []

        loan_total = sum(a.ead for a in portfolio.loans)

        for period, severity in enumerate(severity_path, start=1):
            period_losses: Dict[str, float] = {}
            for sector in set(a.sector for a in portfolio.assets):
                base_rate = self._sector_loss_rate(sector)
                period_rate = base_rate * severity
                sector_ead = sum(
                    a.ead if isinstance(a, LoanAsset) else a.market_value
                    for a in portfolio.assets
                    if a.sector == sector
                )
                period_losses[sector] = period_rate * sector_ead

            total = sum(period_losses.values())
            rows.append({
                period_name: period,
                "severity_index": severity,
                "total_loss": total,
                **{f"{s}_loss": v for s, v in period_losses.items()},
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Portfolio computation
    # ------------------------------------------------------------------

    def compute(self, portfolio) -> TopDownResult:
        """Run top-down macro stress on the portfolio."""
        from ..portfolio.assets import (
            BondAsset,
            EquityAsset,
            LoanAsset,
            RealEstateAsset,
        )

        sector_losses: Dict[str, float] = {}
        total_credit_loss = 0.0

        # Group by sector
        sectors = set(a.sector for a in portfolio.assets)
        for sector in sectors:
            loss_rate = self._sector_loss_rate(sector)
            sector_ead = sum(
                a.ead if isinstance(a, LoanAsset) else a.market_value
                for a in portfolio.assets
                if a.sector == sector
            )
            loss = loss_rate * sector_ead
            sector_losses[sector] = loss
            total_credit_loss += loss

        # Market value loss (bonds + equities)
        market_loss = 0.0
        for asset in portfolio.bonds:
            dy = self.scenario.yield_curve_shift_bps / 10_000.0
            sp = self.scenario.credit_spread_shock.get(asset.rating.value, 0.0) / 10_000.0
            dp = -asset.duration * (dy + sp)
            market_loss += -dp * asset.market_value

        for asset in portfolio.equities:
            market_loss += -asset.beta * self.scenario.equity_market_decline * asset.market_value

        for asset in portfolio.real_estate:
            market_loss += -self.scenario.real_estate_decline * asset.market_value

        # NII
        nii = self._nii_impact(portfolio)

        # Provision build (CECL/IFRS-9 forward-looking)
        provision = total_credit_loss * (self.provision_multiplier - 1.0)

        return TopDownResult(
            total_credit_loss=total_credit_loss,
            total_market_loss=market_loss,
            nii_impact=nii,
            provision_increase=provision,
            sector_losses=sector_losses,
            macro_scenario=self.scenario.name,
        )


def _build_severity_path(n_periods: int) -> np.ndarray:
    """
    Build a severity index path that ramps up, peaks, then recovers.
    Normalised so that the peak = 1.0 and the average ≈ 0.6.
    """
    x = np.linspace(0, np.pi, n_periods)
    path = np.sin(x) ** 0.5
    # Shift to have mild start
    path = 0.1 + 0.9 * path
    return path
