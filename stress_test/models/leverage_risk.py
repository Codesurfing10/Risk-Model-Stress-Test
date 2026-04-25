"""
Leverage risk and capital adequacy module (Basel III / IV framework).

Computes:
  - Tier-1 leverage ratio (Basel III, minimum 3%)
  - CET1 capital ratio (minimum 4.5% + 2.5% buffer = 7%)
  - Total capital ratio (minimum 8% + 2.5% buffer = 10.5%)
  - Stressed leverage and capital under shock scenarios
  - Debt-service coverage and interest coverage under stress
  - DFAST / CCAR-style capital adequacy assessment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basel III / IV regulatory minimums
# ---------------------------------------------------------------------------

BASEL_MINIMUMS = {
    "cet1_ratio": 0.045,              # CET1 minimum
    "cet1_conservation_buffer": 0.025,# Capital conservation buffer
    "tier1_ratio": 0.060,             # Tier-1 minimum
    "total_capital_ratio": 0.080,     # Total capital minimum
    "leverage_ratio": 0.030,          # Tier-1 leverage ratio
    "g_sib_surcharge": 0.010,         # G-SIB surcharge (1-3.5%, assume 1% here)
}

DFAST_PASS_THRESHOLD = {
    "cet1_ratio": 0.045,
    "tier1_ratio": 0.060,
    "total_capital_ratio": 0.080,
    "leverage_ratio": 0.030,
}


@dataclass
class CapitalMetrics:
    """Pre- and post-stress capital metrics."""

    # Capital inputs
    cet1_capital: float
    tier1_capital: float
    tier2_capital: float
    total_assets: float
    risk_weighted_assets: float
    total_liabilities: float

    # Pre-stress ratios
    cet1_ratio: float = 0.0
    tier1_leverage_ratio: float = 0.0
    tier1_rwa_ratio: float = 0.0
    total_capital_ratio: float = 0.0
    debt_to_equity: float = 0.0
    leverage_multiple: float = 0.0    # assets / equity

    def __post_init__(self):
        rwa = self.risk_weighted_assets or 1e-9
        self.cet1_ratio = self.cet1_capital / rwa
        self.tier1_rwa_ratio = self.tier1_capital / rwa
        self.total_capital_ratio = (self.tier1_capital + self.tier2_capital) / rwa
        self.tier1_leverage_ratio = (
            self.tier1_capital / self.total_assets if self.total_assets > 0 else 0.0
        )
        equity = self.tier1_capital + self.tier2_capital
        self.debt_to_equity = self.total_liabilities / equity if equity > 0 else float("inf")
        self.leverage_multiple = self.total_assets / equity if equity > 0 else float("inf")

    def passes_basel(self) -> Dict[str, bool]:
        return {
            "cet1_ratio": self.cet1_ratio >= BASEL_MINIMUMS["cet1_ratio"],
            "tier1_ratio": self.tier1_rwa_ratio >= BASEL_MINIMUMS["tier1_ratio"],
            "total_capital_ratio": self.total_capital_ratio >= BASEL_MINIMUMS["total_capital_ratio"],
            "leverage_ratio": self.tier1_leverage_ratio >= BASEL_MINIMUMS["leverage_ratio"],
        }

    def capital_headroom(self) -> Dict[str, float]:
        """Basis-point headroom above minimum for each ratio."""
        return {
            "cet1_ratio_bp": (self.cet1_ratio - BASEL_MINIMUMS["cet1_ratio"]) * 10_000,
            "tier1_ratio_bp": (self.tier1_rwa_ratio - BASEL_MINIMUMS["tier1_ratio"]) * 10_000,
            "total_capital_bp": (
                self.total_capital_ratio - BASEL_MINIMUMS["total_capital_ratio"]
            ) * 10_000,
            "leverage_bp": (
                self.tier1_leverage_ratio - BASEL_MINIMUMS["leverage_ratio"]
            ) * 10_000,
        }


@dataclass
class LeverageStressResult:
    """Results from leverage and capital adequacy stress test."""

    pre_stress: CapitalMetrics
    post_stress: CapitalMetrics
    stressed_loss: float
    capital_depletion_pct: float          # % of pre-stress capital consumed
    cet1_ratio_change_bp: float
    leverage_ratio_change_bp: float
    passes_dfast: bool
    capital_shortfall: float              # negative = surplus
    buffer_exhausted: bool
    detailed_checks: Dict[str, bool] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Metric": "Pre-Stress CET1 Ratio", "Value": f"{self.pre_stress.cet1_ratio:.2%}"},
            {"Metric": "Post-Stress CET1 Ratio", "Value": f"{self.post_stress.cet1_ratio:.2%}"},
            {"Metric": "CET1 Change (bps)", "Value": f"{self.cet1_ratio_change_bp:.1f}"},
            {"Metric": "Pre-Stress Tier-1 Leverage", "Value": f"{self.pre_stress.tier1_leverage_ratio:.2%}"},
            {"Metric": "Post-Stress Tier-1 Leverage", "Value": f"{self.post_stress.tier1_leverage_ratio:.2%}"},
            {"Metric": "Leverage Change (bps)", "Value": f"{self.leverage_ratio_change_bp:.1f}"},
            {"Metric": "Stressed Loss (USD)", "Value": f"{self.stressed_loss:,.0f}"},
            {"Metric": "Capital Depletion %", "Value": f"{self.capital_depletion_pct:.1%}"},
            {"Metric": "Capital Shortfall (USD)", "Value": f"{self.capital_shortfall:,.0f}"},
            {"Metric": "Passes DFAST", "Value": str(self.passes_dfast)},
            {"Metric": "Conservation Buffer Exhausted", "Value": str(self.buffer_exhausted)},
        ]
        return pd.DataFrame(rows)


class LeverageRiskModel:
    """
    Computes leverage and capital adequacy metrics under stress.

    Parameters
    ----------
    is_gsib : bool
        Whether the firm is a Global Systemically Important Bank.
    gsib_surcharge : float
        Additional CET1 buffer required for G-SIBs.
    nii_stress_pct : float
        Net interest income reduction under stress (reduces capital build).
    """

    def __init__(
        self,
        is_gsib: bool = True,
        gsib_surcharge: float = 0.01,
        nii_stress_pct: float = 0.30,
    ) -> None:
        self.is_gsib = is_gsib
        self.gsib_surcharge = gsib_surcharge
        self.nii_stress_pct = nii_stress_pct

    # ------------------------------------------------------------------
    # Capital metrics computation
    # ------------------------------------------------------------------

    def compute_capital_metrics(self, portfolio) -> CapitalMetrics:
        """Compute pre-stress capital metrics from a Portfolio."""
        rwa = portfolio.risk_weighted_assets()
        total_assets = portfolio.total_market_value
        cet1 = portfolio.tier1_capital   # Simplified: CET1 ≈ Tier-1

        return CapitalMetrics(
            cet1_capital=cet1,
            tier1_capital=portfolio.tier1_capital,
            tier2_capital=portfolio.tier2_capital,
            total_assets=total_assets,
            risk_weighted_assets=rwa,
            total_liabilities=portfolio.total_liabilities,
        )

    # ------------------------------------------------------------------
    # Stressed capital
    # ------------------------------------------------------------------

    def compute_stressed_metrics(
        self,
        portfolio,
        stressed_loss: float,
        stressed_rwa_increase_pct: float = 0.20,
    ) -> CapitalMetrics:
        """
        Compute capital metrics after applying a stressed loss.

        Parameters
        ----------
        portfolio : Portfolio
        stressed_loss : float
            Post-tax losses applied to capital (positive = adverse).
        stressed_rwa_increase_pct : float
            RWA typically rises in stress due to rating downgrades and model risk.
        """
    def compute_stressed_metrics(
        self,
        portfolio,
        stressed_loss: float,
        stressed_rwa_increase_pct: float = 0.20,
    ) -> CapitalMetrics:
        """
        Compute capital metrics after applying a stressed loss.

        Net capital consumption = stressed_loss - pre_provision_net_revenue,
        where PPNR is estimated from the stressed NIM over a 9-quarter horizon
        and partially offsets the loss.

        Parameters
        ----------
        portfolio : Portfolio
        stressed_loss : float
            Pre-tax losses applied to capital (positive = adverse).
        stressed_rwa_increase_pct : float
            RWA typically rises in stress due to rating downgrades and model risk.
        """
        base_rwa = portfolio.risk_weighted_assets()
        stressed_rwa = base_rwa * (1.0 + stressed_rwa_increase_pct)

        # Estimate pre-provision net revenue (NII + fees) over 9-quarter horizon.
        # Stressed NIM ≈ base NIM × (1 - nii_stress_pct) on loan portfolio.
        total_loans_mv = sum(a.market_value for a in portfolio.loans)
        annual_nim = total_loans_mv * 0.025          # approximate 2.5% NIM
        stressed_annual_nim = annual_nim * (1.0 - self.nii_stress_pct)
        ppnr_offset = stressed_annual_nim * 2.25     # 9 quarters = 2.25 years

        # Net capital impact (losses net of pre-provision revenue)
        net_capital_impact = max(stressed_loss - ppnr_offset, 0.0)

        # Capital absorbs net losses
        stressed_tier1 = max(portfolio.tier1_capital - net_capital_impact, 0.0)
        stressed_cet1 = stressed_tier1   # Simplified: CET1 ≈ Tier-1

        return CapitalMetrics(
            cet1_capital=stressed_cet1,
            tier1_capital=stressed_tier1,
            tier2_capital=portfolio.tier2_capital * 0.90,   # Tier-2 may be impaired
            total_assets=max(portfolio.total_market_value - net_capital_impact, 0.0),
            risk_weighted_assets=stressed_rwa,
            total_liabilities=portfolio.total_liabilities,
        )

    # ------------------------------------------------------------------
    # Full stress assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        portfolio,
        stressed_loss: float,
        stressed_rwa_increase_pct: float = 0.20,
    ) -> LeverageStressResult:
        """
        Run full leverage / capital adequacy stress assessment.

        Parameters
        ----------
        portfolio : Portfolio
        stressed_loss : float
            Aggregate post-tax stressed loss.
        stressed_rwa_increase_pct : float
            Expected RWA increase under stress.

        Returns
        -------
        LeverageStressResult
        """
        pre = self.compute_capital_metrics(portfolio)
        post = self.compute_stressed_metrics(portfolio, stressed_loss, stressed_rwa_increase_pct)

        cet1_change_bp = (post.cet1_ratio - pre.cet1_ratio) * 10_000
        lev_change_bp = (post.tier1_leverage_ratio - pre.tier1_leverage_ratio) * 10_000

        # Capital depletion
        pre_capital = pre.tier1_capital + pre.tier2_capital
        depletion_pct = stressed_loss / pre_capital if pre_capital > 0 else 1.0

        # DFAST pass/fail
        dfast_checks = {
            "cet1_ratio": post.cet1_ratio >= DFAST_PASS_THRESHOLD["cet1_ratio"],
            "tier1_ratio": post.tier1_rwa_ratio >= DFAST_PASS_THRESHOLD["tier1_ratio"],
            "total_capital_ratio": post.total_capital_ratio >= DFAST_PASS_THRESHOLD["total_capital_ratio"],
            "leverage_ratio": post.tier1_leverage_ratio >= DFAST_PASS_THRESHOLD["leverage_ratio"],
        }
        passes_dfast = all(dfast_checks.values())

        # Conservation buffer exhausted?
        cet1_floor = BASEL_MINIMUMS["cet1_ratio"]
        if self.is_gsib:
            cet1_floor += self.gsib_surcharge
        buffer_exhausted = post.cet1_ratio < (cet1_floor + BASEL_MINIMUMS["cet1_conservation_buffer"])

        # Capital shortfall vs minimum
        required_cet1_capital = DFAST_PASS_THRESHOLD["cet1_ratio"] * post.risk_weighted_assets
        shortfall = required_cet1_capital - post.cet1_capital   # positive = shortfall

        return LeverageStressResult(
            pre_stress=pre,
            post_stress=post,
            stressed_loss=stressed_loss,
            capital_depletion_pct=depletion_pct,
            cet1_ratio_change_bp=cet1_change_bp,
            leverage_ratio_change_bp=lev_change_bp,
            passes_dfast=passes_dfast,
            capital_shortfall=shortfall,
            buffer_exhausted=buffer_exhausted,
            detailed_checks=dfast_checks,
        )

    # ------------------------------------------------------------------
    # Stress testing leverage limits
    # ------------------------------------------------------------------

    def leverage_sensitivity(
        self,
        portfolio,
        loss_range: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute CET1 ratio as a function of loss size to find breakeven capital level.

        Returns
        -------
        pd.DataFrame with columns: loss, cet1_ratio, passes_minimum
        """
        if loss_range is None:
            base_mv = portfolio.total_market_value
            loss_range = np.linspace(0, base_mv * 0.30, 50)

        rows = []
        for loss in loss_range:
            post = self.compute_stressed_metrics(portfolio, loss)
            rows.append({
                "stressed_loss": loss,
                "cet1_ratio": post.cet1_ratio,
                "tier1_leverage_ratio": post.tier1_leverage_ratio,
                "passes_cet1_min": post.cet1_ratio >= DFAST_PASS_THRESHOLD["cet1_ratio"],
            })
        return pd.DataFrame(rows)
