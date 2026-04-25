"""
Bottom-up stress loss model — instrument-level credit and market loss calculation.

The bottom-up approach:
  1. For each loan/bond: apply stressed PD × LGD × EAD (credit loss)
  2. For each bond: apply duration × stressed yield shock (market loss)
  3. For each equity: apply beta × market shock + idiosyncratic
  4. Aggregate to portfolio level with concentration adjustments

Uses obligor-level correlation via sector Vasicek model for credit losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..portfolio.assets import (
    AssetType,
    BondAsset,
    CreditRating,
    DerivativeAsset,
    EquityAsset,
    LoanAsset,
    RealEstateAsset,
)
from ..scenarios.recession_scenarios import RecessionScenario


@dataclass
class ObligorLoss:
    asset_id: str
    name: str
    asset_type: str
    base_el: float             # Expected loss (pre-stress)
    stressed_el: float         # Expected loss (post-stress)
    incremental_loss: float    # stressed - base
    pd: float
    lgd: float
    ead: float


@dataclass
class BottomUpResult:
    """Aggregated bottom-up stress results."""

    total_base_el: float
    total_stressed_el: float
    total_incremental_loss: float
    credit_loss: float
    market_loss: float
    obligor_losses: List[ObligorLoss]
    concentration_adjustment: float = 0.0
    unexpected_loss_99: float = 0.0

    @property
    def total_loss(self) -> float:
        return (
            self.total_stressed_el
            + self.market_loss
            + self.concentration_adjustment
        )

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Component": "Base Expected Loss", "Loss (USD)": self.total_base_el},
            {"Component": "Stressed Expected Loss", "Loss (USD)": self.total_stressed_el},
            {"Component": "Incremental Credit Loss", "Loss (USD)": self.total_incremental_loss},
            {"Component": "Market Value Loss", "Loss (USD)": self.market_loss},
            {"Component": "Concentration Adjustment", "Loss (USD)": self.concentration_adjustment},
            {"Component": "Unexpected Loss (99%)", "Loss (USD)": self.unexpected_loss_99},
            {"Component": "Total Stressed Loss", "Loss (USD)": self.total_loss},
        ]
        return pd.DataFrame(rows)

    def obligor_detail(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "asset_id": o.asset_id,
                "name": o.name,
                "type": o.asset_type,
                "pd": o.pd,
                "lgd": o.lgd,
                "ead": o.ead,
                "base_el": o.base_el,
                "stressed_el": o.stressed_el,
                "incremental_loss": o.incremental_loss,
            }
            for o in self.obligor_losses
        ])


class BottomUpModel:
    """
    Bottom-up stress test engine using obligor-level PD/LGD/EAD.

    Uses the single-factor Vasicek model to account for systematic
    correlation between obligors and compute Unexpected Loss at
    a given confidence level.

    Parameters
    ----------
    scenario : RecessionScenario
        The macro stress scenario to apply.
    asset_correlation : float
        Intra-sector asset correlation (Vasicek model, default 0.15 for corporates).
    ul_confidence : float
        Confidence level for Unexpected Loss calculation.
    """

    def __init__(
        self,
        scenario: RecessionScenario,
        asset_correlation: float = 0.15,
        ul_confidence: float = 0.999,
    ) -> None:
        self.scenario = scenario
        self.asset_correlation = asset_correlation
        self.ul_confidence = ul_confidence

    # ------------------------------------------------------------------
    # Vasicek / ASRF model for credit loss
    # ------------------------------------------------------------------

    def vasicek_quantile_pd(self, pd: float, rho: float, confidence: float) -> float:
        """
        Conditional PD at given confidence under Vasicek single-factor model.
        This is the Basel II ASRF formula.

        q = N( (N^-1(PD) - sqrt(rho)*N^-1(1-confidence)) / sqrt(1-rho) )
        """
        if pd <= 0 or pd >= 1:
            return pd
        nrm = stats.norm
        q = nrm.cdf(
            (nrm.ppf(pd) - np.sqrt(rho) * nrm.ppf(1 - confidence))
            / np.sqrt(1 - rho)
        )
        return float(np.clip(q, 0, 1))

    # ------------------------------------------------------------------
    # Per-asset stress loss
    # ------------------------------------------------------------------

    def _stress_loan(self, asset: LoanAsset) -> ObligorLoss:
        stressed_pd = asset.stressed_pd(self.scenario.pd_multiplier)
        stressed_lgd = min(asset.lgd * self.scenario.lgd_multiplier, 1.0)
        base_el = asset.pd * asset.lgd * asset.ead
        stressed_el = stressed_pd * stressed_lgd * asset.ead
        return ObligorLoss(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type="loan",
            base_el=base_el,
            stressed_el=stressed_el,
            incremental_loss=stressed_el - base_el,
            pd=stressed_pd,
            lgd=stressed_lgd,
            ead=asset.ead,
        )

    def _stress_bond(self, asset: BondAsset) -> ObligorLoss:
        """Bond credit loss: migration to default + MTM from spread widening."""
        stressed_pd = min(asset.rating.pd_base * self.scenario.pd_multiplier, 1.0)
        stressed_lgd = 0.40 * self.scenario.lgd_multiplier  # unsecured bond LGD
        base_el = asset.rating.pd_base * 0.40 * asset.market_value
        stressed_el = stressed_pd * stressed_lgd * asset.market_value

        # Add MTM loss from yield/spread shock
        delta_spread = self.scenario.credit_spread_shock.get(asset.rating.value, 0.0)
        delta_rate = self.scenario.yield_curve_shift_bps / 10_000.0
        dy = delta_rate + delta_spread / 10_000.0
        mtm_loss = -(asset.price_change_from_rate_shock(dy))

        total_stressed_el = stressed_el + mtm_loss

        return ObligorLoss(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type="bond",
            base_el=base_el,
            stressed_el=total_stressed_el,
            incremental_loss=total_stressed_el - base_el,
            pd=stressed_pd,
            lgd=stressed_lgd,
            ead=asset.market_value,
        )

    def _stress_equity(self, asset: EquityAsset) -> ObligorLoss:
        shock = asset.beta * self.scenario.equity_market_decline
        loss = -shock * asset.market_value
        return ObligorLoss(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type="equity",
            base_el=0.0,
            stressed_el=loss,
            incremental_loss=loss,
            pd=0.0,
            lgd=0.0,
            ead=asset.market_value,
        )

    def _stress_real_estate(self, asset: RealEstateAsset) -> ObligorLoss:
        loss = -self.scenario.real_estate_decline * asset.market_value
        return ObligorLoss(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type="real_estate",
            base_el=0.0,
            stressed_el=loss,
            incremental_loss=loss,
            pd=0.0,
            lgd=0.0,
            ead=asset.market_value,
        )

    def _stress_derivative(self, asset: DerivativeAsset) -> ObligorLoss:
        """
        Stress a derivative using notional-based percentage shocks routed
        by the ``underlying`` attribute of the derivative.
        """
        underlying = asset.underlying.lower()

        if underlying in ("interest_rate", "rate"):
            shock_pct = self.scenario.yield_curve_shift_bps / 10_000.0
            pnl_pct = asset.delta * shock_pct + 0.5 * asset.gamma * shock_pct ** 2
        elif underlying in ("equity_index", "equity"):
            shock_pct = self.scenario.equity_market_decline
            pnl_pct = asset.delta * shock_pct + 0.5 * asset.gamma * shock_pct ** 2
        elif underlying == "fx":
            shock_pct = self.scenario.fx_depreciation
            pnl_pct = asset.delta * shock_pct
        elif underlying == "credit":
            bbb_spread_pct = self.scenario.credit_spread_shock.get("BBB", 0.0) / 10_000.0
            pnl_pct = asset.delta * bbb_spread_pct * 5.0
        else:
            shock_pct = self.scenario.equity_market_decline
            pnl_pct = asset.delta * shock_pct

        pnl_vega = asset.vega * 1.0 if hasattr(asset, "vega") else 0.0
        pnl = pnl_pct * asset.notional + pnl_vega
        if not asset.is_long:
            pnl = -pnl
        loss = -pnl
        return ObligorLoss(
            asset_id=asset.asset_id,
            name=asset.name,
            asset_type="derivative",
            base_el=0.0,
            stressed_el=loss,
            incremental_loss=loss,
            pd=0.0,
            lgd=0.0,
            ead=abs(asset.notional),
        )

    # ------------------------------------------------------------------
    # Concentration adjustment (HHI-based)
    # ------------------------------------------------------------------

    def _concentration_adjustment(self, portfolio, base_credit_loss: float) -> float:
        """
        Add a granularity adjustment for concentration risk.
        Uses portfolio HHI — more concentrated → higher adjustment.
        """
        hhi = portfolio.herfindahl_index()
        # Adjustment: HHI above 0.05 (20 names) = concentration surcharge
        excess_hhi = max(hhi - 0.05, 0.0)
        adj = base_credit_loss * excess_hhi * 2.0
        return adj

    # ------------------------------------------------------------------
    # Unexpected loss via Vasicek
    # ------------------------------------------------------------------

    def _unexpected_loss(self, portfolio) -> float:
        """
        Compute economic capital (Unexpected Loss) at self.ul_confidence.
        Sum over loans of conditional_PD × LGD × EAD at the given confidence.
        """
        ul = 0.0
        for asset in portfolio.assets:
            if isinstance(asset, LoanAsset):
                cond_pd = self.vasicek_quantile_pd(
                    asset.stressed_pd(self.scenario.pd_multiplier),
                    self.asset_correlation,
                    self.ul_confidence,
                )
                stressed_lgd = min(asset.lgd * self.scenario.lgd_multiplier, 1.0)
                ul += cond_pd * stressed_lgd * asset.ead
        return ul

    # ------------------------------------------------------------------
    # Portfolio computation
    # ------------------------------------------------------------------

    def compute(self, portfolio) -> BottomUpResult:
        """Run bottom-up stress test on all assets."""
        obligor_losses: List[ObligorLoss] = []
        credit_loss = 0.0
        market_loss = 0.0

        for asset in portfolio.assets:
            if isinstance(asset, LoanAsset):
                ol = self._stress_loan(asset)
                credit_loss += ol.stressed_el
            elif isinstance(asset, BondAsset):
                ol = self._stress_bond(asset)
                credit_loss += ol.stressed_el
            elif isinstance(asset, EquityAsset):
                ol = self._stress_equity(asset)
                market_loss += ol.stressed_el
            elif isinstance(asset, RealEstateAsset):
                ol = self._stress_real_estate(asset)
                market_loss += ol.stressed_el
            elif isinstance(asset, DerivativeAsset):
                ol = self._stress_derivative(asset)
                market_loss += ol.stressed_el
            else:
                continue
            obligor_losses.append(ol)

        total_base_el = sum(o.base_el for o in obligor_losses)
        total_stressed_el = credit_loss
        total_incremental = total_stressed_el - total_base_el

        conc_adj = self._concentration_adjustment(portfolio, credit_loss)
        ul_99 = self._unexpected_loss(portfolio)

        return BottomUpResult(
            total_base_el=total_base_el,
            total_stressed_el=total_stressed_el,
            total_incremental_loss=total_incremental,
            credit_loss=credit_loss,
            market_loss=market_loss,
            obligor_losses=obligor_losses,
            concentration_adjustment=conc_adj,
            unexpected_loss_99=ul_99,
        )
