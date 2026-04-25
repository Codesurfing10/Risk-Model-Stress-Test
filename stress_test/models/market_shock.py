"""
Market shock module — instantaneous parallel shocks to asset prices.

Implements:
  - Historical scenario replays (GFC, COVID, Black Monday, etc.)
  - Factor-model shock propagation to individual assets
  - Liquidity-adjusted losses (bid-ask widening + fire-sale discounts)
  - Volatility regime shifts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..portfolio.assets import (
    Asset,
    BondAsset,
    CreditRating,
    DerivativeAsset,
    EquityAsset,
    LoanAsset,
    RealEstateAsset,
)


@dataclass
class AssetShockResult:
    asset_id: str
    name: str
    asset_type: str
    pre_shock_value: float
    post_shock_value: float
    loss: float
    loss_pct: float


@dataclass
class MarketShockResult:
    """Aggregated market shock results across the portfolio."""

    scenario_name: str
    total_loss: float
    loss_by_type: Dict[str, float]
    asset_results: List[AssetShockResult]
    liquidity_adjustment: float = 0.0

    @property
    def total_loss_with_liquidity(self) -> float:
        return self.total_loss + self.liquidity_adjustment

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Component": "Market Shock Loss", "Loss (USD)": self.total_loss},
            {"Component": "Liquidity Adjustment", "Loss (USD)": self.liquidity_adjustment},
            {"Component": "Total (incl. Liquidity)", "Loss (USD)": self.total_loss_with_liquidity},
        ]
        for asset_type, loss in self.loss_by_type.items():
            rows.insert(-2, {"Component": f"  {asset_type.title()} Loss", "Loss (USD)": loss})
        return pd.DataFrame(rows)

    def asset_detail(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "asset_id": r.asset_id,
                "name": r.name,
                "type": r.asset_type,
                "pre_shock": r.pre_shock_value,
                "post_shock": r.post_shock_value,
                "loss": r.loss,
                "loss_pct": r.loss_pct * 100,
            }
            for r in self.asset_results
        ])


class MarketShockModel:
    """
    Applies instantaneous market shocks to an institutional portfolio.

    Parameters
    ----------
    equity_shock : float
        Fractional equity market decline (e.g. -0.30 for -30%).
    rate_shift_bps : float
        Parallel yield curve shift in basis points.
    credit_spread_shock : dict
        Rating → bps widening.
    fx_shock : float
        FX depreciation of non-USD assets vs USD.
    real_estate_shock : float
        Property value decline fraction.
    oil_shock : float
        Oil price change fraction.
    vol_regime_change : float
        Multiplicative change in implied vol (e.g. 1.5 = 50% vol rise).
    apply_liquidity_haircut : bool
        Whether to add liquidity-adjusted fire-sale losses.
    """

    def __init__(
        self,
        equity_shock: float = -0.30,
        rate_shift_bps: float = 200.0,
        credit_spread_shock: Optional[Dict[str, float]] = None,
        fx_shock: float = 0.10,
        real_estate_shock: float = -0.20,
        oil_shock: float = -0.40,
        vol_regime_change: float = 2.0,
        apply_liquidity_haircut: bool = True,
        scenario_name: str = "Custom Market Shock",
    ) -> None:
        self.equity_shock = equity_shock
        self.rate_shift_bps = rate_shift_bps
        self.credit_spread_shock = credit_spread_shock or {
            "AAA": 50, "AA": 100, "A": 200, "BBB": 400, "BB": 700, "B": 1200, "CCC": 2200,
        }
        self.fx_shock = fx_shock
        self.real_estate_shock = real_estate_shock
        self.oil_shock = oil_shock
        self.vol_regime_change = vol_regime_change
        self.apply_liquidity_haircut = apply_liquidity_haircut
        self.scenario_name = scenario_name

    # ------------------------------------------------------------------
    # Liquidity haircut
    # ------------------------------------------------------------------

    _LIQUIDITY_HAIRCUT: Dict[str, float] = {
        "loan": 0.00,           # Held-to-maturity, no immediate mark
        "bond": 0.015,          # 1.5% bid-ask + price impact
        "equity": 0.025,        # 2.5% market impact for large positions
        "derivative": 0.05,     # 5% for OTC derivatives
        "real_estate": 0.08,    # 8% illiquidity premium
    }

    def _liquidity_loss(self, asset: Asset, base_loss: float) -> float:
        """Additional loss from widening bid-ask spreads and forced selling."""
        haircut = self._LIQUIDITY_HAIRCUT.get(asset.asset_type.value, 0.0)
        return abs(asset.market_value) * haircut

    # ------------------------------------------------------------------
    # Per-asset shock application
    # ------------------------------------------------------------------

    def _shock_loan(self, asset: LoanAsset) -> Tuple[float, float]:
        """
        Loans: no immediate MTM, but model as NPV change from spread/PD stress.
        Returns (post_shock_value, loss).
        """
        rating_shock = self.credit_spread_shock.get(asset.rating.value, 0.0)
        # Approximate: PD rises in proportion to macro shock; simple spread discount
        approx_duration = min(asset.maturity_years / 2.0, 3.0)
        dp = -approx_duration * rating_shock / 10_000.0
        stressed_mv = asset.market_value * (1.0 + dp)
        return stressed_mv, asset.market_value - stressed_mv

    def _shock_bond(self, asset: BondAsset) -> Tuple[float, float]:
        rate_dy = self.rate_shift_bps / 10_000.0
        spread_shock = self.credit_spread_shock.get(asset.rating.value, 0.0)
        spread_dy = spread_shock / 10_000.0
        total_dy = rate_dy + spread_dy
        dp = -asset.duration * total_dy + 0.5 * asset.convexity * total_dy ** 2
        stressed_mv = asset.market_value * (1.0 + dp)
        return stressed_mv, asset.market_value - stressed_mv

    def _shock_equity(self, asset: EquityAsset) -> Tuple[float, float]:
        effective_shock = asset.beta * self.equity_shock
        fx_adj = (self.fx_shock * 0.5) if asset.currency != "USD" else 0.0
        total_shock = effective_shock + fx_adj
        stressed_mv = asset.market_value * (1.0 + total_shock)
        return stressed_mv, asset.market_value - stressed_mv

    def _shock_derivative(self, asset: DerivativeAsset) -> Tuple[float, float]:
        """
        Apply scenario shock to a derivative.

        Greeks are applied to dimensionless percentage shocks on the notional.
        The ``underlying`` attribute routes each instrument to the correct factor.
        """
        underlying = asset.underlying.lower()

        if underlying in ("interest_rate", "rate"):
            # delta treated as modified-duration; applied to rate shock
            shock_pct = self.rate_shift_bps / 10_000.0
            pnl_pct = asset.delta * shock_pct + 0.5 * asset.gamma * shock_pct ** 2
        elif underlying in ("equity_index", "equity"):
            shock_pct = self.equity_shock
            pnl_pct = asset.delta * shock_pct + 0.5 * asset.gamma * shock_pct ** 2
        elif underlying == "fx":
            shock_pct = self.fx_shock
            pnl_pct = asset.delta * shock_pct
        elif underlying == "credit":
            bbb_spread_pct = self.credit_spread_shock.get("BBB", 0.0) / 10_000.0
            # sold protection (negative delta) → widen spreads → loss
            pnl_pct = asset.delta * bbb_spread_pct * 5.0
        else:
            shock_pct = self.equity_shock
            pnl_pct = asset.delta * shock_pct

        # Vega P&L: vol rises sharply in stress (~doubled); vega is in $ per unit vol
        vol_shock = (self.vol_regime_change - 1.0)
        pnl_vega = asset.vega * vol_shock if hasattr(asset, "vega") else 0.0

        pnl = pnl_pct * asset.notional + pnl_vega
        if not asset.is_long:
            pnl = -pnl
        stressed_mv = asset.market_value + pnl
        return stressed_mv, -pnl  # loss = positive means adverse

    def _shock_real_estate(self, asset: RealEstateAsset) -> Tuple[float, float]:
        stressed_mv = asset.market_value * (1.0 + self.real_estate_shock)
        return stressed_mv, asset.market_value - stressed_mv

    # ------------------------------------------------------------------
    # Portfolio computation
    # ------------------------------------------------------------------

    def compute(self, portfolio) -> MarketShockResult:
        """Apply market shock to all assets in the portfolio."""
        asset_results: List[AssetShockResult] = []
        loss_by_type: Dict[str, float] = {
            "loan": 0.0, "bond": 0.0, "equity": 0.0,
            "derivative": 0.0, "real_estate": 0.0,
        }
        total_loss = 0.0
        total_liquidity_adj = 0.0

        for asset in portfolio.assets:
            atype = asset.asset_type.value

            if isinstance(asset, LoanAsset):
                stressed_mv, loss = self._shock_loan(asset)
            elif isinstance(asset, BondAsset):
                stressed_mv, loss = self._shock_bond(asset)
            elif isinstance(asset, EquityAsset):
                stressed_mv, loss = self._shock_equity(asset)
            elif isinstance(asset, DerivativeAsset):
                stressed_mv, loss = self._shock_derivative(asset)
            elif isinstance(asset, RealEstateAsset):
                stressed_mv, loss = self._shock_real_estate(asset)
            else:
                stressed_mv = asset.market_value
                loss = 0.0

            liq_loss = self._liquidity_loss(asset, loss) if self.apply_liquidity_haircut else 0.0
            total_liquidity_adj += liq_loss
            loss_pct = loss / asset.market_value if asset.market_value != 0 else 0.0

            asset_results.append(AssetShockResult(
                asset_id=asset.asset_id,
                name=asset.name,
                asset_type=atype,
                pre_shock_value=asset.market_value,
                post_shock_value=stressed_mv,
                loss=loss,
                loss_pct=loss_pct,
            ))

            loss_by_type[atype] = loss_by_type.get(atype, 0.0) + loss
            total_loss += loss

        return MarketShockResult(
            scenario_name=self.scenario_name,
            total_loss=total_loss,
            loss_by_type=loss_by_type,
            asset_results=asset_results,
            liquidity_adjustment=total_liquidity_adj,
        )

    @classmethod
    def from_scenario(cls, scenario) -> "MarketShockModel":
        """Construct a MarketShockModel from a RecessionScenario."""
        return cls(
            equity_shock=scenario.equity_market_decline,
            rate_shift_bps=scenario.yield_curve_shift_bps,
            credit_spread_shock=scenario.credit_spread_shock,
            fx_shock=scenario.fx_depreciation,
            real_estate_shock=scenario.real_estate_decline,
            oil_shock=scenario.oil_price_change,
            scenario_name=scenario.name,
        )
