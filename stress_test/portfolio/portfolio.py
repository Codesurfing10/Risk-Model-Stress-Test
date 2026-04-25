"""Portfolio container for institutional multi-asset stress testing."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .assets import (
    Asset,
    AssetType,
    BondAsset,
    CreditRating,
    DerivativeAsset,
    EquityAsset,
    LoanAsset,
    RealEstateAsset,
)


class Portfolio:
    """
    Represents a large institutional portfolio holding loans, bonds, equities,
    derivatives, and real estate.

    Attributes
    ----------
    name : str
        Descriptive name of the portfolio / entity.
    assets : list of Asset
        All positions held.
    tier1_capital : float
        Tier-1 regulatory capital (USD).
    tier2_capital : float
        Tier-2 regulatory capital (USD).
    total_liabilities : float
        Total liabilities used for leverage calculations (USD).
    """

    def __init__(
        self,
        name: str = "Institutional Portfolio",
        tier1_capital: float = 0.0,
        tier2_capital: float = 0.0,
        total_liabilities: float = 0.0,
    ) -> None:
        self.name = name
        self.tier1_capital = tier1_capital
        self.tier2_capital = tier2_capital
        self.total_liabilities = total_liabilities
        self.assets: List[Asset] = []

    # ------------------------------------------------------------------
    # Portfolio construction helpers
    # ------------------------------------------------------------------

    def add_asset(self, asset: Asset) -> None:
        self.assets.append(asset)

    def add_assets(self, assets: List[Asset]) -> None:
        self.assets.extend(assets)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @property
    def total_market_value(self) -> float:
        return sum(a.market_value for a in self.assets)

    @property
    def total_notional(self) -> float:
        return sum(a.notional for a in self.assets)

    @property
    def total_capital(self) -> float:
        return self.tier1_capital + self.tier2_capital

    @property
    def leverage_ratio(self) -> float:
        """Simple leverage ratio: total assets / tier-1 capital."""
        if self.tier1_capital <= 0:
            return float("inf")
        return self.total_market_value / self.tier1_capital

    @property
    def debt_to_equity(self) -> float:
        if self.total_capital <= 0:
            return float("inf")
        return self.total_liabilities / self.total_capital

    # ------------------------------------------------------------------
    # Asset-type breakdowns
    # ------------------------------------------------------------------

    def _assets_by_type(self, asset_type: AssetType) -> List[Asset]:
        return [a for a in self.assets if a.asset_type == asset_type]

    @property
    def loans(self) -> List[LoanAsset]:
        return self._assets_by_type(AssetType.LOAN)  # type: ignore[return-value]

    @property
    def bonds(self) -> List[BondAsset]:
        return self._assets_by_type(AssetType.BOND)  # type: ignore[return-value]

    @property
    def equities(self) -> List[EquityAsset]:
        return self._assets_by_type(AssetType.EQUITY)  # type: ignore[return-value]

    @property
    def derivatives(self) -> List[DerivativeAsset]:
        return self._assets_by_type(AssetType.DERIVATIVE)  # type: ignore[return-value]

    @property
    def real_estate(self) -> List[RealEstateAsset]:
        return self._assets_by_type(AssetType.REAL_ESTATE)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Risk-weighted assets (simplified Basel III)
    # ------------------------------------------------------------------

    _RW_MAP: Dict[str, float] = {
        "AAA": 0.20,
        "AA": 0.20,
        "A": 0.50,
        "BBB": 1.00,
        "BB": 1.00,
        "B": 1.50,
        "CCC": 1.50,
        "D": 1.50,
    }

    def risk_weighted_assets(self) -> float:
        """Simplified RWA calculation for Basel III CET1 ratio."""
        rwa = 0.0
        for asset in self.assets:
            rw = 1.0  # default 100% risk weight
            if isinstance(asset, LoanAsset):
                rw = self._RW_MAP.get(asset.rating.value, 1.0)
                if asset.is_secured:
                    rw *= 0.75
                rwa += asset.ead * rw
            elif isinstance(asset, BondAsset):
                rw = 0.0 if asset.is_sovereign else self._RW_MAP.get(asset.rating.value, 1.0)
                rwa += asset.market_value * rw
            elif isinstance(asset, EquityAsset):
                rwa += asset.market_value * 1.50   # listed equity 150%
            elif isinstance(asset, DerivativeAsset):
                rwa += abs(asset.market_value) * 1.00
            elif isinstance(asset, RealEstateAsset):
                rwa += asset.market_value * (0.50 if asset.property_type == "residential" else 1.00)
        return rwa

    def cet1_ratio(self) -> float:
        rwa = self.risk_weighted_assets()
        if rwa <= 0:
            return float("inf")
        return self.tier1_capital / rwa

    # ------------------------------------------------------------------
    # Summary DataFrames
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        rows = []
        for a in self.assets:
            row = {
                "asset_id": a.asset_id,
                "name": a.name,
                "type": a.asset_type.value,
                "sector": a.sector,
                "notional": a.notional,
                "market_value": a.market_value,
            }
            if isinstance(a, (LoanAsset, BondAsset)):
                row["rating"] = a.rating.value
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def concentration_by_sector(self) -> pd.Series:
        totals: Dict[str, float] = defaultdict(float)
        total_mv = self.total_market_value or 1.0
        for a in self.assets:
            totals[a.sector] += a.market_value
        return pd.Series({k: v / total_mv for k, v in totals.items()}).sort_values(ascending=False)

    def concentration_by_type(self) -> pd.Series:
        totals: Dict[str, float] = defaultdict(float)
        total_mv = self.total_market_value or 1.0
        for a in self.assets:
            totals[a.asset_type.value] += a.market_value
        return pd.Series({k: v / total_mv for k, v in totals.items()}).sort_values(ascending=False)

    def herfindahl_index(self) -> float:
        """HHI concentration index across sectors (0=diversified, 1=concentrated)."""
        conc = self.concentration_by_sector()
        return float((conc ** 2).sum())
