"""Asset class definitions for institutional portfolio stress testing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class AssetType(str, Enum):
    LOAN = "loan"
    BOND = "bond"
    EQUITY = "equity"
    DERIVATIVE = "derivative"
    REAL_ESTATE = "real_estate"


class CreditRating(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    D = "D"

    @property
    def pd_base(self) -> float:
        """Approximate annual probability of default (Moody's long-run averages)."""
        mapping = {
            "AAA": 0.0001,
            "AA": 0.0003,
            "A": 0.0008,
            "BBB": 0.0020,
            "BB": 0.0100,
            "B": 0.0450,
            "CCC": 0.2000,
            "D": 1.0000,
        }
        return mapping[self.value]

    @property
    def spread_base_bps(self) -> float:
        """Approximate investment-grade / HY credit spread in basis points."""
        mapping = {
            "AAA": 20,
            "AA": 35,
            "A": 60,
            "BBB": 120,
            "BB": 280,
            "B": 500,
            "CCC": 1000,
            "D": 3000,
        }
        return mapping[self.value]


@dataclass
class Asset:
    """Base class for all portfolio assets."""

    asset_id: str
    name: str
    asset_type: AssetType
    notional: float           # Face / notional value in USD
    market_value: float       # Current mark-to-market value in USD
    sector: str = "financials"
    currency: str = "USD"
    country: str = "US"
    metadata: Dict = field(default_factory=dict)

    @property
    def weight(self) -> float:
        return self.market_value

    def stressed_value(self, shock_pct: float) -> float:
        """Return market value after applying a percentage shock (e.g. -0.30)."""
        return self.market_value * (1.0 + shock_pct)


@dataclass
class LoanAsset(Asset):
    """Commercial or consumer loan held on balance sheet."""

    rating: CreditRating = CreditRating.BBB
    pd: float = 0.0          # Annual probability of default (overrides rating)
    lgd: float = 0.45         # Loss given default
    ead: float = 0.0          # Exposure at default (defaults to notional)
    maturity_years: float = 3.0
    collateral_value: float = 0.0
    is_secured: bool = False
    industry: str = "general"

    def __post_init__(self):
        self.asset_type = AssetType.LOAN
        if self.ead == 0.0:
            self.ead = self.notional
        if self.pd == 0.0:
            self.pd = self.rating.pd_base
        if self.market_value == 0.0:
            self.market_value = self.notional

    @property
    def expected_loss(self) -> float:
        return self.pd * self.lgd * self.ead

    def stressed_pd(self, macro_multiplier: float) -> float:
        """Apply macro stress to PD, capped at 100%."""
        return min(self.pd * macro_multiplier, 1.0)

    def stressed_loss(self, macro_multiplier: float) -> float:
        """Expected credit loss under stressed PD."""
        return self.stressed_pd(macro_multiplier) * self.lgd * self.ead


@dataclass
class BondAsset(Asset):
    """Fixed-income bond (sovereign, corporate, structured)."""

    rating: CreditRating = CreditRating.BBB
    coupon: float = 0.05
    yield_to_maturity: float = 0.05
    maturity_years: float = 5.0
    duration: float = 0.0     # Modified duration; computed if 0
    convexity: float = 0.0
    spread_bps: float = 0.0   # Current OAS in bps; defaults from rating
    is_sovereign: bool = False

    def __post_init__(self):
        self.asset_type = AssetType.BOND
        if self.spread_bps == 0.0:
            self.spread_bps = self.rating.spread_base_bps
        if self.duration == 0.0:
            self.duration = self._approx_modified_duration()

    def _approx_modified_duration(self) -> float:
        """Approximate modified duration using Macaulay formula for annual coupon bond."""
        y = self.yield_to_maturity
        c = self.coupon
        n = self.maturity_years
        if c == 0:
            return n / (1 + y)
        # Macaulay duration
        mac = (
            (1 + y) / y
            - (1 + y + n * (c - y)) / (c * ((1 + y) ** n - 1) + y)
        )
        return mac / (1 + y)

    def price_change_from_rate_shock(self, delta_yield: float) -> float:
        """Dollar P&L from parallel yield shift (uses duration + convexity)."""
        dp_pct = -self.duration * delta_yield + 0.5 * self.convexity * delta_yield ** 2
        return self.market_value * dp_pct

    def price_change_from_spread_widening(self, delta_spread_bps: float) -> float:
        """Dollar P&L from credit spread widening."""
        delta_yield = delta_spread_bps / 10_000.0
        return self.price_change_from_rate_shock(delta_yield)


@dataclass
class EquityAsset(Asset):
    """Listed equity or equity fund position."""

    beta: float = 1.0          # Market beta
    idiosyncratic_vol: float = 0.20  # Annualised idiosyncratic vol
    dividend_yield: float = 0.02
    sector_beta: float = 1.0   # Sector-relative sensitivity
    shares: float = 0.0
    price_per_share: float = 0.0

    def __post_init__(self):
        self.asset_type = AssetType.EQUITY
        if self.shares > 0 and self.price_per_share > 0:
            self.market_value = self.shares * self.price_per_share

    def stressed_value(self, market_return: float, idio_shock: float = 0.0) -> float:
        """Return stressed market value given market return and idiosyncratic shock."""
        total_return = self.beta * market_return + idio_shock
        return self.market_value * (1.0 + total_return)


@dataclass
class DerivativeAsset(Asset):
    """OTC or exchange-traded derivative (interest rate, FX, credit)."""

    underlying: str = "equity_index"
    delta: float = 1.0         # First-order sensitivity to underlying
    gamma: float = 0.0         # Second-order sensitivity
    vega: float = 0.0          # Sensitivity to implied volatility
    theta: float = 0.0         # Daily time decay
    notional_multiplier: float = 1.0
    is_long: bool = True
    counterparty_rating: CreditRating = CreditRating.A

    def __post_init__(self):
        self.asset_type = AssetType.DERIVATIVE

    def pnl_from_shock(self, underlying_shock_pct: float, vol_shock_pct: float = 0.0) -> float:
        """
        Estimate P&L using delta-gamma-vega approximation.

        ``underlying_shock_pct`` is a dimensionless fraction (e.g. -0.30 for -30%),
        applied to the notional.  Greeks are also defined per unit notional.
        """
        greek_pnl = (
            self.delta * underlying_shock_pct
            + 0.5 * self.gamma * underlying_shock_pct ** 2
            + self.vega * vol_shock_pct
        ) * self.notional
        return greek_pnl if self.is_long else -greek_pnl


@dataclass
class RealEstateAsset(Asset):
    """Commercial or residential real estate exposure."""

    property_type: str = "commercial"
    ltv_ratio: float = 0.70    # Loan-to-value ratio
    cap_rate: float = 0.05     # Capitalisation rate
    vacancy_rate: float = 0.05
    net_operating_income: float = 0.0

    def __post_init__(self):
        self.asset_type = AssetType.REAL_ESTATE
        if self.net_operating_income == 0.0 and self.market_value > 0:
            self.net_operating_income = self.market_value * self.cap_rate

    def stressed_value(self, price_decline_pct: float, noi_decline_pct: float = 0.0) -> float:
        """Stressed market value given property price decline."""
        return self.market_value * (1.0 + price_decline_pct)

    def implied_ltv_stressed(self, price_decline_pct: float) -> float:
        """LTV after property value decline."""
        stressed_val = self.stressed_value(price_decline_pct)
        if stressed_val <= 0:
            return float("inf")
        debt = self.market_value * self.ltv_ratio
        return debt / stressed_val
