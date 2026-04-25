"""
Predefined macro-economic stress scenarios for banking stress tests.

Each scenario specifies shocks to:
  - GDP growth (annualised %)
  - Unemployment rate change (pp)
  - Equity market decline (%)
  - Credit spread widening per rating bucket (bps)
  - Parallel yield curve shift (bps)
  - FX depreciation vs USD (%)
  - Real-estate price decline (%)
  - Oil price change (%)
  - Interbank / funding spread (bps)
  - PD multiplier (applied to all loan PDs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RecessionScenario:
    """
    Container for a named macro-economic stress scenario.

    All shock magnitudes are signed (negative = adverse).
    """

    name: str
    description: str

    # Macro shocks
    gdp_shock_pct: float = 0.0           # e.g. -0.05 means GDP contracts 5%
    unemployment_rise_pp: float = 0.0     # Percentage-point rise in unemployment
    equity_market_decline: float = 0.0   # Fraction: -0.40 = 40% equity fall
    yield_curve_shift_bps: float = 0.0   # Parallel shift in risk-free rates
    fx_depreciation: float = 0.0         # Local-currency depreciation vs USD

    # Credit market shocks
    credit_spread_shock: Dict[str, float] = field(default_factory=dict)
    # Keys are rating strings e.g. "AAA","BBB" etc.; values in bps

    # Asset-class specific
    real_estate_decline: float = 0.0     # Fraction: -0.25 = 25% property fall
    oil_price_change: float = 0.0        # Fraction
    funding_spread_bps: float = 0.0      # Interbank spread widening

    # Derived/aggregate stress intensity
    pd_multiplier: float = 1.0           # Multiply all loan PDs by this factor
    lgd_multiplier: float = 1.0          # Multiply all LGDs by this factor

    # Severity label
    severity: str = "severe"             # "mild" | "moderate" | "severe" | "extreme"

    def apply_severity_override(self, multiplier: float) -> "RecessionScenario":
        """Return a new scenario with all shock magnitudes scaled by multiplier."""
        import copy
        s = copy.deepcopy(self)
        s.gdp_shock_pct *= multiplier
        s.unemployment_rise_pp *= multiplier
        s.equity_market_decline *= multiplier
        s.yield_curve_shift_bps *= multiplier
        s.fx_depreciation *= multiplier
        s.real_estate_decline *= multiplier
        s.oil_price_change *= multiplier
        s.funding_spread_bps *= multiplier
        s.pd_multiplier = 1.0 + (s.pd_multiplier - 1.0) * multiplier
        s.lgd_multiplier = 1.0 + (s.lgd_multiplier - 1.0) * multiplier
        s.credit_spread_shock = {k: v * multiplier for k, v in s.credit_spread_shock.items()}
        return s


# ---------------------------------------------------------------------------
# Pre-built scenario library
# ---------------------------------------------------------------------------

_DEFAULT_CREDIT_SHOCKS = {
    "AAA": 25,
    "AA": 50,
    "A": 100,
    "BBB": 200,
    "BB": 400,
    "B": 700,
    "CCC": 1500,
}

_SEVERE_CREDIT_SHOCKS = {
    "AAA": 60,
    "AA": 120,
    "A": 250,
    "BBB": 500,
    "BB": 800,
    "B": 1400,
    "CCC": 2500,
}

_EXTREME_CREDIT_SHOCKS = {
    "AAA": 150,
    "AA": 300,
    "A": 600,
    "BBB": 1000,
    "BB": 1800,
    "B": 3000,
    "CCC": 5000,
}

SCENARIO_LIBRARY: Dict[str, RecessionScenario] = {
    # ------------------------------------------------------------------
    # 2008-style Global Financial Crisis
    # ------------------------------------------------------------------
    "gfc_2008": RecessionScenario(
        name="2008 Global Financial Crisis",
        description=(
            "Severe financial crisis triggered by subprime mortgage collapse, "
            "systemic bank failures, and global credit freeze."
        ),
        gdp_shock_pct=-0.045,
        unemployment_rise_pp=5.0,
        equity_market_decline=-0.55,
        yield_curve_shift_bps=-200,      # Flight to safety; rates fall
        fx_depreciation=0.10,            # USD strengthens; EM currencies fall
        credit_spread_shock=_SEVERE_CREDIT_SHOCKS,
        real_estate_decline=-0.35,
        oil_price_change=-0.60,
        funding_spread_bps=350,
        pd_multiplier=5.0,
        lgd_multiplier=1.20,
        severity="extreme",
    ),

    # ------------------------------------------------------------------
    # COVID-19 (2020) sudden stop
    # ------------------------------------------------------------------
    "covid_2020": RecessionScenario(
        name="COVID-19 Pandemic Shock",
        description=(
            "Sudden economic stop due to global pandemic, lockdowns, and supply "
            "chain disruption; followed by unprecedented fiscal/monetary stimulus."
        ),
        gdp_shock_pct=-0.065,
        unemployment_rise_pp=10.0,
        equity_market_decline=-0.34,
        yield_curve_shift_bps=-150,
        fx_depreciation=0.05,
        credit_spread_shock={
            "AAA": 50,
            "AA": 100,
            "A": 200,
            "BBB": 400,
            "BB": 700,
            "B": 1200,
            "CCC": 2200,
        },
        real_estate_decline=-0.15,
        oil_price_change=-0.70,
        funding_spread_bps=150,
        pd_multiplier=4.0,
        lgd_multiplier=1.10,
        severity="severe",
    ),

    # ------------------------------------------------------------------
    # Stagflation scenario (high inflation + recession)
    # ------------------------------------------------------------------
    "stagflation": RecessionScenario(
        name="Stagflation",
        description=(
            "Combination of persistent high inflation forcing aggressive rate hikes "
            "alongside economic contraction. Inspired by 1970s stagflation and "
            "2022-style energy price shocks."
        ),
        gdp_shock_pct=-0.03,
        unemployment_rise_pp=3.0,
        equity_market_decline=-0.35,
        yield_curve_shift_bps=300,       # Rates rise sharply
        fx_depreciation=0.15,
        credit_spread_shock={
            "AAA": 40,
            "AA": 80,
            "A": 160,
            "BBB": 350,
            "BB": 600,
            "B": 1000,
            "CCC": 2000,
        },
        real_estate_decline=-0.25,
        oil_price_change=0.80,           # Oil surges
        funding_spread_bps=200,
        pd_multiplier=3.0,
        lgd_multiplier=1.15,
        severity="severe",
    ),

    # ------------------------------------------------------------------
    # Sovereign debt crisis (eurozone-style)
    # ------------------------------------------------------------------
    "sovereign_debt_crisis": RecessionScenario(
        name="Sovereign Debt Crisis",
        description=(
            "Sovereign bond market dislocation similar to the 2010-2012 European "
            "debt crisis, with contagion to banking sector and sharp credit tightening."
        ),
        gdp_shock_pct=-0.025,
        unemployment_rise_pp=4.0,
        equity_market_decline=-0.30,
        yield_curve_shift_bps=250,       # Peripheral sovereign yields spike
        fx_depreciation=0.12,
        credit_spread_shock={
            "AAA": 30,
            "AA": 70,
            "A": 150,
            "BBB": 350,
            "BB": 650,
            "B": 1100,
            "CCC": 2200,
        },
        real_estate_decline=-0.20,
        oil_price_change=-0.20,
        funding_spread_bps=180,
        pd_multiplier=3.5,
        lgd_multiplier=1.10,
        severity="severe",
    ),

    # ------------------------------------------------------------------
    # Moderate recession (business-cycle downturn)
    # ------------------------------------------------------------------
    "moderate_recession": RecessionScenario(
        name="Moderate Recession",
        description=(
            "Standard business-cycle recession with moderate GDP contraction "
            "and manageable credit stress."
        ),
        gdp_shock_pct=-0.02,
        unemployment_rise_pp=2.5,
        equity_market_decline=-0.20,
        yield_curve_shift_bps=-75,
        fx_depreciation=0.05,
        credit_spread_shock=_DEFAULT_CREDIT_SHOCKS,
        real_estate_decline=-0.10,
        oil_price_change=-0.25,
        funding_spread_bps=80,
        pd_multiplier=2.0,
        lgd_multiplier=1.05,
        severity="moderate",
    ),

    # ------------------------------------------------------------------
    # Extreme tail / doomsday scenario
    # ------------------------------------------------------------------
    "extreme_tail": RecessionScenario(
        name="Extreme Tail Risk",
        description=(
            "Highly adverse scenario combining simultaneous geopolitical crisis, "
            "sovereign defaults, banking system collapse, and liquidity freeze. "
            "Designed for reverse stress testing."
        ),
        gdp_shock_pct=-0.10,
        unemployment_rise_pp=15.0,
        equity_market_decline=-0.70,
        yield_curve_shift_bps=-300,
        fx_depreciation=0.30,
        credit_spread_shock=_EXTREME_CREDIT_SHOCKS,
        real_estate_decline=-0.50,
        oil_price_change=-0.80,
        funding_spread_bps=500,
        pd_multiplier=10.0,
        lgd_multiplier=1.40,
        severity="extreme",
    ),

    # ------------------------------------------------------------------
    # Rising rate shock (central-bank tightening)
    # ------------------------------------------------------------------
    "rate_shock": RecessionScenario(
        name="Rapid Rate Rise Shock",
        description=(
            "Central bank forced to raise rates sharply to combat inflation, "
            "causing bond-market losses and tightening financial conditions "
            "without a deep recession."
        ),
        gdp_shock_pct=-0.01,
        unemployment_rise_pp=1.0,
        equity_market_decline=-0.15,
        yield_curve_shift_bps=400,
        fx_depreciation=0.08,
        credit_spread_shock={
            "AAA": 20,
            "AA": 40,
            "A": 90,
            "BBB": 200,
            "BB": 380,
            "B": 650,
            "CCC": 1300,
        },
        real_estate_decline=-0.15,
        oil_price_change=0.30,
        funding_spread_bps=120,
        pd_multiplier=1.8,
        lgd_multiplier=1.05,
        severity="moderate",
    ),
}


class ScenarioLibrary:
    """Helper for accessing and composing scenarios."""

    def __init__(self, scenarios: Dict[str, RecessionScenario] = SCENARIO_LIBRARY) -> None:
        self._scenarios = dict(scenarios)

    def get(self, name: str) -> RecessionScenario:
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found. Available: {list(self._scenarios)}")
        return self._scenarios[name]

    def all_scenarios(self) -> Dict[str, RecessionScenario]:
        return dict(self._scenarios)

    def add_custom(self, scenario: RecessionScenario) -> None:
        self._scenarios[scenario.name] = scenario

    def scaled_scenario(self, name: str, severity_multiplier: float) -> RecessionScenario:
        """Return a scaled version of a named scenario."""
        return self.get(name).apply_severity_override(severity_multiplier)
