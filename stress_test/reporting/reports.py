"""
Structured reporting for banking stress test results.

Generates:
  - Executive summary tables
  - Per-scenario detailed breakdown
  - Capital adequacy waterfall
  - Loss attribution by asset type and sector
  - Multi-scenario comparison heatmap (text-based)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


def _fmt_usd(val, scale: float = 1e6, suffix: str = "M") -> str:
    """Format a USD value for display."""
    if isinstance(val, float):
        return f"${val / scale:,.1f}{suffix}"
    return str(val)


def _fmt_pct(val) -> str:
    if isinstance(val, float):
        return f"{val:.2%}"
    return str(val)


def _table(df: pd.DataFrame, fmt: str = "simple") -> str:
    if _HAS_TABULATE:
        return tabulate(df, headers="keys", tablefmt=fmt, showindex=False)
    return df.to_string(index=False)


class StressTestReport:
    """
    Generates human-readable stress test reports from SimulationResult objects.

    Parameters
    ----------
    simulation_result : SimulationResult
        Output from StressTestSimulator.run_all_scenarios().
    portfolio : Portfolio
        The portfolio that was stress-tested.
    scale : float
        Dollar scale for reporting (default 1e6 = millions).
    scale_suffix : str
        Unit label (default "M").
    """

    def __init__(self, simulation_result, portfolio, scale: float = 1e6, scale_suffix: str = "M") -> None:
        self.result = simulation_result
        self.portfolio = portfolio
        self.scale = scale
        self.suffix = scale_suffix

    def _usd(self, val: float) -> str:
        return _fmt_usd(val, self.scale, self.suffix)

    # ------------------------------------------------------------------
    # Portfolio overview
    # ------------------------------------------------------------------

    def portfolio_overview(self) -> str:
        p = self.portfolio
        lines = [
            "=" * 70,
            f"  PORTFOLIO OVERVIEW: {p.name}",
            "=" * 70,
            f"  Total Market Value:     {self._usd(p.total_market_value)}",
            f"  Total Notional:         {self._usd(p.total_notional)}",
            f"  Tier-1 Capital:         {self._usd(p.tier1_capital)}",
            f"  Tier-2 Capital:         {self._usd(p.tier2_capital)}",
            f"  Total Liabilities:      {self._usd(p.total_liabilities)}",
            f"  Leverage Ratio (assets/T1): {p.leverage_ratio:.1f}x",
            f"  Debt-to-Equity:         {p.debt_to_equity:.1f}x",
            f"  RWA:                    {self._usd(p.risk_weighted_assets())}",
            f"  CET1 Ratio:             {_fmt_pct(p.cet1_ratio())}",
            "",
            "  Asset Type Breakdown:",
        ]
        conc = p.concentration_by_type()
        for atype, pct in conc.items():
            lines.append(f"    {atype:<20} {_fmt_pct(pct)}")
        lines.append("")
        lines.append("  Sector Breakdown:")
        for sector, pct in p.concentration_by_sector().items():
            lines.append(f"    {sector:<20} {_fmt_pct(pct)}")
        lines.append(f"  HHI Concentration Index: {p.herfindahl_index():.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scenario executive summary
    # ------------------------------------------------------------------

    def scenario_summary(self, scenario_key: str) -> str:
        res = self.result.scenario_results[scenario_key]
        sc = res.scenario
        mv = self.portfolio.total_market_value

        lines = [
            "",
            "─" * 70,
            f"  SCENARIO: {sc.name}  [{sc.severity.upper()}]",
            f"  {sc.description}",
            "─" * 70,
            "  Macro Shocks:",
            f"    GDP Shock:              {sc.gdp_shock_pct:.1%}",
            f"    Unemployment Rise:      +{sc.unemployment_rise_pp:.1f}pp",
            f"    Equity Market Decline:  {sc.equity_market_decline:.1%}",
            f"    Yield Curve Shift:      {sc.yield_curve_shift_bps:+.0f}bps",
            f"    Real Estate Decline:    {sc.real_estate_decline:.1%}",
            f"    Funding Spread:         +{sc.funding_spread_bps:.0f}bps",
            f"    PD Multiplier:          {sc.pd_multiplier:.1f}x",
            "",
            "  Loss Estimates:",
            f"    Bottom-Up Loss:         {self._usd(res.bottom_up.total_loss)}  "
            f"({res.bottom_up.total_loss / mv:.1%} of portfolio)",
            f"    Top-Down Loss:          {self._usd(res.top_down.total_loss)}  "
            f"({res.top_down.total_loss / mv:.1%} of portfolio)",
            f"    Market Shock Loss:      {self._usd(res.market_shock.total_loss)}  "
            f"({res.market_shock.total_loss / mv:.1%} of portfolio)",
            f"    Credit Spread Loss:     {self._usd(res.credit_spread.total_loss)}",
            f"    MC VaR 99%:             {self._usd(res.monte_carlo.var_99)}",
            f"    MC Expected Shortfall:  {self._usd(res.monte_carlo.es_99)}",
            f"  ► Combined Loss Est.:    {self._usd(res.combined_loss_estimate)}  "
            f"({res.combined_loss_estimate / mv:.1%} of portfolio)",
            "",
            "  Capital Adequacy (Post-Stress):",
            f"    Pre-Stress CET1:        {_fmt_pct(res.leverage.pre_stress.cet1_ratio)}",
            f"    Post-Stress CET1:       {_fmt_pct(res.leverage.post_stress.cet1_ratio)}",
            f"    CET1 Change:            {res.leverage.cet1_ratio_change_bp:+.0f}bps",
            f"    Pre-Stress T1 Leverage: {_fmt_pct(res.leverage.pre_stress.tier1_leverage_ratio)}",
            f"    Post-Stress T1 Leverage:{_fmt_pct(res.leverage.post_stress.tier1_leverage_ratio)}",
            f"    Capital Depletion:      {res.leverage.capital_depletion_pct:.1%}",
            f"    Capital Shortfall:      {self._usd(res.leverage.capital_shortfall)}",
            f"    Passes DFAST:           {'✓ PASS' if res.leverage.passes_dfast else '✗ FAIL'}",
            f"    Conservation Buffer:    {'Breached' if res.leverage.buffer_exhausted else 'Intact'}",
            "─" * 70,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Multi-scenario comparison
    # ------------------------------------------------------------------

    def comparison_report(self) -> str:
        df = self.result.comparison_table()
        lines = [
            "",
            "=" * 70,
            "  MULTI-SCENARIO COMPARISON",
            "=" * 70,
        ]

        display = df.copy()
        for col in ["Bottom-Up Loss", "Top-Down Loss", "Market Shock Loss",
                    "MC VaR 99%", "MC ES 99%", "Combined Loss"]:
            display[col] = display[col].apply(self._usd)
        display["Post-Stress CET1"] = display["Post-Stress CET1"].apply(_fmt_pct)

        lines.append(_table(display))
        lines.append("")
        worst = self.result.worst_case_scenario()
        lines.append(f"  ► Worst-case scenario: {worst}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> str:
        sections = [
            self.portfolio_overview(),
            self.comparison_report(),
        ]
        for key in self.result.scenario_results:
            sections.append(self.scenario_summary(key))
        return "\n".join(sections)

    def print_report(self) -> None:
        print(self.full_report())

    def to_excel(self, filepath: str) -> None:
        """Export all results to an Excel workbook."""
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Portfolio summary
            self.portfolio.summary().to_excel(writer, sheet_name="Portfolio", index=False)

            # Comparison table
            self.result.comparison_table().to_excel(
                writer, sheet_name="Scenario Comparison", index=False
            )

            # Per-scenario detail sheets
            for key, res in self.result.scenario_results.items():
                sheet_name = key[:31]   # Excel sheet name limit

                detail_rows = []
                # Bottom-up
                for row in res.bottom_up.summary().to_dict("records"):
                    row["Module"] = "Bottom-Up"
                    detail_rows.append(row)
                # Top-down
                for row in res.top_down.summary().to_dict("records"):
                    row["Module"] = "Top-Down"
                    detail_rows.append(row)
                # Capital
                for row in res.leverage.summary().to_dict("records"):
                    row["Module"] = "Capital"
                    detail_rows.append(row)

                pd.DataFrame(detail_rows).to_excel(writer, sheet_name=sheet_name, index=False)

                # Monte Carlo loss distribution per scenario
                mc_sheet = f"{key[:25]}_MC"
                res.monte_carlo.summary().to_excel(writer, sheet_name=mc_sheet, index=False)
