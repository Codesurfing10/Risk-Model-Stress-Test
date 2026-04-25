"""
End-to-end example: Banking Stress Test for a large institutional firm.

Portfolio composition (approximate $500B total market value):
  - Commercial & industrial loans: $150B
  - Residential mortgage book: $80B
  - Corporate bond portfolio: $100B
  - Listed equity portfolio: $50B
  - Commercial real estate: $60B
  - OTC derivatives book: $30B
  - Sovereign bonds: $30B

Capital structure:
  - Tier-1 capital: $50B
  - Tier-2 capital: $10B
  - Total liabilities: $440B

Run with:
    pip install -r requirements.txt
    python examples/run_stress_test.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stress_test.portfolio.assets import (
    BondAsset,
    CreditRating,
    DerivativeAsset,
    EquityAsset,
    LoanAsset,
    RealEstateAsset,
)
from stress_test.portfolio.portfolio import Portfolio
from stress_test.scenarios.recession_scenarios import RecessionScenario
from stress_test.simulator import StressTestSimulator
from stress_test.reporting.reports import StressTestReport


# ---------------------------------------------------------------------------
# Build synthetic institutional portfolio
# ---------------------------------------------------------------------------

def build_portfolio() -> Portfolio:
    portfolio = Portfolio(
        name="Global Systemic Bank — Consolidated Balance Sheet",
        tier1_capital=50_000_000_000,    # $50B
        tier2_capital=10_000_000_000,    # $10B
        total_liabilities=440_000_000_000,  # $440B
    )

    # ------------------------------------------------------------------
    # Commercial & Industrial Loans
    # ------------------------------------------------------------------
    loans = [
        LoanAsset(
            asset_id="LOAN_001", name="Large-Cap Corp A (BBB)",
            asset_type=None, notional=20_000_000_000, market_value=20_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            lgd=0.40, maturity_years=5.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="LOAN_002", name="Large-Cap Corp B (BB)",
            asset_type=None, notional=8_000_000_000, market_value=8_000_000_000,
            sector="corporate", rating=CreditRating.BB,
            lgd=0.45, maturity_years=3.0,
        ),
        LoanAsset(
            asset_id="LOAN_003", name="Energy Sector Loans (BBB)",
            asset_type=None, notional=15_000_000_000, market_value=15_000_000_000,
            sector="energy", rating=CreditRating.BBB,
            lgd=0.45, maturity_years=4.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="LOAN_004", name="Mid-Market Loans (BB/B)",
            asset_type=None, notional=10_000_000_000, market_value=10_000_000_000,
            sector="general", rating=CreditRating.B,
            lgd=0.55, maturity_years=3.0,
        ),
        LoanAsset(
            asset_id="LOAN_005", name="Technology Sector (A)",
            asset_type=None, notional=12_000_000_000, market_value=12_000_000_000,
            sector="technology", rating=CreditRating.A,
            lgd=0.35, maturity_years=5.0,
        ),
        LoanAsset(
            asset_id="LOAN_006", name="Consumer Credit Portfolio",
            asset_type=None, notional=25_000_000_000, market_value=25_000_000_000,
            sector="consumer", rating=CreditRating.BBB,
            lgd=0.60, maturity_years=2.0,
        ),
        LoanAsset(
            asset_id="LOAN_007", name="Leveraged Buyout Loans (B)",
            asset_type=None, notional=5_000_000_000, market_value=5_000_000_000,
            sector="financials", rating=CreditRating.B,
            lgd=0.55, maturity_years=7.0,
        ),
        LoanAsset(
            asset_id="LOAN_008", name="Residential Mortgages (AAA)",
            asset_type=None, notional=55_000_000_000, market_value=55_000_000_000,
            sector="real_estate", rating=CreditRating.AAA,
            lgd=0.25, maturity_years=20.0, is_secured=True,
            collateral_value=75_000_000_000,
        ),
    ]

    # ------------------------------------------------------------------
    # Corporate & Sovereign Bonds
    # ------------------------------------------------------------------
    bonds = [
        BondAsset(
            asset_id="BOND_001", name="IG Corporate Bonds Portfolio (A)",
            asset_type=None, notional=40_000_000_000, market_value=38_000_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.04, yield_to_maturity=0.045, maturity_years=7.0,
        ),
        BondAsset(
            asset_id="BOND_002", name="HY Corporate Bonds (BB)",
            asset_type=None, notional=15_000_000_000, market_value=14_000_000_000,
            sector="corporate", rating=CreditRating.BB,
            coupon=0.07, yield_to_maturity=0.085, maturity_years=5.0,
        ),
        BondAsset(
            asset_id="BOND_003", name="US Treasuries (10Y)",
            asset_type=None, notional=20_000_000_000, market_value=19_500_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.038, yield_to_maturity=0.042, maturity_years=10.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="BOND_004", name="European Sovereign Bonds (AA)",
            asset_type=None, notional=10_000_000_000, market_value=9_800_000_000,
            sector="government", rating=CreditRating.AA,
            coupon=0.025, yield_to_maturity=0.030, maturity_years=8.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="BOND_005", name="EM Sovereign Bonds (BBB)",
            asset_type=None, notional=8_000_000_000, market_value=7_500_000_000,
            sector="government", rating=CreditRating.BBB,
            coupon=0.060, yield_to_maturity=0.070, maturity_years=6.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="BOND_006", name="Structured Credit (ABS/MBS) (BBB)",
            asset_type=None, notional=12_000_000_000, market_value=11_200_000_000,
            sector="financials", rating=CreditRating.BBB,
            coupon=0.05, yield_to_maturity=0.055, maturity_years=5.0,
        ),
    ]

    # ------------------------------------------------------------------
    # Listed Equities
    # ------------------------------------------------------------------
    equities = [
        EquityAsset(
            asset_id="EQ_001", name="Global Equity Index Fund",
            asset_type=None, notional=20_000_000_000, market_value=20_000_000_000,
            sector="technology", beta=1.10, idiosyncratic_vol=0.18,
        ),
        EquityAsset(
            asset_id="EQ_002", name="Financial Sector Equities",
            asset_type=None, notional=15_000_000_000, market_value=15_000_000_000,
            sector="financials", beta=1.30, idiosyncratic_vol=0.25,
        ),
        EquityAsset(
            asset_id="EQ_003", name="Energy Sector Equities",
            asset_type=None, notional=8_000_000_000, market_value=8_000_000_000,
            sector="energy", beta=0.90, idiosyncratic_vol=0.30,
        ),
        EquityAsset(
            asset_id="EQ_004", name="EM Equity Portfolio",
            asset_type=None, notional=7_000_000_000, market_value=7_000_000_000,
            sector="general", beta=1.20, currency="EM",
        ),
    ]

    # ------------------------------------------------------------------
    # Commercial Real Estate
    # ------------------------------------------------------------------
    real_estate = [
        RealEstateAsset(
            asset_id="RE_001", name="Office Portfolio (CBD)",
            asset_type=None, notional=25_000_000_000, market_value=25_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.60, cap_rate=0.055,
        ),
        RealEstateAsset(
            asset_id="RE_002", name="Retail & Mixed-Use Properties",
            asset_type=None, notional=15_000_000_000, market_value=15_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.65, cap_rate=0.06,
        ),
        RealEstateAsset(
            asset_id="RE_003", name="Industrial / Logistics",
            asset_type=None, notional=10_000_000_000, market_value=10_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.55, cap_rate=0.045,
        ),
        RealEstateAsset(
            asset_id="RE_004", name="Residential Development Loans",
            asset_type=None, notional=10_000_000_000, market_value=10_000_000_000,
            sector="real_estate", property_type="residential",
            ltv_ratio=0.70, cap_rate=0.04,
        ),
    ]

    # ------------------------------------------------------------------
    # OTC Derivatives
    # ------------------------------------------------------------------
    derivatives = [
        DerivativeAsset(
            asset_id="DERIV_001", name="Interest Rate Swap (Pay Fixed)",
            asset_type=None, notional=50_000_000_000, market_value=2_000_000_000,
            sector="financials", delta=-8.0, gamma=0.01, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="DERIV_002", name="Equity Index Options (Long Puts)",
            asset_type=None, notional=10_000_000_000, market_value=500_000_000,
            sector="financials", delta=-0.40, gamma=0.05, vega=200_000_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="DERIV_003", name="CDS Protection Sold (IG)",
            asset_type=None, notional=15_000_000_000, market_value=-300_000_000,
            sector="financials", delta=1.0, gamma=0.0, vega=0.0,
            underlying="credit", is_long=False,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="DERIV_004", name="FX Forward Hedges",
            asset_type=None, notional=8_000_000_000, market_value=150_000_000,
            sector="financials", delta=0.90, gamma=0.0, vega=0.0,
            underlying="fx", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
    ]

    portfolio.add_assets(loans)
    portfolio.add_assets(bonds)
    portfolio.add_assets(equities)
    portfolio.add_assets(real_estate)
    portfolio.add_assets(derivatives)

    return portfolio


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  BANKING STRESS TEST SIMULATOR")
    print("  Bottom-Up + Top-Down + Monte Carlo")
    print("  Credit Spreads + Market Shock + Leverage Risk")
    print("=" * 70)

    # 1. Build portfolio
    print("\nBuilding institutional portfolio...")
    portfolio = build_portfolio()
    print(f"  Portfolio: {portfolio.name}")
    print(f"  Assets: {len(portfolio.assets)} positions")
    print(f"  Total Market Value: ${portfolio.total_market_value / 1e9:.1f}B")
    print(f"  Tier-1 Capital: ${portfolio.tier1_capital / 1e9:.1f}B")
    print(f"  Pre-Stress CET1: {portfolio.cet1_ratio():.2%}")
    print(f"  Leverage Multiple: {portfolio.leverage_ratio:.1f}x")

    # 2. Run stress tests
    print("\nInitialising stress test simulator...")
    simulator = StressTestSimulator(
        portfolio=portfolio,
        n_simulations=5_000,     # Reduced for demo speed; use 50_000+ in production
        copula="student_t",
        t_df=5,
        seed=42,
        is_gsib=True,
        verbose=True,
    )

    # Run all built-in scenarios
    print("\nRunning all stress scenarios...")
    results = simulator.run_all_scenarios()

    # 3. Print reports
    print("\n")
    report = StressTestReport(results, portfolio, scale=1e9, scale_suffix="B")
    report.print_report()

    # 4. Reverse stress test
    print("\n\nRunning Reverse Stress Test (find CET1 breach threshold)...")
    reverse = simulator.reverse_stress_test(
        target_cet1_ratio=0.045,
        base_scenario_name="moderate_recession",
        severity_steps=25,
    )
    if reverse["critical_multiplier"] is not None:
        print(f"\n  ► Breach at {reverse['critical_multiplier']:.2f}x severity")
        print(f"    Critical Loss: ${reverse['critical_loss'] / 1e9:.1f}B")
        print(f"    Post-Stress CET1: {reverse['post_stress_cet1']:.2%}")
    else:
        print("  ► No CET1 breach found within 5x severity range.")

    # 5. Custom scenario
    print("\n\nRunning Custom Scenario: 'Geopolitical Energy Crisis'...")
    custom_scenario = RecessionScenario(
        name="Geopolitical Energy Crisis",
        description="Escalating geopolitical tensions cause energy price surge and supply shock.",
        gdp_shock_pct=-0.035,
        unemployment_rise_pp=3.5,
        equity_market_decline=-0.28,
        yield_curve_shift_bps=150,
        fx_depreciation=0.12,
        credit_spread_shock={
            "AAA": 30, "AA": 60, "A": 130, "BBB": 280,
            "BB": 520, "B": 900, "CCC": 1800,
        },
        real_estate_decline=-0.12,
        oil_price_change=1.20,    # Oil surges 120%
        funding_spread_bps=100,
        pd_multiplier=2.8,
        lgd_multiplier=1.10,
        severity="severe",
    )
    custom_res = simulator.run_custom_scenario(custom_scenario)
    print(f"  Combined Loss: ${custom_res.combined_loss_estimate / 1e9:.1f}B")
    print(f"  Post-Stress CET1: {custom_res.leverage.post_stress.cet1_ratio:.2%}")
    print(f"  Passes DFAST: {custom_res.leverage.passes_dfast}")

    print("\n" + "=" * 70)
    print("  Stress test complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
