"""
Simulated portfolios for major broker-dealers and custodians.

Institutions covered:
    - JPMorgan Chase
    - Goldman Sachs
    - Fidelity Investments
    - Bank of America
    - Credit Suisse
    - LPL Financial
    - UBS
    - Barclays

All figures are synthetic / illustrative and are calibrated to publicly
disclosed balance-sheet magnitudes (total assets, Tier-1 capital ratios,
approximate business-mix).  They are NOT actual reported positions.

Usage::

    from stress_test.portfolio.broker_dealer_portfolios import (
        build_jpmorgan_portfolio,
        build_goldman_sachs_portfolio,
        build_fidelity_portfolio,
        build_bank_of_america_portfolio,
        build_credit_suisse_portfolio,
        build_lpl_financial_portfolio,
        build_ubs_portfolio,
        build_barclays_portfolio,
        ALL_BROKER_DEALER_PORTFOLIOS,
    )
"""

from __future__ import annotations

from typing import Dict

from .assets import BondAsset, CreditRating, DerivativeAsset, EquityAsset, LoanAsset, RealEstateAsset
from .portfolio import Portfolio


# ---------------------------------------------------------------------------
# JPMorgan Chase  (~$3.9T total assets, ~$260B Tier-1)
# ---------------------------------------------------------------------------

def build_jpmorgan_portfolio() -> Portfolio:
    """Simulated consolidated balance sheet for JPMorgan Chase & Co."""
    portfolio = Portfolio(
        name="JPMorgan Chase & Co. — Simulated Portfolio",
        tier1_capital=260_000_000_000,   # $260B
        tier2_capital=40_000_000_000,    # $40B
        total_liabilities=3_550_000_000_000,  # $3.55T
    )

    loans = [
        LoanAsset(
            asset_id="JPM_LOAN_001", name="Consumer & Retail Banking Loans",
            asset_type=None, notional=500_000_000_000, market_value=500_000_000_000,
            sector="consumer", rating=CreditRating.BBB,
            lgd=0.40, maturity_years=3.0, is_secured=False,
        ),
        LoanAsset(
            asset_id="JPM_LOAN_002", name="Commercial & Industrial Loans",
            asset_type=None, notional=300_000_000_000, market_value=300_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            lgd=0.40, maturity_years=4.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="JPM_LOAN_003", name="Residential Mortgage Book",
            asset_type=None, notional=250_000_000_000, market_value=250_000_000_000,
            sector="real_estate", rating=CreditRating.AA,
            lgd=0.20, maturity_years=20.0, is_secured=True,
            collateral_value=350_000_000_000,
        ),
        LoanAsset(
            asset_id="JPM_LOAN_004", name="Commercial Real Estate Loans",
            asset_type=None, notional=150_000_000_000, market_value=150_000_000_000,
            sector="real_estate", rating=CreditRating.BBB,
            lgd=0.35, maturity_years=5.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="JPM_LOAN_005", name="Credit Card Receivables",
            asset_type=None, notional=200_000_000_000, market_value=200_000_000_000,
            sector="consumer", rating=CreditRating.BB,
            lgd=0.65, maturity_years=1.5, is_secured=False,
        ),
        LoanAsset(
            asset_id="JPM_LOAN_006", name="Leveraged & Syndicated Loans",
            asset_type=None, notional=120_000_000_000, market_value=120_000_000_000,
            sector="corporate", rating=CreditRating.B,
            lgd=0.50, maturity_years=6.0, is_secured=True,
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="JPM_BOND_001", name="US Treasuries (HTM + AFS)",
            asset_type=None, notional=350_000_000_000, market_value=340_000_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.038, yield_to_maturity=0.042, maturity_years=7.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="JPM_BOND_002", name="Agency MBS Portfolio",
            asset_type=None, notional=280_000_000_000, market_value=270_000_000_000,
            sector="financials", rating=CreditRating.AA,
            coupon=0.035, yield_to_maturity=0.040, maturity_years=8.0,
        ),
        BondAsset(
            asset_id="JPM_BOND_003", name="Investment-Grade Corporate Bonds",
            asset_type=None, notional=180_000_000_000, market_value=175_000_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.045, yield_to_maturity=0.048, maturity_years=6.0,
        ),
        BondAsset(
            asset_id="JPM_BOND_004", name="High-Yield Corporate Bonds",
            asset_type=None, notional=60_000_000_000, market_value=57_000_000_000,
            sector="corporate", rating=CreditRating.BB,
            coupon=0.072, yield_to_maturity=0.080, maturity_years=5.0,
        ),
        BondAsset(
            asset_id="JPM_BOND_005", name="Emerging Market Sovereign Bonds",
            asset_type=None, notional=40_000_000_000, market_value=38_000_000_000,
            sector="government", rating=CreditRating.BBB,
            coupon=0.060, yield_to_maturity=0.068, maturity_years=7.0,
            is_sovereign=True,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="JPM_EQ_001", name="Principal Equity Investments",
            asset_type=None, notional=50_000_000_000, market_value=50_000_000_000,
            sector="financials", beta=1.20, idiosyncratic_vol=0.22,
        ),
        EquityAsset(
            asset_id="JPM_EQ_002", name="Private Equity & Venture Holdings",
            asset_type=None, notional=30_000_000_000, market_value=30_000_000_000,
            sector="technology", beta=1.40, idiosyncratic_vol=0.35,
        ),
    ]

    real_estate = [
        RealEstateAsset(
            asset_id="JPM_RE_001", name="Commercial Real Estate Owned (REO)",
            asset_type=None, notional=40_000_000_000, market_value=40_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.55, cap_rate=0.05,
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="JPM_DERIV_001", name="Interest Rate Derivatives (Net)",
            asset_type=None, notional=800_000_000_000, market_value=15_000_000_000,
            sector="financials", delta=-6.0, gamma=0.005, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="JPM_DERIV_002", name="Credit Derivatives (CDS)",
            asset_type=None, notional=150_000_000_000, market_value=-2_000_000_000,
            sector="financials", delta=1.0, gamma=0.0, vega=0.0,
            underlying="credit", is_long=False,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="JPM_DERIV_003", name="Equity & Commodity Derivatives",
            asset_type=None, notional=80_000_000_000, market_value=3_000_000_000,
            sector="financials", delta=0.60, gamma=0.02, vega=500_000_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="JPM_DERIV_004", name="FX Derivatives",
            asset_type=None, notional=200_000_000_000, market_value=1_500_000_000,
            sector="financials", delta=0.85, gamma=0.0, vega=0.0,
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
# Goldman Sachs  (~$1.6T total assets, ~$110B Tier-1)
# ---------------------------------------------------------------------------

def build_goldman_sachs_portfolio() -> Portfolio:
    """Simulated consolidated balance sheet for The Goldman Sachs Group, Inc."""
    portfolio = Portfolio(
        name="Goldman Sachs Group, Inc. — Simulated Portfolio",
        tier1_capital=110_000_000_000,   # $110B
        tier2_capital=18_000_000_000,    # $18B
        total_liabilities=1_430_000_000_000,  # $1.43T
    )

    loans = [
        LoanAsset(
            asset_id="GS_LOAN_001", name="Corporate & Institutional Loans",
            asset_type=None, notional=120_000_000_000, market_value=120_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            lgd=0.42, maturity_years=4.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="GS_LOAN_002", name="Wealth Management Lending",
            asset_type=None, notional=50_000_000_000, market_value=50_000_000_000,
            sector="consumer", rating=CreditRating.A,
            lgd=0.30, maturity_years=3.0, is_secured=True,
            collateral_value=70_000_000_000,
        ),
        LoanAsset(
            asset_id="GS_LOAN_003", name="Leveraged Finance Originations",
            asset_type=None, notional=60_000_000_000, market_value=60_000_000_000,
            sector="corporate", rating=CreditRating.B,
            lgd=0.55, maturity_years=5.0,
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="GS_BOND_001", name="US Government Securities",
            asset_type=None, notional=200_000_000_000, market_value=195_000_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.040, yield_to_maturity=0.042, maturity_years=5.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="GS_BOND_002", name="Mortgage-Backed Securities",
            asset_type=None, notional=120_000_000_000, market_value=115_000_000_000,
            sector="financials", rating=CreditRating.AA,
            coupon=0.035, yield_to_maturity=0.039, maturity_years=7.0,
        ),
        BondAsset(
            asset_id="GS_BOND_003", name="IG Corporate Bond Trading Book",
            asset_type=None, notional=150_000_000_000, market_value=147_000_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.046, yield_to_maturity=0.049, maturity_years=5.0,
        ),
        BondAsset(
            asset_id="GS_BOND_004", name="High-Yield & Distressed Debt",
            asset_type=None, notional=80_000_000_000, market_value=74_000_000_000,
            sector="corporate", rating=CreditRating.B,
            coupon=0.080, yield_to_maturity=0.095, maturity_years=4.0,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="GS_EQ_001", name="Global Equity Market Making",
            asset_type=None, notional=100_000_000_000, market_value=100_000_000_000,
            sector="financials", beta=1.35, idiosyncratic_vol=0.28,
        ),
        EquityAsset(
            asset_id="GS_EQ_002", name="Principal Investments (PE & VC)",
            asset_type=None, notional=45_000_000_000, market_value=45_000_000_000,
            sector="technology", beta=1.50, idiosyncratic_vol=0.40,
        ),
        EquityAsset(
            asset_id="GS_EQ_003", name="Merchant Banking Equity Positions",
            asset_type=None, notional=25_000_000_000, market_value=25_000_000_000,
            sector="general", beta=1.20, idiosyncratic_vol=0.30,
        ),
    ]

    real_estate = [
        RealEstateAsset(
            asset_id="GS_RE_001", name="Real Estate Principal Investments",
            asset_type=None, notional=30_000_000_000, market_value=30_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.50, cap_rate=0.055,
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="GS_DERIV_001", name="Interest Rate Derivatives",
            asset_type=None, notional=500_000_000_000, market_value=8_000_000_000,
            sector="financials", delta=-7.0, gamma=0.004, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="GS_DERIV_002", name="Credit Default Swaps (Net)",
            asset_type=None, notional=120_000_000_000, market_value=1_500_000_000,
            sector="financials", delta=0.90, gamma=0.0, vega=0.0,
            underlying="credit", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="GS_DERIV_003", name="Equity Derivatives (Delta-One & Options)",
            asset_type=None, notional=180_000_000_000, market_value=6_000_000_000,
            sector="financials", delta=0.70, gamma=0.03, vega=1_200_000_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="GS_DERIV_004", name="Commodity Derivatives",
            asset_type=None, notional=60_000_000_000, market_value=2_000_000_000,
            sector="energy", delta=0.80, gamma=0.01, vega=0.0,
            underlying="commodity", is_long=True,
            counterparty_rating=CreditRating.BBB,
        ),
    ]

    portfolio.add_assets(loans)
    portfolio.add_assets(bonds)
    portfolio.add_assets(equities)
    portfolio.add_assets(real_estate)
    portfolio.add_assets(derivatives)
    return portfolio


# ---------------------------------------------------------------------------
# Fidelity Investments  (~$300B balance-sheet assets, custodian / broker-dealer)
# ---------------------------------------------------------------------------

def build_fidelity_portfolio() -> Portfolio:
    """
    Simulated portfolio for Fidelity Investments.

    As a privately held custodian and broker-dealer, Fidelity's own balance
    sheet is smaller than money-center banks.  The portfolio reflects its
    broker-dealer subsidiary and clearing operations.
    """
    portfolio = Portfolio(
        name="Fidelity Investments — Simulated Portfolio",
        tier1_capital=20_000_000_000,    # $20B
        tier2_capital=4_000_000_000,     # $4B
        total_liabilities=265_000_000_000,  # $265B
    )

    loans = [
        LoanAsset(
            asset_id="FID_LOAN_001", name="Margin Lending to Brokerage Clients",
            asset_type=None, notional=30_000_000_000, market_value=30_000_000_000,
            sector="consumer", rating=CreditRating.A,
            lgd=0.20, maturity_years=0.5, is_secured=True,
            collateral_value=45_000_000_000,
        ),
        LoanAsset(
            asset_id="FID_LOAN_002", name="Securities-Backed Lending",
            asset_type=None, notional=15_000_000_000, market_value=15_000_000_000,
            sector="financials", rating=CreditRating.AA,
            lgd=0.15, maturity_years=1.0, is_secured=True,
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="FID_BOND_001", name="US Treasury Portfolio (Money Market)",
            asset_type=None, notional=80_000_000_000, market_value=79_500_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.052, yield_to_maturity=0.053, maturity_years=0.5,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="FID_BOND_002", name="Agency Securities",
            asset_type=None, notional=50_000_000_000, market_value=49_200_000_000,
            sector="government", rating=CreditRating.AA,
            coupon=0.038, yield_to_maturity=0.041, maturity_years=3.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="FID_BOND_003", name="Corporate Bond Inventory",
            asset_type=None, notional=30_000_000_000, market_value=29_500_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.045, yield_to_maturity=0.048, maturity_years=4.0,
        ),
        BondAsset(
            asset_id="FID_BOND_004", name="Municipal Bonds",
            asset_type=None, notional=20_000_000_000, market_value=19_800_000_000,
            sector="government", rating=CreditRating.AA,
            coupon=0.030, yield_to_maturity=0.032, maturity_years=6.0,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="FID_EQ_001", name="Proprietary Equity Seed Positions",
            asset_type=None, notional=15_000_000_000, market_value=15_000_000_000,
            sector="technology", beta=1.10, idiosyncratic_vol=0.20,
        ),
        EquityAsset(
            asset_id="FID_EQ_002", name="Equity Clearing & Settlement Float",
            asset_type=None, notional=10_000_000_000, market_value=10_000_000_000,
            sector="financials", beta=1.00, idiosyncratic_vol=0.15,
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="FID_DERIV_001", name="Interest Rate Hedges",
            asset_type=None, notional=40_000_000_000, market_value=500_000_000,
            sector="financials", delta=-4.0, gamma=0.002, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="FID_DERIV_002", name="Equity Index Futures (Client Hedges)",
            asset_type=None, notional=20_000_000_000, market_value=200_000_000,
            sector="financials", delta=0.95, gamma=0.0, vega=0.0,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
    ]

    portfolio.add_assets(loans)
    portfolio.add_assets(bonds)
    portfolio.add_assets(equities)
    portfolio.add_assets(derivatives)
    return portfolio


# ---------------------------------------------------------------------------
# Bank of America  (~$3.3T total assets, ~$200B Tier-1)
# ---------------------------------------------------------------------------

def build_bank_of_america_portfolio() -> Portfolio:
    """Simulated consolidated balance sheet for Bank of America Corporation."""
    portfolio = Portfolio(
        name="Bank of America Corporation — Simulated Portfolio",
        tier1_capital=200_000_000_000,   # $200B
        tier2_capital=32_000_000_000,    # $32B
        total_liabilities=3_000_000_000_000,  # $3.0T
    )

    loans = [
        LoanAsset(
            asset_id="BAC_LOAN_001", name="Consumer Banking Loans",
            asset_type=None, notional=450_000_000_000, market_value=450_000_000_000,
            sector="consumer", rating=CreditRating.BBB,
            lgd=0.42, maturity_years=3.0, is_secured=False,
        ),
        LoanAsset(
            asset_id="BAC_LOAN_002", name="Residential Mortgages",
            asset_type=None, notional=220_000_000_000, market_value=220_000_000_000,
            sector="real_estate", rating=CreditRating.AA,
            lgd=0.22, maturity_years=22.0, is_secured=True,
            collateral_value=310_000_000_000,
        ),
        LoanAsset(
            asset_id="BAC_LOAN_003", name="Commercial & Industrial Loans",
            asset_type=None, notional=280_000_000_000, market_value=280_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            lgd=0.40, maturity_years=4.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="BAC_LOAN_004", name="Commercial Real Estate",
            asset_type=None, notional=90_000_000_000, market_value=90_000_000_000,
            sector="real_estate", rating=CreditRating.BBB,
            lgd=0.38, maturity_years=5.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="BAC_LOAN_005", name="Credit Card Loans",
            asset_type=None, notional=100_000_000_000, market_value=100_000_000_000,
            sector="consumer", rating=CreditRating.BB,
            lgd=0.70, maturity_years=1.5, is_secured=False,
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="BAC_BOND_001", name="US Treasuries & Agency Securities",
            asset_type=None, notional=600_000_000_000, market_value=580_000_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.036, yield_to_maturity=0.040, maturity_years=6.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="BAC_BOND_002", name="Mortgage-Backed Securities",
            asset_type=None, notional=250_000_000_000, market_value=242_000_000_000,
            sector="financials", rating=CreditRating.AA,
            coupon=0.034, yield_to_maturity=0.038, maturity_years=9.0,
        ),
        BondAsset(
            asset_id="BAC_BOND_003", name="Corporate Bonds (Investment Grade)",
            asset_type=None, notional=100_000_000_000, market_value=97_000_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.044, yield_to_maturity=0.047, maturity_years=5.0,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="BAC_EQ_001", name="Global Markets Equity Positions",
            asset_type=None, notional=40_000_000_000, market_value=40_000_000_000,
            sector="financials", beta=1.25, idiosyncratic_vol=0.24,
        ),
        EquityAsset(
            asset_id="BAC_EQ_002", name="Strategic Equity Investments",
            asset_type=None, notional=20_000_000_000, market_value=20_000_000_000,
            sector="technology", beta=1.10, idiosyncratic_vol=0.28,
        ),
    ]

    real_estate = [
        RealEstateAsset(
            asset_id="BAC_RE_001", name="Bank-Owned Commercial Properties",
            asset_type=None, notional=25_000_000_000, market_value=25_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.50, cap_rate=0.05,
        ),
        RealEstateAsset(
            asset_id="BAC_RE_002", name="Foreclosed Residential Properties",
            asset_type=None, notional=10_000_000_000, market_value=10_000_000_000,
            sector="real_estate", property_type="residential",
            ltv_ratio=0.75, cap_rate=0.04,
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="BAC_DERIV_001", name="Interest Rate Derivatives",
            asset_type=None, notional=600_000_000_000, market_value=12_000_000_000,
            sector="financials", delta=-7.0, gamma=0.004, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="BAC_DERIV_002", name="Credit Derivatives (CDS Bought)",
            asset_type=None, notional=80_000_000_000, market_value=1_200_000_000,
            sector="financials", delta=0.95, gamma=0.0, vega=0.0,
            underlying="credit", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="BAC_DERIV_003", name="FX & Commodity Derivatives",
            asset_type=None, notional=100_000_000_000, market_value=800_000_000,
            sector="financials", delta=0.80, gamma=0.0, vega=0.0,
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
# Credit Suisse  (~$580B total assets, ~$42B Tier-1; as of pre-UBS merger)
# ---------------------------------------------------------------------------

def build_credit_suisse_portfolio() -> Portfolio:
    """
    Simulated consolidated balance sheet for Credit Suisse Group AG.

    Reflects the approximate structure prior to the UBS acquisition in 2023.
    Elevated HY/distressed exposure and franchise stress are incorporated via
    lower credit ratings on a portion of the loan and trading book.
    """
    portfolio = Portfolio(
        name="Credit Suisse Group AG — Simulated Portfolio",
        tier1_capital=42_000_000_000,    # $42B
        tier2_capital=8_000_000_000,     # $8B
        total_liabilities=520_000_000_000,  # $520B
    )

    loans = [
        LoanAsset(
            asset_id="CS_LOAN_001", name="Wealth Management Lombard Loans",
            asset_type=None, notional=60_000_000_000, market_value=60_000_000_000,
            sector="consumer", rating=CreditRating.A,
            lgd=0.25, maturity_years=1.5, is_secured=True,
            collateral_value=90_000_000_000,
        ),
        LoanAsset(
            asset_id="CS_LOAN_002", name="Investment Bank Leveraged Finance",
            asset_type=None, notional=40_000_000_000, market_value=40_000_000_000,
            sector="corporate", rating=CreditRating.B,
            lgd=0.58, maturity_years=5.0,
        ),
        LoanAsset(
            asset_id="CS_LOAN_003", name="Swiss Corporate Loans",
            asset_type=None, notional=35_000_000_000, market_value=35_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            lgd=0.38, maturity_years=3.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="CS_LOAN_004", name="Distressed & Restructured Loans",
            asset_type=None, notional=15_000_000_000, market_value=12_000_000_000,
            sector="corporate", rating=CreditRating.CCC,
            lgd=0.75, maturity_years=3.0,
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="CS_BOND_001", name="Swiss Government & SNB Bills",
            asset_type=None, notional=80_000_000_000, market_value=79_500_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.010, yield_to_maturity=0.012, maturity_years=3.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="CS_BOND_002", name="European Government Bonds",
            asset_type=None, notional=50_000_000_000, market_value=48_500_000_000,
            sector="government", rating=CreditRating.AA,
            coupon=0.020, yield_to_maturity=0.025, maturity_years=6.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="CS_BOND_003", name="Investment Bank Bond Trading Book",
            asset_type=None, notional=70_000_000_000, market_value=67_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            coupon=0.055, yield_to_maturity=0.062, maturity_years=4.0,
        ),
        BondAsset(
            asset_id="CS_BOND_004", name="Structured Products (CDO/CLO)",
            asset_type=None, notional=25_000_000_000, market_value=21_000_000_000,
            sector="financials", rating=CreditRating.BB,
            coupon=0.065, yield_to_maturity=0.082, maturity_years=5.0,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="CS_EQ_001", name="Listed Equity Trading Positions",
            asset_type=None, notional=30_000_000_000, market_value=30_000_000_000,
            sector="financials", beta=1.30, idiosyncratic_vol=0.30,
        ),
        EquityAsset(
            asset_id="CS_EQ_002", name="Private Equity & Strategic Stakes",
            asset_type=None, notional=10_000_000_000, market_value=10_000_000_000,
            sector="general", beta=1.20, idiosyncratic_vol=0.38,
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="CS_DERIV_001", name="Interest Rate Swaps & Options",
            asset_type=None, notional=200_000_000_000, market_value=3_000_000_000,
            sector="financials", delta=-5.5, gamma=0.003, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="CS_DERIV_002", name="Credit Derivatives",
            asset_type=None, notional=60_000_000_000, market_value=-500_000_000,
            sector="financials", delta=1.0, gamma=0.0, vega=0.0,
            underlying="credit", is_long=False,
            counterparty_rating=CreditRating.BBB,
        ),
        DerivativeAsset(
            asset_id="CS_DERIV_003", name="Equity Derivatives (Prime Brokerage)",
            asset_type=None, notional=40_000_000_000, market_value=1_200_000_000,
            sector="financials", delta=0.65, gamma=0.025, vega=400_000_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.BBB,
        ),
    ]

    portfolio.add_assets(loans)
    portfolio.add_assets(bonds)
    portfolio.add_assets(equities)
    portfolio.add_assets(derivatives)
    return portfolio


# ---------------------------------------------------------------------------
# LPL Financial  (~$100B client assets, independent broker-dealer)
# ---------------------------------------------------------------------------

def build_lpl_financial_portfolio() -> Portfolio:
    """
    Simulated portfolio for LPL Financial Holdings Inc.

    As the largest independent broker-dealer in the US, LPL's own balance sheet
    is significantly smaller than bulge-bracket banks.  The portfolio reflects its
    clearing, custody, and capital markets operations.
    """
    portfolio = Portfolio(
        name="LPL Financial Holdings Inc. — Simulated Portfolio",
        tier1_capital=2_000_000_000,     # $2B
        tier2_capital=500_000_000,       # $0.5B
        total_liabilities=90_000_000_000,  # $90B
    )

    loans = [
        LoanAsset(
            asset_id="LPL_LOAN_001", name="Advisor & Client Margin Loans",
            asset_type=None, notional=4_000_000_000, market_value=4_000_000_000,
            sector="consumer", rating=CreditRating.BBB,
            lgd=0.25, maturity_years=0.5, is_secured=True,
            collateral_value=6_000_000_000,
        ),
        LoanAsset(
            asset_id="LPL_LOAN_002", name="Advisor Forgivable Loans",
            asset_type=None, notional=1_500_000_000, market_value=1_500_000_000,
            sector="consumer", rating=CreditRating.BB,
            lgd=0.50, maturity_years=7.0, is_secured=False,
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="LPL_BOND_001", name="Short-Term US Treasuries",
            asset_type=None, notional=8_000_000_000, market_value=7_980_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.052, yield_to_maturity=0.053, maturity_years=0.5,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="LPL_BOND_002", name="Investment-Grade Bond Inventory",
            asset_type=None, notional=5_000_000_000, market_value=4_900_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.044, yield_to_maturity=0.047, maturity_years=4.0,
        ),
        BondAsset(
            asset_id="LPL_BOND_003", name="Municipal Bond Inventory",
            asset_type=None, notional=3_000_000_000, market_value=2_950_000_000,
            sector="government", rating=CreditRating.AA,
            coupon=0.031, yield_to_maturity=0.033, maturity_years=5.0,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="LPL_EQ_001", name="Equity Inventory (Market Making)",
            asset_type=None, notional=6_000_000_000, market_value=6_000_000_000,
            sector="financials", beta=1.05, idiosyncratic_vol=0.18,
        ),
        EquityAsset(
            asset_id="LPL_EQ_002", name="Alternative Investment Fund Stakes",
            asset_type=None, notional=2_000_000_000, market_value=2_000_000_000,
            sector="general", beta=0.90, idiosyncratic_vol=0.22,
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="LPL_DERIV_001", name="Interest Rate Hedges",
            asset_type=None, notional=5_000_000_000, market_value=80_000_000,
            sector="financials", delta=-3.5, gamma=0.001, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
    ]

    portfolio.add_assets(loans)
    portfolio.add_assets(bonds)
    portfolio.add_assets(equities)
    portfolio.add_assets(derivatives)
    return portfolio


# ---------------------------------------------------------------------------
# UBS  (~$1.5T total assets, ~$60B Tier-1)
# ---------------------------------------------------------------------------

def build_ubs_portfolio() -> Portfolio:
    """Simulated consolidated balance sheet for UBS Group AG."""
    portfolio = Portfolio(
        name="UBS Group AG — Simulated Portfolio",
        tier1_capital=60_000_000_000,    # $60B
        tier2_capital=15_000_000_000,    # $15B
        total_liabilities=1_380_000_000_000,  # $1.38T
    )

    loans = [
        LoanAsset(
            asset_id="UBS_LOAN_001", name="Wealth Management Lombard & Mortgage Loans",
            asset_type=None, notional=180_000_000_000, market_value=180_000_000_000,
            sector="consumer", rating=CreditRating.A,
            lgd=0.22, maturity_years=4.0, is_secured=True,
            collateral_value=270_000_000_000,
        ),
        LoanAsset(
            asset_id="UBS_LOAN_002", name="Swiss Corporate Loans",
            asset_type=None, notional=60_000_000_000, market_value=60_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            lgd=0.36, maturity_years=3.5, is_secured=True,
        ),
        LoanAsset(
            asset_id="UBS_LOAN_003", name="Investment Bank Leveraged Finance",
            asset_type=None, notional=30_000_000_000, market_value=30_000_000_000,
            sector="corporate", rating=CreditRating.B,
            lgd=0.52, maturity_years=5.0,
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="UBS_BOND_001", name="Swiss & European Government Bonds",
            asset_type=None, notional=200_000_000_000, market_value=196_000_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.015, yield_to_maturity=0.018, maturity_years=5.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="UBS_BOND_002", name="US Agency & Treasury Holdings",
            asset_type=None, notional=150_000_000_000, market_value=146_000_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.038, yield_to_maturity=0.041, maturity_years=6.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="UBS_BOND_003", name="Investment-Grade Corporate Bonds",
            asset_type=None, notional=100_000_000_000, market_value=97_000_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.043, yield_to_maturity=0.047, maturity_years=5.0,
        ),
        BondAsset(
            asset_id="UBS_BOND_004", name="Structured Finance & ABS",
            asset_type=None, notional=40_000_000_000, market_value=37_500_000_000,
            sector="financials", rating=CreditRating.BBB,
            coupon=0.052, yield_to_maturity=0.058, maturity_years=4.0,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="UBS_EQ_001", name="Equity Trading & Market Making",
            asset_type=None, notional=80_000_000_000, market_value=80_000_000_000,
            sector="financials", beta=1.20, idiosyncratic_vol=0.24,
        ),
        EquityAsset(
            asset_id="UBS_EQ_002", name="Principal Investments (PE/Alternatives)",
            asset_type=None, notional=20_000_000_000, market_value=20_000_000_000,
            sector="general", beta=1.10, idiosyncratic_vol=0.32,
        ),
    ]

    real_estate = [
        RealEstateAsset(
            asset_id="UBS_RE_001", name="Real Estate Investment Portfolio",
            asset_type=None, notional=15_000_000_000, market_value=15_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.48, cap_rate=0.042,
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="UBS_DERIV_001", name="Interest Rate Derivatives",
            asset_type=None, notional=300_000_000_000, market_value=5_000_000_000,
            sector="financials", delta=-6.0, gamma=0.003, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="UBS_DERIV_002", name="Credit Derivatives",
            asset_type=None, notional=50_000_000_000, market_value=800_000_000,
            sector="financials", delta=0.85, gamma=0.0, vega=0.0,
            underlying="credit", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="UBS_DERIV_003", name="FX & Cross-Currency Swaps",
            asset_type=None, notional=120_000_000_000, market_value=1_200_000_000,
            sector="financials", delta=0.88, gamma=0.0, vega=0.0,
            underlying="fx", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="UBS_DERIV_004", name="Equity Derivatives",
            asset_type=None, notional=60_000_000_000, market_value=2_000_000_000,
            sector="financials", delta=0.60, gamma=0.02, vega=600_000_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
    ]

    portfolio.add_assets(loans)
    portfolio.add_assets(bonds)
    portfolio.add_assets(equities)
    portfolio.add_assets(real_estate)
    portfolio.add_assets(derivatives)
    return portfolio


# ---------------------------------------------------------------------------
# Barclays  (~$1.7T total assets, ~$67B Tier-1)
# ---------------------------------------------------------------------------

def build_barclays_portfolio() -> Portfolio:
    """Simulated consolidated balance sheet for Barclays PLC."""
    portfolio = Portfolio(
        name="Barclays PLC — Simulated Portfolio",
        tier1_capital=67_000_000_000,    # $67B
        tier2_capital=18_000_000_000,    # $18B
        total_liabilities=1_580_000_000_000,  # $1.58T
    )

    loans = [
        LoanAsset(
            asset_id="BARC_LOAN_001", name="UK Retail & Consumer Loans",
            asset_type=None, notional=200_000_000_000, market_value=200_000_000_000,
            sector="consumer", rating=CreditRating.BBB,
            lgd=0.42, maturity_years=3.0, is_secured=False,
            country="GB",
        ),
        LoanAsset(
            asset_id="BARC_LOAN_002", name="UK Residential Mortgages",
            asset_type=None, notional=180_000_000_000, market_value=180_000_000_000,
            sector="real_estate", rating=CreditRating.AA,
            lgd=0.18, maturity_years=18.0, is_secured=True,
            collateral_value=260_000_000_000, country="GB",
        ),
        LoanAsset(
            asset_id="BARC_LOAN_003", name="Corporate & Investment Bank Loans",
            asset_type=None, notional=150_000_000_000, market_value=150_000_000_000,
            sector="corporate", rating=CreditRating.BBB,
            lgd=0.40, maturity_years=4.0, is_secured=True,
        ),
        LoanAsset(
            asset_id="BARC_LOAN_004", name="Barclaycard Credit Card Receivables",
            asset_type=None, notional=60_000_000_000, market_value=60_000_000_000,
            sector="consumer", rating=CreditRating.BB,
            lgd=0.68, maturity_years=1.5, is_secured=False, country="GB",
        ),
    ]

    bonds = [
        BondAsset(
            asset_id="BARC_BOND_001", name="UK Gilts Portfolio",
            asset_type=None, notional=200_000_000_000, market_value=195_000_000_000,
            sector="government", rating=CreditRating.AA,
            coupon=0.040, yield_to_maturity=0.044, maturity_years=8.0,
            is_sovereign=True, country="GB",
        ),
        BondAsset(
            asset_id="BARC_BOND_002", name="US Treasuries & Agency MBS",
            asset_type=None, notional=150_000_000_000, market_value=145_000_000_000,
            sector="government", rating=CreditRating.AAA,
            coupon=0.038, yield_to_maturity=0.041, maturity_years=6.0,
            is_sovereign=True,
        ),
        BondAsset(
            asset_id="BARC_BOND_003", name="Investment-Grade Corporate Bonds",
            asset_type=None, notional=120_000_000_000, market_value=116_000_000_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.047, yield_to_maturity=0.050, maturity_years=5.0,
        ),
        BondAsset(
            asset_id="BARC_BOND_004", name="High-Yield & EM Corporate Bonds",
            asset_type=None, notional=40_000_000_000, market_value=37_000_000_000,
            sector="corporate", rating=CreditRating.BB,
            coupon=0.075, yield_to_maturity=0.085, maturity_years=4.0,
        ),
    ]

    equities = [
        EquityAsset(
            asset_id="BARC_EQ_001", name="Markets Equity Positions",
            asset_type=None, notional=50_000_000_000, market_value=50_000_000_000,
            sector="financials", beta=1.25, idiosyncratic_vol=0.26,
        ),
        EquityAsset(
            asset_id="BARC_EQ_002", name="Private Equity & Strategic Holdings",
            asset_type=None, notional=12_000_000_000, market_value=12_000_000_000,
            sector="general", beta=1.05, idiosyncratic_vol=0.30,
        ),
    ]

    real_estate = [
        RealEstateAsset(
            asset_id="BARC_RE_001", name="UK Commercial Real Estate",
            asset_type=None, notional=20_000_000_000, market_value=20_000_000_000,
            sector="real_estate", property_type="commercial",
            ltv_ratio=0.55, cap_rate=0.048, country="GB",
        ),
    ]

    derivatives = [
        DerivativeAsset(
            asset_id="BARC_DERIV_001", name="Interest Rate Derivatives",
            asset_type=None, notional=350_000_000_000, market_value=6_000_000_000,
            sector="financials", delta=-6.5, gamma=0.004, vega=0.0,
            underlying="interest_rate", is_long=True,
            counterparty_rating=CreditRating.AA,
        ),
        DerivativeAsset(
            asset_id="BARC_DERIV_002", name="Credit Default Swaps",
            asset_type=None, notional=70_000_000_000, market_value=900_000_000,
            sector="financials", delta=0.90, gamma=0.0, vega=0.0,
            underlying="credit", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="BARC_DERIV_003", name="Equity & Index Derivatives",
            asset_type=None, notional=80_000_000_000, market_value=2_500_000_000,
            sector="financials", delta=0.55, gamma=0.022, vega=700_000_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
        DerivativeAsset(
            asset_id="BARC_DERIV_004", name="FX Derivatives",
            asset_type=None, notional=100_000_000_000, market_value=1_000_000_000,
            sector="financials", delta=0.82, gamma=0.0, vega=0.0,
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
# Registry: all broker-dealer portfolios
# ---------------------------------------------------------------------------

def build_all_broker_dealer_portfolios() -> Dict[str, Portfolio]:
    """
    Return a dict of all simulated broker-dealer / custodian portfolios.

    Keys are short institution identifiers; values are Portfolio instances.
    """
    return {
        "jpmorgan": build_jpmorgan_portfolio(),
        "goldman_sachs": build_goldman_sachs_portfolio(),
        "fidelity": build_fidelity_portfolio(),
        "bank_of_america": build_bank_of_america_portfolio(),
        "credit_suisse": build_credit_suisse_portfolio(),
        "lpl_financial": build_lpl_financial_portfolio(),
        "ubs": build_ubs_portfolio(),
        "barclays": build_barclays_portfolio(),
    }


#: Convenience mapping of institution key → builder function.
ALL_BROKER_DEALER_PORTFOLIOS: Dict[str, "callable"] = {
    "jpmorgan": build_jpmorgan_portfolio,
    "goldman_sachs": build_goldman_sachs_portfolio,
    "fidelity": build_fidelity_portfolio,
    "bank_of_america": build_bank_of_america_portfolio,
    "credit_suisse": build_credit_suisse_portfolio,
    "lpl_financial": build_lpl_financial_portfolio,
    "ubs": build_ubs_portfolio,
    "barclays": build_barclays_portfolio,
}
