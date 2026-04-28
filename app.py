"""
Streamlit Web Dashboard — Banking Stress Test Simulator
========================================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stress_test.portfolio.assets import (
    BondAsset,
    CreditRating,
    DerivativeAsset,
    EquityAsset,
    LoanAsset,
    RealEstateAsset,
)
from stress_test.portfolio.portfolio import Portfolio
from stress_test.scenarios.recession_scenarios import RecessionScenario, SCENARIO_LIBRARY
from stress_test.simulator import StressTestSimulator
from stress_test.reporting.reports import StressTestReport

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Banking Stress Test Simulator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "mild": "#2ecc71",
    "moderate": "#f39c12",
    "severe": "#e74c3c",
    "extreme": "#8e44ad",
}

PASS_COLOR = "#2ecc71"
FAIL_COLOR = "#e74c3c"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_b(val: float) -> str:
    return f"${val / 1e9:.1f}B"


def _fmt_pct(val: float) -> str:
    return f"{val:.2%}"


def _severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity.lower(), "#95a5a6")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em">{severity.upper()}</span>'


# ---------------------------------------------------------------------------
# Portfolio builder — cached demo portfolio
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_demo_portfolio() -> Portfolio:
    portfolio = Portfolio(
        name="Global Systemic Bank — Consolidated Balance Sheet",
        tier1_capital=50_000_000_000,
        tier2_capital=10_000_000_000,
        total_liabilities=440_000_000_000,
    )
    loans = [
        LoanAsset(asset_id="LOAN_001", name="Large-Cap Corp A (BBB)",   asset_type=None, notional=20e9, market_value=20e9, sector="corporate",   rating=CreditRating.BBB, lgd=0.40, maturity_years=5.0,  is_secured=True),
        LoanAsset(asset_id="LOAN_002", name="Large-Cap Corp B (BB)",    asset_type=None, notional= 8e9, market_value= 8e9, sector="corporate",   rating=CreditRating.BB,  lgd=0.45, maturity_years=3.0),
        LoanAsset(asset_id="LOAN_003", name="Energy Sector Loans",      asset_type=None, notional=15e9, market_value=15e9, sector="energy",      rating=CreditRating.BBB, lgd=0.45, maturity_years=4.0,  is_secured=True),
        LoanAsset(asset_id="LOAN_004", name="Mid-Market Loans (B)",     asset_type=None, notional=10e9, market_value=10e9, sector="general",     rating=CreditRating.B,   lgd=0.55, maturity_years=3.0),
        LoanAsset(asset_id="LOAN_005", name="Technology Sector (A)",    asset_type=None, notional=12e9, market_value=12e9, sector="technology",  rating=CreditRating.A,   lgd=0.35, maturity_years=5.0),
        LoanAsset(asset_id="LOAN_006", name="Consumer Credit",          asset_type=None, notional=25e9, market_value=25e9, sector="consumer",    rating=CreditRating.BBB, lgd=0.60, maturity_years=2.0),
        LoanAsset(asset_id="LOAN_007", name="LBO Loans (B)",            asset_type=None, notional= 5e9, market_value= 5e9, sector="financials",  rating=CreditRating.B,   lgd=0.55, maturity_years=7.0),
        LoanAsset(asset_id="LOAN_008", name="Residential Mortgages",    asset_type=None, notional=55e9, market_value=55e9, sector="real_estate", rating=CreditRating.AAA, lgd=0.25, maturity_years=20.0, is_secured=True, collateral_value=75e9),
    ]
    bonds = [
        BondAsset(asset_id="BOND_001", name="IG Corporate Bonds (A)",    asset_type=None, notional=40e9,   market_value=38e9,   sector="corporate",  rating=CreditRating.A,   coupon=0.04,  yield_to_maturity=0.045, maturity_years=7.0),
        BondAsset(asset_id="BOND_002", name="HY Corporate Bonds (BB)",   asset_type=None, notional=15e9,   market_value=14e9,   sector="corporate",  rating=CreditRating.BB,  coupon=0.07,  yield_to_maturity=0.085, maturity_years=5.0),
        BondAsset(asset_id="BOND_003", name="US Treasuries (10Y)",       asset_type=None, notional=20e9,   market_value=19.5e9, sector="government", rating=CreditRating.AAA, coupon=0.038, yield_to_maturity=0.042, maturity_years=10.0, is_sovereign=True),
        BondAsset(asset_id="BOND_004", name="European Sovereign (AA)",   asset_type=None, notional=10e9,   market_value=9.8e9,  sector="government", rating=CreditRating.AA,  coupon=0.025, yield_to_maturity=0.030, maturity_years=8.0,  is_sovereign=True),
        BondAsset(asset_id="BOND_005", name="EM Sovereign (BBB)",        asset_type=None, notional= 8e9,   market_value=7.5e9,  sector="government", rating=CreditRating.BBB, coupon=0.060, yield_to_maturity=0.070, maturity_years=6.0,  is_sovereign=True),
        BondAsset(asset_id="BOND_006", name="Structured Credit (BBB)",   asset_type=None, notional=12e9,   market_value=11.2e9, sector="financials", rating=CreditRating.BBB, coupon=0.05,  yield_to_maturity=0.055, maturity_years=5.0),
    ]
    equities = [
        EquityAsset(asset_id="EQ_001", name="Global Equity Index Fund",   asset_type=None, notional=20e9, market_value=20e9, sector="technology", beta=1.10, idiosyncratic_vol=0.18),
        EquityAsset(asset_id="EQ_002", name="Financial Sector Equities",  asset_type=None, notional=15e9, market_value=15e9, sector="financials", beta=1.30, idiosyncratic_vol=0.25),
        EquityAsset(asset_id="EQ_003", name="Energy Sector Equities",     asset_type=None, notional= 8e9, market_value= 8e9, sector="energy",     beta=0.90, idiosyncratic_vol=0.30),
        EquityAsset(asset_id="EQ_004", name="EM Equity Portfolio",        asset_type=None, notional= 7e9, market_value= 7e9, sector="general",    beta=1.20, currency="EM"),
    ]
    real_estate = [
        RealEstateAsset(asset_id="RE_001", name="Office Portfolio (CBD)",         asset_type=None, notional=25e9, market_value=25e9, sector="real_estate", property_type="commercial",  ltv_ratio=0.60, cap_rate=0.055),
        RealEstateAsset(asset_id="RE_002", name="Retail & Mixed-Use Properties",  asset_type=None, notional=15e9, market_value=15e9, sector="real_estate", property_type="commercial",  ltv_ratio=0.65, cap_rate=0.06),
        RealEstateAsset(asset_id="RE_003", name="Industrial / Logistics",          asset_type=None, notional=10e9, market_value=10e9, sector="real_estate", property_type="commercial",  ltv_ratio=0.55, cap_rate=0.045),
        RealEstateAsset(asset_id="RE_004", name="Residential Development Loans",   asset_type=None, notional=10e9, market_value=10e9, sector="real_estate", property_type="residential", ltv_ratio=0.70, cap_rate=0.04),
    ]
    derivatives = [
        DerivativeAsset(asset_id="DERIV_001", name="IR Swap (Pay Fixed)",       asset_type=None, notional=50e9,   market_value=2e9,    sector="financials", delta=-8.0, gamma=0.01, vega=0.0,    underlying="interest_rate", is_long=True,  counterparty_rating=CreditRating.AA),
        DerivativeAsset(asset_id="DERIV_002", name="Equity Index Puts (Long)",  asset_type=None, notional=10e9,   market_value=0.5e9,  sector="financials", delta=-0.40, gamma=0.05, vega=200e6,  underlying="equity_index",  is_long=True,  counterparty_rating=CreditRating.A),
        DerivativeAsset(asset_id="DERIV_003", name="CDS Protection Sold (IG)",  asset_type=None, notional=15e9,   market_value=-0.3e9, sector="financials", delta=1.0,  gamma=0.0,  vega=0.0,    underlying="credit",        is_long=False, counterparty_rating=CreditRating.A),
        DerivativeAsset(asset_id="DERIV_004", name="FX Forward Hedges",         asset_type=None, notional= 8e9,   market_value=0.15e9, sector="financials", delta=0.90, gamma=0.0,  vega=0.0,    underlying="fx",            is_long=True,  counterparty_rating=CreditRating.AA),
    ]
    portfolio.add_assets(loans + bonds + equities + real_estate + derivatives)
    return portfolio


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "results": None,
        "portfolio": None,
        "n_simulations": 2000,
        "copula": "student_t",
        "selected_scenarios": list(SCENARIO_LIBRARY.keys()),
        "run_reverse": False,
        "reverse_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.sidebar.title("🏦 Stress Test Setup")

    st.sidebar.subheader("Portfolio")
    portfolio_choice = st.sidebar.radio(
        "Portfolio", ["Demo (~$360B institutional)", "Custom"],
        label_visibility="collapsed",
    )

    if portfolio_choice == "Demo (~$360B institutional)":
        portfolio = build_demo_portfolio()
    else:
        st.sidebar.markdown("**Capital Structure (USD billions)**")
        t1 = st.sidebar.number_input("Tier-1 Capital ($B)", value=50.0, min_value=0.1, step=1.0)
        t2 = st.sidebar.number_input("Tier-2 Capital ($B)", value=10.0, min_value=0.0, step=1.0)
        liab = st.sidebar.number_input("Total Liabilities ($B)", value=440.0, min_value=1.0, step=10.0)
        name = st.sidebar.text_input("Portfolio name", value="Custom Bank")
        portfolio = build_demo_portfolio()
        portfolio.name = name
        portfolio.tier1_capital = t1 * 1e9
        portfolio.tier2_capital = t2 * 1e9
        portfolio.total_liabilities = liab * 1e9

    st.session_state["portfolio"] = portfolio

    st.sidebar.divider()
    st.sidebar.subheader("Simulation Settings")
    st.session_state["n_simulations"] = st.sidebar.select_slider(
        "Monte Carlo paths", options=[500, 1000, 2000, 5000, 10000], value=2000
    )
    st.session_state["copula"] = st.sidebar.selectbox(
        "Copula", ["student_t", "gaussian"], index=0
    )

    st.sidebar.divider()
    st.sidebar.subheader("Scenarios to run")
    scenario_labels = {
        "gfc_2008":            "2008 GFC",
        "covid_2020":          "COVID-19",
        "stagflation":         "Stagflation",
        "sovereign_debt_crisis":"Sovereign Debt",
        "moderate_recession":  "Moderate Recession",
        "rate_shock":          "Rate Shock",
        "extreme_tail":        "Extreme Tail",
    }
    selected = []
    for key, label in scenario_labels.items():
        sc = SCENARIO_LIBRARY[key]
        color = SEVERITY_COLORS.get(sc.severity, "#95a5a6")
        checked = st.sidebar.checkbox(
            f"{label}  ·  *{sc.severity}*", value=True, key=f"sc_{key}"
        )
        if checked:
            selected.append(key)
    st.session_state["selected_scenarios"] = selected

    st.sidebar.divider()
    run_clicked = st.sidebar.button("▶  Run Stress Tests", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.session_state["run_reverse"] = st.sidebar.checkbox(
        "Also run Reverse Stress Test", value=False
    )

    return run_clicked


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

def run_simulation():
    portfolio = st.session_state["portfolio"]
    scenarios = st.session_state["selected_scenarios"]

    if not scenarios:
        st.error("Select at least one scenario.")
        return

    with st.spinner("Running stress test simulation -- this may take 15-30 seconds..."):
        simulator = StressTestSimulator(
            portfolio=portfolio,
            n_simulations=st.session_state["n_simulations"],
            copula=st.session_state["copula"],
            t_df=5,
            seed=42,
            is_gsib=True,
            verbose=False,
        )
        results = simulator.run_all_scenarios(scenario_names=scenarios)
        st.session_state["results"] = results

        if st.session_state.get("run_reverse"):
            with st.spinner("Running reverse stress test..."):
                st.session_state["reverse_result"] = simulator.reverse_stress_test(
                    target_cet1_ratio=0.045,
                    base_scenario_name="moderate_recession" if "moderate_recession" in scenarios else scenarios[0],
                    severity_steps=25,
                )

    st.success("Simulation complete!")


# ---------------------------------------------------------------------------
# Tab: Portfolio Overview
# ---------------------------------------------------------------------------

def tab_portfolio_overview():
    portfolio = st.session_state["portfolio"]
    if portfolio is None:
        st.info("Configure the portfolio in the sidebar.")
        return

    st.header(f"📊 Portfolio: {portfolio.name}")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Market Value", _fmt_b(portfolio.total_market_value))
    col2.metric("Tier-1 Capital",     _fmt_b(portfolio.tier1_capital))
    col3.metric("CET1 Ratio",         _fmt_pct(portfolio.cet1_ratio()))
    col4.metric("Leverage Ratio",     f"{portfolio.leverage_ratio:.1f}x")
    col5.metric("HHI Concentration",  f"{portfolio.herfindahl_index():.4f}")

    st.divider()

    left, right = st.columns(2)

    # Asset type pie
    type_conc = portfolio.concentration_by_type()
    fig_type = px.pie(
        values=type_conc.values,
        names=type_conc.index,
        title="Asset Type Breakdown",
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.35,
    )
    fig_type.update_traces(textinfo="percent+label")
    left.plotly_chart(fig_type, use_container_width=True)

    # Sector bar
    sector_conc = portfolio.concentration_by_sector()
    fig_sector = px.bar(
        x=sector_conc.values * 100,
        y=sector_conc.index,
        orientation="h",
        title="Sector Concentration (%)",
        labels={"x": "% of portfolio", "y": "Sector"},
        color=sector_conc.values,
        color_continuous_scale="Blues",
    )
    fig_sector.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    right.plotly_chart(fig_sector, use_container_width=True)

    # Asset table
    st.subheader("Asset Detail")
    df = portfolio.summary()
    df["market_value"] = df["market_value"].apply(lambda x: f"${x/1e9:.2f}B")
    df["notional"]     = df["notional"].apply(lambda x: f"${x/1e9:.2f}B")
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Scenario Results
# ---------------------------------------------------------------------------

def tab_results():
    results = st.session_state.get("results")
    portfolio = st.session_state.get("portfolio")

    if results is None:
        st.info("Run the stress tests first (click **▶ Run Stress Tests** in the sidebar).")
        return

    report = StressTestReport(results, portfolio, scale=1e9, scale_suffix="B")
    comp = results.comparison_table()

    # ------------------------------------------------------------------
    # KPI strip — worst case
    # ------------------------------------------------------------------
    worst = results.worst_case_scenario()
    worst_res = results.scenario_results[worst]
    st.header("🔬 Stress Test Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Worst Scenario",        SCENARIO_LIBRARY[worst].name if worst in SCENARIO_LIBRARY else worst)
    col2.metric("Worst-Case Loss",       _fmt_b(worst_res.combined_loss_estimate))
    col3.metric("Post-Stress CET1",      _fmt_pct(worst_res.leverage.post_stress.cet1_ratio),
                delta=_fmt_pct(worst_res.leverage.post_stress.cet1_ratio - worst_res.leverage.pre_stress.cet1_ratio))
    col4.metric("DFAST", "PASS" if worst_res.leverage.passes_dfast else "FAIL")

    st.divider()

    # ------------------------------------------------------------------
    # Multi-scenario comparison chart
    # ------------------------------------------------------------------
    st.subheader("Multi-Scenario Loss Comparison")

    loss_df = comp[["Scenario", "Bottom-Up Loss", "Top-Down Loss", "Market Shock Loss", "Combined Loss"]].copy()
    for col in ["Bottom-Up Loss", "Top-Down Loss", "Market Shock Loss", "Combined Loss"]:
        loss_df[col] = loss_df[col] / 1e9

    fig_compare = go.Figure()
    bar_cols = {"Bottom-Up Loss": "#3498db", "Top-Down Loss": "#e67e22", "Market Shock Loss": "#9b59b6"}
    for col, color in bar_cols.items():
        fig_compare.add_trace(go.Bar(name=col, x=loss_df["Scenario"], y=loss_df[col], marker_color=color))
    fig_compare.add_trace(go.Scatter(
        name="Combined Loss",
        x=loss_df["Scenario"], y=loss_df["Combined Loss"],
        mode="markers+lines", marker=dict(size=10, color="#e74c3c", symbol="diamond"),
        line=dict(color="#e74c3c", dash="dot"),
    ))
    fig_compare.update_layout(
        barmode="group",
        yaxis_title="Loss ($B)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ------------------------------------------------------------------
    # CET1 waterfall
    # ------------------------------------------------------------------
    st.subheader("Post-Stress CET1 Ratio by Scenario")
    cet1_df = comp[["Scenario", "Post-Stress CET1"]].copy()
    cet1_df["Post-Stress CET1 %"] = cet1_df["Post-Stress CET1"] * 100
    cet1_df["Pass"] = cet1_df["Post-Stress CET1"] >= 0.045
    cet1_df["Color"] = cet1_df["Pass"].map({True: PASS_COLOR, False: FAIL_COLOR})

    fig_cet1 = go.Figure()
    fig_cet1.add_trace(go.Bar(
        x=cet1_df["Scenario"],
        y=cet1_df["Post-Stress CET1 %"],
        marker_color=cet1_df["Color"].tolist(),
        text=cet1_df["Post-Stress CET1 %"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside",
    ))
    fig_cet1.add_hline(y=4.5, line_dash="dash", line_color="red", annotation_text="CET1 minimum 4.5%")
    fig_cet1.add_hline(y=7.0, line_dash="dot",  line_color="orange", annotation_text="CET1 + buffer 7.0%")
    fig_cet1.update_layout(yaxis_title="CET1 Ratio (%)", height=380)
    st.plotly_chart(fig_cet1, use_container_width=True)

    # ------------------------------------------------------------------
    # MC VaR / ES
    # ------------------------------------------------------------------
    st.subheader("Monte Carlo VaR 99% & Expected Shortfall by Scenario")
    mc_df = comp[["Scenario", "MC VaR 99%", "MC ES 99%"]].copy()
    mc_df["MC VaR 99% ($B)"] = mc_df["MC VaR 99%"] / 1e9
    mc_df["MC ES 99% ($B)"]  = mc_df["MC ES 99%"]  / 1e9

    fig_mc = px.bar(
        mc_df, x="Scenario",
        y=["MC VaR 99% ($B)", "MC ES 99% ($B)"],
        barmode="group",
        color_discrete_map={"MC VaR 99% ($B)": "#2980b9", "MC ES 99% ($B)": "#8e44ad"},
        title="",
        labels={"value": "Loss ($B)", "variable": "Metric"},
    )
    fig_mc.update_layout(height=380)
    st.plotly_chart(fig_mc, use_container_width=True)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    st.subheader("Full Scenario Comparison Table")
    display = comp.copy()
    for col in ["Bottom-Up Loss", "Top-Down Loss", "Market Shock Loss", "MC VaR 99%", "MC ES 99%", "Combined Loss"]:
        display[col] = display[col].apply(lambda v: f"${v/1e9:.2f}B")
    display["Post-Stress CET1"] = display["Post-Stress CET1"].apply(_fmt_pct)
    display["Passes DFAST"] = display["Passes DFAST"].apply(lambda v: "✓ PASS" if v else "✗ FAIL")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Per-scenario drill-down
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📋 Per-Scenario Deep Dive")
    scenario_keys = list(results.scenario_results.keys())
    selected_key = st.selectbox("Select scenario", scenario_keys,
                                format_func=lambda k: SCENARIO_LIBRARY[k].name if k in SCENARIO_LIBRARY else k)

    res = results.scenario_results[selected_key]
    sc  = res.scenario

    # Severity badge + description
    st.markdown(
        f"**{sc.name}** &nbsp; {_severity_badge(sc.severity)} &nbsp; *{sc.description}*",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP Shock",         f"{sc.gdp_shock_pct:.1%}")
    c2.metric("Unemployment Rise", f"+{sc.unemployment_rise_pp:.1f}pp")
    c3.metric("Equity Decline",    f"{sc.equity_market_decline:.1%}")
    c4.metric("Yield Curve Shift", f"{sc.yield_curve_shift_bps:+.0f}bps")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("RE Decline",        f"{sc.real_estate_decline:.1%}")
    d2.metric("Funding Spread",    f"+{sc.funding_spread_bps:.0f}bps")
    d3.metric("PD Multiplier",     f"{sc.pd_multiplier:.1f}x")
    d4.metric("Capital Shortfall", _fmt_b(res.leverage.capital_shortfall))

    # Loss module breakdown
    module_data = {
        "Module": ["Bottom-Up", "Top-Down", "Market Shock", "Credit Spread", "Combined"],
        "Loss ($B)": [
            res.bottom_up.total_loss / 1e9,
            res.top_down.total_loss / 1e9,
            res.market_shock.total_loss / 1e9,
            res.credit_spread.total_loss / 1e9,
            res.combined_loss_estimate / 1e9,
        ],
    }
    fig_mod = px.bar(
        pd.DataFrame(module_data), x="Module", y="Loss ($B)",
        color="Module",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Loss by Module",
    )
    fig_mod.update_layout(showlegend=False, height=320)

    # Capital adequacy gauge
    post_cet1 = res.leverage.post_stress.cet1_ratio * 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=post_cet1,
        title={"text": "Post-Stress CET1 (%)"},
        gauge={
            "axis": {"range": [0, 20]},
            "steps": [
                {"range": [0, 4.5],  "color": "#e74c3c"},
                {"range": [4.5, 7.0],"color": "#f39c12"},
                {"range": [7.0, 20], "color": "#2ecc71"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.8,
                "value": 4.5,
            },
            "bar": {"color": "#2c3e50"},
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig_gauge.update_layout(height=300)

    gl, gr = st.columns(2)
    gl.plotly_chart(fig_mod, use_container_width=True)
    gr.plotly_chart(fig_gauge, use_container_width=True)

    # Credit spread shocks
    if sc.credit_spread_shock:
        spread_df = pd.DataFrame(
            {"Rating": list(sc.credit_spread_shock.keys()),
             "Shock (bps)": list(sc.credit_spread_shock.values())}
        )
        fig_spread = px.bar(
            spread_df, x="Rating", y="Shock (bps)",
            color="Shock (bps)", color_continuous_scale="Reds",
            title="Credit Spread Shocks by Rating",
        )
        fig_spread.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig_spread, use_container_width=True)

    # Export
    st.divider()
    if st.button("📥 Export Results to Excel"):
        import io
        buf = io.BytesIO()
        report.to_excel(buf)
        buf.seek(0)
        st.download_button(
            label="Download Excel report",
            data=buf,
            file_name="stress_test_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ---------------------------------------------------------------------------
# Tab: Reverse Stress Test
# ---------------------------------------------------------------------------

def tab_reverse():
    st.header("🔄 Reverse Stress Test")
    st.markdown(
        "Find the **minimum scenario severity multiplier** that breaches the CET1 minimum of **4.5%**."
    )

    rev = st.session_state.get("reverse_result")
    if rev is None:
        st.info(
            "Enable **Also run Reverse Stress Test** in the sidebar and click **▶ Run Stress Tests**."
        )
        return

    if rev["critical_multiplier"] is None:
        st.success("✅ No CET1 breach found within 5× severity range.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Critical Severity Multiplier", f"{rev['critical_multiplier']:.2f}×")
    col2.metric("Critical Loss",                _fmt_b(rev["critical_loss"]))
    col3.metric("Post-Stress CET1 at Breach",   _fmt_pct(rev["post_stress_cet1"]))

    sc = rev["scenario"]
    if sc:
        st.subheader("Breach Scenario Parameters")
        param_df = pd.DataFrame({
            "Parameter": ["GDP Shock", "Unemployment Rise", "Equity Decline",
                          "Yield Curve Shift", "RE Decline", "PD Multiplier"],
            "Value": [
                f"{sc.gdp_shock_pct:.2%}",
                f"+{sc.unemployment_rise_pp:.1f}pp",
                f"{sc.equity_market_decline:.2%}",
                f"{sc.yield_curve_shift_bps:+.0f}bps",
                f"{sc.real_estate_decline:.2%}",
                f"{sc.pd_multiplier:.1f}×",
            ],
        })
        st.table(param_df)


# ---------------------------------------------------------------------------
# Tab: About
# ---------------------------------------------------------------------------

def tab_about():
    st.header("ℹ️ About")
    st.markdown("""
### Banking Stress Test Simulator

A comprehensive stress testing framework combining **six complementary risk models**:

| Module | Description |
|---|---|
| **Bottom-Up** | Instrument-level PD/LGD/EAD credit loss + Vasicek unexpected loss |
| **Top-Down** | Macro-factor regression on sector loss rates, CECL/IFRS-9 provisioning |
| **Monte Carlo** | 13-factor correlated simulation (Gaussian or Student-t copula), VaR/ES |
| **Market Shock** | Instantaneous scenario shocks across all asset classes |
| **Credit Spreads** | Spread widening P&L, rating migration losses, credit VaR |
| **Leverage Risk** | Basel III/IV CET1, Tier-1 leverage, DFAST pass/fail, capital shortfall |

#### Capital Adequacy Framework (Basel III / DFAST)

| Ratio | Minimum |
|---|---|
| CET1 | 4.5% |
| Tier-1 | 6.0% |
| Total Capital | 8.0% |
| Tier-1 Leverage | 3.0% |
| Capital Conservation Buffer | +2.5% |

#### Running in CLI mode

```bash
python examples/run_stress_test.py
```

#### Running tests

```bash
python -m pytest tests/test_stress_test.py -v
```
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_state()
    run_clicked = render_sidebar()

    if run_clicked:
        run_simulation()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Overview",
        "🔬 Stress Test Results",
        "🔄 Reverse Stress Test",
        "ℹ️ About",
    ])

    with tab1:
        tab_portfolio_overview()

    with tab2:
        tab_results()

    with tab3:
        tab_reverse()

    with tab4:
        tab_about()


if __name__ == "__main__":
    main()
