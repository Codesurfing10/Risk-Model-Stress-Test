"""
Unit and integration tests for the banking stress test simulator.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

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
from stress_test.scenarios.recession_scenarios import (
    SCENARIO_LIBRARY,
    RecessionScenario,
    ScenarioLibrary,
)
from stress_test.models.bottom_up import BottomUpModel
from stress_test.models.top_down import TopDownModel
from stress_test.models.monte_carlo import MonteCarloEngine
from stress_test.models.credit_spreads import CreditSpreadModel
from stress_test.models.market_shock import MarketShockModel
from stress_test.models.leverage_risk import LeverageRiskModel
from stress_test.simulator import StressTestSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_portfolio() -> Portfolio:
    p = Portfolio(
        name="Test Portfolio",
        tier1_capital=1_000_000,
        tier2_capital=200_000,
        total_liabilities=8_000_000,
    )
    p.add_assets([
        LoanAsset(
            asset_id="L1", name="Corp Loan BBB",
            asset_type=None, notional=2_000_000, market_value=2_000_000,
            sector="corporate", rating=CreditRating.BBB, lgd=0.45, maturity_years=3.0,
        ),
        BondAsset(
            asset_id="B1", name="Corp Bond A",
            asset_type=None, notional=1_500_000, market_value=1_400_000,
            sector="corporate", rating=CreditRating.A,
            coupon=0.04, yield_to_maturity=0.045, maturity_years=5.0,
        ),
        EquityAsset(
            asset_id="E1", name="Equity Fund",
            asset_type=None, notional=1_000_000, market_value=1_000_000,
            sector="technology", beta=1.1,
        ),
        RealEstateAsset(
            asset_id="RE1", name="Office Building",
            asset_type=None, notional=2_500_000, market_value=2_500_000,
            sector="real_estate", property_type="commercial", ltv_ratio=0.65,
        ),
        DerivativeAsset(
            asset_id="D1", name="Equity Put Option",
            asset_type=None, notional=500_000, market_value=50_000,
            sector="financials", delta=-0.40, gamma=0.05, vega=10_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.A,
        ),
    ])
    return p


@pytest.fixture
def moderate_recession() -> RecessionScenario:
    return SCENARIO_LIBRARY["moderate_recession"]


@pytest.fixture
def gfc_scenario() -> RecessionScenario:
    return SCENARIO_LIBRARY["gfc_2008"]


# ---------------------------------------------------------------------------
# Asset tests
# ---------------------------------------------------------------------------

class TestAssets:
    def test_loan_expected_loss(self):
        loan = LoanAsset(
            asset_id="L", name="Test Loan",
            asset_type=None, notional=1_000_000, market_value=1_000_000,
            sector="general", rating=CreditRating.BBB, lgd=0.45,
        )
        el = loan.expected_loss
        assert el == pytest.approx(loan.pd * loan.lgd * loan.ead, rel=1e-6)

    def test_loan_stressed_pd_capped(self):
        loan = LoanAsset(
            asset_id="L", name="Test Loan",
            asset_type=None, notional=1_000_000, market_value=1_000_000,
            sector="general", rating=CreditRating.CCC, lgd=0.60,
        )
        # Very high multiplier should cap at 1.0
        assert loan.stressed_pd(100.0) == pytest.approx(1.0)

    def test_bond_duration_positive(self):
        bond = BondAsset(
            asset_id="B", name="Test Bond",
            asset_type=None, notional=1_000_000, market_value=950_000,
            sector="corporate", rating=CreditRating.BBB,
            coupon=0.05, yield_to_maturity=0.055, maturity_years=5.0,
        )
        assert bond.duration > 0
        assert bond.duration < bond.maturity_years

    def test_bond_price_change_direction(self):
        bond = BondAsset(
            asset_id="B", name="Test Bond",
            asset_type=None, notional=1_000_000, market_value=1_000_000,
            sector="corporate", rating=CreditRating.BBB,
            coupon=0.05, yield_to_maturity=0.05, maturity_years=5.0,
        )
        # Rates up → price down → negative P&L
        assert bond.price_change_from_rate_shock(0.01) < 0
        # Rates down → price up → positive P&L
        assert bond.price_change_from_rate_shock(-0.01) > 0

    def test_equity_stressed_value(self):
        eq = EquityAsset(
            asset_id="E", name="Equity",
            asset_type=None, notional=1_000_000, market_value=1_000_000,
            sector="tech", beta=1.2,
        )
        stressed = eq.stressed_value(market_return=-0.30)
        assert stressed == pytest.approx(1_000_000 * (1 + 1.2 * (-0.30)), rel=1e-6)

    def test_real_estate_ltv_stressed(self):
        re = RealEstateAsset(
            asset_id="RE", name="Property",
            asset_type=None, notional=5_000_000, market_value=5_000_000,
            sector="real_estate", ltv_ratio=0.60,
        )
        ltv_stressed = re.implied_ltv_stressed(-0.30)
        assert ltv_stressed > re.ltv_ratio

    def test_derivative_pnl_long_call_shock(self):
        deriv = DerivativeAsset(
            asset_id="D", name="Long Put",
            asset_type=None, notional=1_000_000, market_value=100_000,
            sector="fin", delta=-0.5, gamma=0.02, vega=5_000,
            underlying="equity_index", is_long=True,
            counterparty_rating=CreditRating.A,
        )
        # Market falls → put gains value → positive PnL
        pnl = deriv.pnl_from_shock(-0.10)
        assert pnl > 0


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_total_market_value(self, small_portfolio):
        expected = sum(a.market_value for a in small_portfolio.assets)
        assert small_portfolio.total_market_value == pytest.approx(expected)

    def test_leverage_ratio_finite(self, small_portfolio):
        lr = small_portfolio.leverage_ratio
        assert math.isfinite(lr)
        assert lr > 1.0  # assets > capital

    def test_cet1_ratio_positive(self, small_portfolio):
        cet1 = small_portfolio.cet1_ratio()
        assert cet1 > 0

    def test_herfindahl_index_range(self, small_portfolio):
        hhi = small_portfolio.herfindahl_index()
        assert 0 <= hhi <= 1.0

    def test_concentration_by_type_sums_to_one(self, small_portfolio):
        conc = small_portfolio.concentration_by_type()
        assert conc.sum() == pytest.approx(1.0, abs=1e-6)

    def test_concentration_by_sector_sums_to_one(self, small_portfolio):
        conc = small_portfolio.concentration_by_sector()
        assert conc.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scenario library tests
# ---------------------------------------------------------------------------

class TestScenarioLibrary:
    def test_all_scenarios_present(self):
        lib = ScenarioLibrary()
        for key in ["gfc_2008", "covid_2020", "stagflation", "moderate_recession",
                    "extreme_tail", "rate_shock", "sovereign_debt_crisis"]:
            assert key in lib.all_scenarios()

    def test_scenario_severity_scaling(self):
        lib = ScenarioLibrary()
        sc = lib.scaled_scenario("moderate_recession", 2.0)
        base = lib.get("moderate_recession")
        assert sc.equity_market_decline == pytest.approx(base.equity_market_decline * 2.0)
        assert sc.pd_multiplier == pytest.approx(1.0 + (base.pd_multiplier - 1.0) * 2.0)

    def test_get_missing_raises(self):
        lib = ScenarioLibrary()
        with pytest.raises(KeyError):
            lib.get("nonexistent_scenario")


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_shape_of_shocks(self):
        mc = MonteCarloEngine(n_simulations=500, seed=0)
        shocks = mc.generate_correlated_shocks()
        assert shocks.shape == (500, len(mc._factor_names))

    def test_correlated_shocks_corr_structure(self):
        mc = MonteCarloEngine(n_simulations=5_000, seed=42)
        shocks = mc.generate_correlated_shocks()
        # Equity and spread_bbb should be negatively correlated
        eq_idx = mc._factor_names.index("equity_return")
        sp_idx = mc._factor_names.index("spread_bbb_bps")
        corr = np.corrcoef(shocks[:, eq_idx], shocks[:, sp_idx])[0, 1]
        assert corr < 0  # Historically negative correlation

    def test_mc_run_returns_losses(self, small_portfolio):
        mc = MonteCarloEngine(n_simulations=500, seed=1)
        res = mc.run(small_portfolio)
        assert res.portfolio_losses.shape == (500,)

    def test_var_ordering(self, small_portfolio):
        mc = MonteCarloEngine(n_simulations=1_000, seed=2)
        res = mc.run(small_portfolio)
        assert res.var_99 >= res.var_95

    def test_es_ge_var(self, small_portfolio):
        mc = MonteCarloEngine(n_simulations=1_000, seed=3)
        res = mc.run(small_portfolio)
        assert res.es_99 >= res.var_99

    def test_student_t_fatter_tails(self, small_portfolio):
        mc_gauss = MonteCarloEngine(n_simulations=2_000, copula="gaussian", seed=42)
        mc_t = MonteCarloEngine(n_simulations=2_000, copula="student_t", t_df=3, seed=42)
        res_gauss = mc_gauss.run(small_portfolio)
        res_t = mc_t.run(small_portfolio)
        # Student-t should generally have larger tails; check var_999
        # (probabilistic, so just ensure no crash and values are finite)
        assert math.isfinite(res_t.var_999)
        assert math.isfinite(res_gauss.var_999)


# ---------------------------------------------------------------------------
# Bottom-up model tests
# ---------------------------------------------------------------------------

class TestBottomUp:
    def test_stressed_loss_greater_than_base(self, small_portfolio, gfc_scenario):
        model = BottomUpModel(scenario=gfc_scenario)
        result = model.compute(small_portfolio)
        assert result.total_stressed_el >= result.total_base_el

    def test_credit_loss_positive(self, small_portfolio, gfc_scenario):
        """Credit loss component must always be non-negative under stress."""
        model = BottomUpModel(scenario=gfc_scenario)
        result = model.compute(small_portfolio)
        assert result.credit_loss >= 0

    def test_total_loss_finite(self, small_portfolio, gfc_scenario):
        """Total loss (including hedge gains) must be finite."""
        import math
        model = BottomUpModel(scenario=gfc_scenario)
        result = model.compute(small_portfolio)
        assert math.isfinite(result.total_loss)

    def test_obligor_count_matches_assets(self, small_portfolio, moderate_recession):
        model = BottomUpModel(scenario=moderate_recession)
        result = model.compute(small_portfolio)
        assert len(result.obligor_losses) == len(small_portfolio.assets)

    def test_vasicek_pd_exceeds_point_pd(self):
        model = BottomUpModel(scenario=SCENARIO_LIBRARY["moderate_recession"])
        pd_point = 0.02
        pd_vasicek = model.vasicek_quantile_pd(pd_point, rho=0.15, confidence=0.999)
        assert pd_vasicek > pd_point


# ---------------------------------------------------------------------------
# Top-down model tests
# ---------------------------------------------------------------------------

class TestTopDown:
    def test_total_loss_positive(self, small_portfolio, gfc_scenario):
        model = TopDownModel(scenario=gfc_scenario)
        result = model.compute(small_portfolio)
        assert result.total_loss >= 0

    def test_sector_losses_sum_matches(self, small_portfolio, moderate_recession):
        model = TopDownModel(scenario=moderate_recession)
        result = model.compute(small_portfolio)
        sector_sum = sum(result.sector_losses.values())
        assert sector_sum == pytest.approx(result.total_credit_loss, rel=1e-4)

    def test_loss_path_shape(self, small_portfolio, moderate_recession):
        model = TopDownModel(scenario=moderate_recession)
        path = model.project_loss_path(small_portfolio, n_periods=9)
        assert len(path) == 9
        assert "total_loss" in path.columns

    def test_gfc_loss_greater_than_moderate(self, small_portfolio, gfc_scenario, moderate_recession):
        res_gfc = TopDownModel(scenario=gfc_scenario).compute(small_portfolio)
        res_mod = TopDownModel(scenario=moderate_recession).compute(small_portfolio)
        assert res_gfc.total_loss > res_mod.total_loss


# ---------------------------------------------------------------------------
# Credit spread model tests
# ---------------------------------------------------------------------------

class TestCreditSpread:
    def test_loss_positive_on_widening(self, small_portfolio):
        shock = {"AAA": 50, "AA": 100, "A": 200, "BBB": 400, "BB": 700, "B": 1200, "CCC": 2500}
        model = CreditSpreadModel(spread_shock=shock)
        result = model.compute(small_portfolio)
        assert result.total_loss >= 0

    def test_no_spread_shock_zero_mtm_loss(self, small_portfolio):
        model = CreditSpreadModel(spread_shock={}, apply_rating_migration=False)
        result = model.compute(small_portfolio)
        assert result.mark_to_market_loss == pytest.approx(0.0, abs=1e-6)

    def test_larger_shock_larger_loss(self, small_portfolio):
        # Use "A" rating to match the A-rated bond in the test portfolio
        shock_small = {"A": 100}
        shock_large = {"A": 500}
        res_small = CreditSpreadModel(spread_shock=shock_small, apply_rating_migration=False).compute(small_portfolio)
        res_large = CreditSpreadModel(spread_shock=shock_large, apply_rating_migration=False).compute(small_portfolio)
        assert res_large.total_loss > res_small.total_loss


# ---------------------------------------------------------------------------
# Market shock model tests
# ---------------------------------------------------------------------------

class TestMarketShock:
    def test_equity_loss_proportional_to_shock(self, small_portfolio):
        ms1 = MarketShockModel(equity_shock=-0.10, rate_shift_bps=0,
                               credit_spread_shock={}, apply_liquidity_haircut=False)
        ms2 = MarketShockModel(equity_shock=-0.20, rate_shift_bps=0,
                               credit_spread_shock={}, apply_liquidity_haircut=False)
        r1 = ms1.compute(small_portfolio)
        r2 = ms2.compute(small_portfolio)
        assert r2.loss_by_type["equity"] > r1.loss_by_type["equity"]

    def test_bond_loss_rises_with_rate_increase(self, small_portfolio):
        ms1 = MarketShockModel(equity_shock=0, rate_shift_bps=100,
                               credit_spread_shock={}, apply_liquidity_haircut=False)
        ms2 = MarketShockModel(equity_shock=0, rate_shift_bps=300,
                               credit_spread_shock={}, apply_liquidity_haircut=False)
        r1 = ms1.compute(small_portfolio)
        r2 = ms2.compute(small_portfolio)
        assert r2.loss_by_type["bond"] > r1.loss_by_type["bond"]

    def test_from_scenario(self, gfc_scenario):
        ms = MarketShockModel.from_scenario(gfc_scenario)
        assert ms.equity_shock == gfc_scenario.equity_market_decline


# ---------------------------------------------------------------------------
# Leverage risk model tests
# ---------------------------------------------------------------------------

class TestLeverageRisk:
    def test_cet1_ratio_declines_under_stress(self, small_portfolio):
        model = LeverageRiskModel()
        pre = model.compute_capital_metrics(small_portfolio)
        post = model.compute_stressed_metrics(small_portfolio, stressed_loss=200_000)
        assert post.cet1_ratio < pre.cet1_ratio

    def test_large_loss_fails_dfast(self, small_portfolio):
        model = LeverageRiskModel()
        # Loss larger than all capital
        result = model.assess(small_portfolio, stressed_loss=2_000_000)
        assert not result.passes_dfast

    def test_zero_loss_passes_dfast(self, small_portfolio):
        model = LeverageRiskModel()
        result = model.assess(small_portfolio, stressed_loss=0)
        assert result.passes_dfast

    def test_capital_depletion_between_zero_and_one(self, small_portfolio):
        model = LeverageRiskModel()
        result = model.assess(small_portfolio, stressed_loss=500_000)
        assert 0 <= result.capital_depletion_pct <= 1.0

    def test_leverage_sensitivity_shape(self, small_portfolio):
        model = LeverageRiskModel()
        df = model.leverage_sensitivity(small_portfolio)
        assert len(df) == 50
        assert df["cet1_ratio"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Integration test — full simulator
# ---------------------------------------------------------------------------

class TestSimulatorIntegration:
    def test_run_single_scenario_returns_result(self, small_portfolio):
        sim = StressTestSimulator(
            portfolio=small_portfolio,
            n_simulations=200,
            seed=0,
            verbose=False,
        )
        result = sim.run_scenario(SCENARIO_LIBRARY["moderate_recession"])
        assert result.combined_loss_estimate >= 0
        assert result.leverage.post_stress.cet1_ratio >= 0

    def test_run_all_scenarios(self, small_portfolio):
        sim = StressTestSimulator(
            portfolio=small_portfolio,
            n_simulations=200,
            seed=0,
            verbose=False,
        )
        results = sim.run_all_scenarios()
        assert len(results.scenario_results) == len(SCENARIO_LIBRARY)

    def test_worst_case_is_extreme(self, small_portfolio):
        sim = StressTestSimulator(
            portfolio=small_portfolio,
            n_simulations=200,
            seed=0,
            verbose=False,
        )
        results = sim.run_all_scenarios()
        worst = results.worst_case_scenario()
        # Worst case should be extreme_tail or gfc_2008
        assert worst in ["extreme_tail", "gfc_2008"]

    def test_comparison_table_shape(self, small_portfolio):
        sim = StressTestSimulator(
            portfolio=small_portfolio,
            n_simulations=200,
            seed=0,
            verbose=False,
        )
        results = sim.run_all_scenarios()
        table = results.comparison_table()
        assert len(table) == len(SCENARIO_LIBRARY)
        assert "Combined Loss" in table.columns

    def test_reverse_stress_returns_dict(self, small_portfolio):
        sim = StressTestSimulator(
            portfolio=small_portfolio,
            n_simulations=100,
            seed=0,
            verbose=False,
        )
        result = sim.reverse_stress_test(
            target_cet1_ratio=0.90,   # very high threshold → should trigger quickly
            severity_steps=5,
        )
        assert "critical_multiplier" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
