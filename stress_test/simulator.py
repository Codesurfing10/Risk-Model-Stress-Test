"""
Main stress test simulator — orchestrates bottom-up, top-down, Monte Carlo,
credit spread, market shock, and leverage risk modules into a unified result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .models.bottom_up import BottomUpModel, BottomUpResult
from .models.credit_spreads import CreditSpreadModel, CreditSpreadResult
from .models.leverage_risk import LeverageRiskModel, LeverageStressResult
from .models.market_shock import MarketShockModel, MarketShockResult
from .models.monte_carlo import MonteCarloEngine, MonteCarloResult
from .models.top_down import TopDownModel, TopDownResult
from .portfolio.portfolio import Portfolio
from .scenarios.recession_scenarios import RecessionScenario, ScenarioLibrary


@dataclass
class ScenarioStressResult:
    """Full stress test result for a single scenario."""

    scenario: RecessionScenario
    bottom_up: BottomUpResult
    top_down: TopDownResult
    market_shock: MarketShockResult
    credit_spread: CreditSpreadResult
    monte_carlo: MonteCarloResult
    leverage: LeverageStressResult

    # Combined headline metrics
    combined_loss_estimate: float = 0.0
    elapsed_seconds: float = 0.0

    def __post_init__(self):
        # Weighted average of bottom-up and top-down as combined estimate
        self.combined_loss_estimate = (
            0.50 * self.bottom_up.total_loss
            + 0.50 * self.top_down.total_loss
        )

    def executive_summary(self) -> pd.DataFrame:
        rows = [
            {"Module": "Bottom-Up Loss", "Estimate (USD)": self.bottom_up.total_loss},
            {"Module": "Top-Down Loss", "Estimate (USD)": self.top_down.total_loss},
            {"Module": "Market Shock Loss", "Estimate (USD)": self.market_shock.total_loss},
            {"Module": "Credit Spread Loss", "Estimate (USD)": self.credit_spread.total_loss},
            {"Module": "Monte Carlo VaR 99%", "Estimate (USD)": self.monte_carlo.var_99},
            {"Module": "Monte Carlo ES 99%", "Estimate (USD)": self.monte_carlo.es_99},
            {"Module": "Combined Loss Estimate", "Estimate (USD)": self.combined_loss_estimate},
            {"Module": "Capital Shortfall", "Estimate (USD)": self.leverage.capital_shortfall},
            {"Module": "Post-Stress CET1 Ratio", "Estimate (USD)": f"{self.leverage.post_stress.cet1_ratio:.2%}"},
            {"Module": "Passes DFAST", "Estimate (USD)": str(self.leverage.passes_dfast)},
        ]
        return pd.DataFrame(rows)


@dataclass
class SimulationResult:
    """Results across all scenarios."""

    portfolio_name: str
    scenario_results: Dict[str, ScenarioStressResult] = field(default_factory=dict)
    run_timestamp: str = ""

    def worst_case_scenario(self) -> str:
        if not self.scenario_results:
            raise ValueError("No scenario results available.")
        return max(
            self.scenario_results,
            key=lambda k: self.scenario_results[k].combined_loss_estimate,
        )

    def comparison_table(self) -> pd.DataFrame:
        rows = []
        for name, res in self.scenario_results.items():
            rows.append({
                "Scenario": name,
                "Bottom-Up Loss": res.bottom_up.total_loss,
                "Top-Down Loss": res.top_down.total_loss,
                "Market Shock Loss": res.market_shock.total_loss,
                "MC VaR 99%": res.monte_carlo.var_99,
                "MC ES 99%": res.monte_carlo.es_99,
                "Combined Loss": res.combined_loss_estimate,
                "Post-Stress CET1": res.leverage.post_stress.cet1_ratio,
                "Passes DFAST": res.leverage.passes_dfast,
            })
        return pd.DataFrame(rows)


class StressTestSimulator:
    """
    Banking Stress Test Simulator.

    Combines bottom-up instrument-level analysis, top-down macro factor
    regression, Monte Carlo simulation with correlated risk factors,
    credit spread analysis, market shock scenarios, and leverage /
    capital adequacy assessment.

    Parameters
    ----------
    portfolio : Portfolio
        The institutional portfolio to stress test.
    n_simulations : int
        Number of Monte Carlo paths per scenario.
    copula : str
        Monte Carlo copula type: "gaussian" or "student_t".
    t_df : int
        Degrees of freedom for Student-t copula.
    seed : int, optional
        Random seed for reproducibility.
    is_gsib : bool
        Whether the firm is a G-SIB.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        n_simulations: int = 10_000,
        copula: str = "student_t",
        t_df: int = 5,
        seed: Optional[int] = 42,
        is_gsib: bool = True,
        verbose: bool = True,
    ) -> None:
        self.portfolio = portfolio
        self.n_simulations = n_simulations
        self.copula = copula
        self.t_df = t_df
        self.seed = seed
        self.is_gsib = is_gsib
        self.verbose = verbose
        self._scenario_library = ScenarioLibrary()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [StressTest] {msg}")

    # ------------------------------------------------------------------
    # Single scenario runner
    # ------------------------------------------------------------------

    def run_scenario(self, scenario: RecessionScenario) -> ScenarioStressResult:
        """Run all stress test modules for a single scenario."""
        t0 = time.time()
        self._log(f"Running scenario: {scenario.name}")

        # 1. Bottom-up model
        self._log("  → Bottom-up model...")
        bu_model = BottomUpModel(scenario=scenario)
        bu_result = bu_model.compute(self.portfolio)

        # 2. Top-down model
        self._log("  → Top-down macro model...")
        td_model = TopDownModel(scenario=scenario)
        td_result = td_model.compute(self.portfolio)

        # 3. Market shock
        self._log("  → Market shock model...")
        ms_model = MarketShockModel.from_scenario(scenario)
        ms_result = ms_model.compute(self.portfolio)

        # 4. Credit spread model
        self._log("  → Credit spread model...")
        cs_model = CreditSpreadModel(spread_shock=scenario.credit_spread_shock)
        cs_result = cs_model.compute(self.portfolio)

        # 5. Monte Carlo with scenario bias
        self._log(f"  → Monte Carlo ({self.n_simulations:,} simulations)...")
        mc_engine = MonteCarloEngine(
            n_simulations=self.n_simulations,
            copula=self.copula,
            t_df=self.t_df,
            seed=self.seed,
        )
        scenario_bias = MonteCarloEngine.scenario_bias_from_recession(scenario)
        mc_result = mc_engine.run(self.portfolio, scenario_bias=scenario_bias)

        # 6. Leverage / capital adequacy
        self._log("  → Leverage & capital adequacy...")
        lev_model = LeverageRiskModel(is_gsib=self.is_gsib)
        # Use combined loss from bottom-up and top-down as the stress loss input
        combined_loss = (
            0.50 * bu_result.total_loss + 0.50 * td_result.total_loss
        )
        lev_result = lev_model.assess(self.portfolio, stressed_loss=combined_loss)

        elapsed = time.time() - t0
        self._log(f"  Completed in {elapsed:.1f}s")

        return ScenarioStressResult(
            scenario=scenario,
            bottom_up=bu_result,
            top_down=td_result,
            market_shock=ms_result,
            credit_spread=cs_result,
            monte_carlo=mc_result,
            leverage=lev_result,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Multi-scenario runner
    # ------------------------------------------------------------------

    def run_all_scenarios(
        self,
        scenario_names: Optional[List[str]] = None,
    ) -> SimulationResult:
        """
        Run stress tests across all (or specified) scenarios.

        Parameters
        ----------
        scenario_names : list of str, optional
            Subset of scenario keys from ScenarioLibrary.
            Defaults to all available scenarios.

        Returns
        -------
        SimulationResult
        """
        import datetime

        if scenario_names is None:
            scenario_names = list(self._scenario_library.all_scenarios().keys())

        self._log(
            f"Starting stress test for portfolio: '{self.portfolio.name}' "
            f"| {len(scenario_names)} scenario(s)"
        )

        results: Dict[str, ScenarioStressResult] = {}
        for name in scenario_names:
            scenario = self._scenario_library.get(name)
            results[name] = self.run_scenario(scenario)

        sim_result = SimulationResult(
            portfolio_name=self.portfolio.name,
            scenario_results=results,
            run_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        self._log("All scenarios complete.")
        return sim_result

    # ------------------------------------------------------------------
    # Custom scenario
    # ------------------------------------------------------------------

    def run_custom_scenario(self, scenario: RecessionScenario) -> ScenarioStressResult:
        """Run stress test with a user-supplied custom scenario."""
        return self.run_scenario(scenario)

    # ------------------------------------------------------------------
    # Reverse stress test
    # ------------------------------------------------------------------

    def reverse_stress_test(
        self,
        target_cet1_ratio: float = 0.045,
        base_scenario_name: str = "moderate_recession",
        severity_steps: int = 20,
    ) -> Dict:
        """
        Find the minimum scenario severity that breaches the CET1 ratio target.

        Parameters
        ----------
        target_cet1_ratio : float
            The CET1 ratio threshold (e.g. 0.045 for 4.5%).
        base_scenario_name : str
            The base scenario to scale up.
        severity_steps : int
            Number of severity increments to test.

        Returns
        -------
        dict with 'critical_multiplier', 'critical_loss', 'scenario'
        """
        self._log(f"Running reverse stress test (target CET1={target_cet1_ratio:.1%})")
        multipliers = np.linspace(0.1, 5.0, severity_steps)

        lev_model = LeverageRiskModel(is_gsib=self.is_gsib)
        bu_model_base = BottomUpModel(
            scenario=self._scenario_library.get(base_scenario_name)
        )

        for mult in multipliers:
            scaled = self._scenario_library.scaled_scenario(base_scenario_name, mult)
            bu_model = BottomUpModel(scenario=scaled)
            bu_result = bu_model.compute(self.portfolio)

            td_model = TopDownModel(scenario=scaled)
            td_result = td_model.compute(self.portfolio)

            combined_loss = 0.5 * bu_result.total_loss + 0.5 * td_result.total_loss
            lev_result = lev_model.assess(self.portfolio, combined_loss)

            if lev_result.post_stress.cet1_ratio < target_cet1_ratio:
                self._log(
                    f"  Critical multiplier: {mult:.2f}x | "
                    f"CET1: {lev_result.post_stress.cet1_ratio:.2%} | "
                    f"Loss: ${combined_loss:,.0f}"
                )
                return {
                    "critical_multiplier": mult,
                    "critical_loss": combined_loss,
                    "post_stress_cet1": lev_result.post_stress.cet1_ratio,
                    "scenario": scaled,
                }

        self._log("No breach found within tested severity range.")
        return {
            "critical_multiplier": None,
            "critical_loss": None,
            "post_stress_cet1": None,
            "scenario": None,
        }
