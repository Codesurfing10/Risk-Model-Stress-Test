"""
Monte Carlo simulation engine with correlated risk factors.

Uses Cholesky decomposition to generate correlated shocks across:
  - Equity returns (by sector)
  - Interest rates (parallel shift + twist)
  - Credit spreads (by rating bucket)
  - FX rates
  - Real-estate prices
  - Oil / commodity prices

Supports Gaussian and Student-t copulas to capture fat-tail dependence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------

RISK_FACTORS = [
    "equity_return",
    "rate_shift_bps",
    "spread_aaa_bps",
    "spread_aa_bps",
    "spread_a_bps",
    "spread_bbb_bps",
    "spread_bb_bps",
    "spread_b_bps",
    "spread_ccc_bps",
    "fx_change",
    "real_estate_return",
    "oil_return",
    "funding_spread_bps",
]

# Approximate annualised volatilities for each factor
FACTOR_VOLS: Dict[str, float] = {
    "equity_return": 0.20,
    "rate_shift_bps": 80.0,
    "spread_aaa_bps": 10.0,
    "spread_aa_bps": 20.0,
    "spread_a_bps": 40.0,
    "spread_bbb_bps": 80.0,
    "spread_bb_bps": 160.0,
    "spread_b_bps": 280.0,
    "spread_ccc_bps": 500.0,
    "fx_change": 0.10,
    "real_estate_return": 0.12,
    "oil_return": 0.35,
    "funding_spread_bps": 40.0,
}

# Correlation matrix (rows/cols ordered as RISK_FACTORS)
# Approximate empirical correlations
_RHO = np.array(
    [
        #  eq  rate saaa  saa   sa  sbbb  sbb   sb  sccc   fx    re   oil  fund
        [1.00,-0.20,-0.15,-0.20,-0.30,-0.50,-0.60,-0.65,-0.65,-0.25, 0.50, 0.30,-0.40],  # equity
        [-0.20,1.00, 0.10, 0.15, 0.10,-0.05,-0.10,-0.15,-0.20,-0.10,-0.20,-0.10, 0.20],  # rate shift
        [-0.15, 0.10,1.00, 0.80, 0.70, 0.55, 0.40, 0.35, 0.30, 0.10,-0.10,-0.05, 0.50],  # spread AAA
        [-0.20, 0.15,0.80, 1.00, 0.85, 0.65, 0.50, 0.45, 0.40, 0.15,-0.15,-0.05, 0.55],  # spread AA
        [-0.30, 0.10,0.70, 0.85, 1.00, 0.80, 0.65, 0.60, 0.55, 0.20,-0.20,-0.10, 0.60],  # spread A
        [-0.50,-0.05,0.55, 0.65, 0.80, 1.00, 0.85, 0.80, 0.75, 0.25,-0.30,-0.15, 0.70],  # spread BBB
        [-0.60,-0.10,0.40, 0.50, 0.65, 0.85, 1.00, 0.92, 0.88, 0.30,-0.35,-0.20, 0.75],  # spread BB
        [-0.65,-0.15,0.35, 0.45, 0.60, 0.80, 0.92, 1.00, 0.95, 0.30,-0.40,-0.25, 0.80],  # spread B
        [-0.65,-0.20,0.30, 0.40, 0.55, 0.75, 0.88, 0.95, 1.00, 0.30,-0.40,-0.25, 0.80],  # spread CCC
        [-0.25,-0.10,0.10, 0.15, 0.20, 0.25, 0.30, 0.30, 0.30, 1.00,-0.15, 0.40, 0.20],  # FX
        [0.50,-0.20,-0.10,-0.15,-0.20,-0.30,-0.35,-0.40,-0.40,-0.15, 1.00, 0.20,-0.30],  # real estate
        [0.30,-0.10,-0.05,-0.05,-0.10,-0.15,-0.20,-0.25,-0.25, 0.40, 0.20, 1.00,-0.10],  # oil
        [-0.40, 0.20,0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.80, 0.20,-0.30,-0.10, 1.00],  # funding
    ]
)


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix onto the nearest positive semi-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run."""

    n_simulations: int
    factor_names: List[str]
    factor_shocks: np.ndarray          # shape (n_simulations, n_factors)
    portfolio_losses: np.ndarray       # shape (n_simulations,)
    loss_by_type: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def var_95(self) -> float:
        return float(np.percentile(self.portfolio_losses, 95))

    @property
    def var_99(self) -> float:
        return float(np.percentile(self.portfolio_losses, 99))

    @property
    def var_999(self) -> float:
        return float(np.percentile(self.portfolio_losses, 99.9))

    @property
    def es_95(self) -> float:
        """Expected Shortfall (CVaR) at 95%."""
        threshold = self.var_95
        tail = self.portfolio_losses[self.portfolio_losses >= threshold]
        return float(tail.mean()) if len(tail) > 0 else self.var_95

    @property
    def es_99(self) -> float:
        threshold = self.var_99
        tail = self.portfolio_losses[self.portfolio_losses >= threshold]
        return float(tail.mean()) if len(tail) > 0 else self.var_99

    @property
    def mean_loss(self) -> float:
        return float(self.portfolio_losses.mean())

    @property
    def max_loss(self) -> float:
        return float(self.portfolio_losses.max())

    def percentile_loss(self, pct: float) -> float:
        return float(np.percentile(self.portfolio_losses, pct))

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Metric": "Mean Loss", "Value": self.mean_loss},
            {"Metric": "VaR 95%", "Value": self.var_95},
            {"Metric": "VaR 99%", "Value": self.var_99},
            {"Metric": "VaR 99.9%", "Value": self.var_999},
            {"Metric": "ES (CVaR) 95%", "Value": self.es_95},
            {"Metric": "ES (CVaR) 99%", "Value": self.es_99},
            {"Metric": "Max Loss", "Value": self.max_loss},
        ]
        return pd.DataFrame(rows)


class MonteCarloEngine:
    """
    Correlated Monte Carlo simulation engine for multi-asset institutional portfolios.

    Supports:
      - Gaussian copula (standard)
      - Student-t copula (fat tails)
      - Custom correlation matrix override

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo paths.
    horizon_years : float
        Time horizon in years (default 1 year).
    copula : str
        "gaussian" or "student_t".
    t_df : int
        Degrees of freedom for Student-t copula.
    correlation_matrix : np.ndarray, optional
        Override default correlation matrix.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        horizon_years: float = 1.0,
        copula: str = "gaussian",
        t_df: int = 5,
        correlation_matrix: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.horizon_years = horizon_years
        self.copula = copula
        self.t_df = t_df
        self.seed = seed

        rho = correlation_matrix if correlation_matrix is not None else _RHO
        self._corr = _nearest_psd(rho)
        self._chol = np.linalg.cholesky(self._corr)
        self._factor_names = list(RISK_FACTORS)
        self._vols = np.array([FACTOR_VOLS[f] for f in self._factor_names])

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Shock generation
    # ------------------------------------------------------------------

    def _generate_standard_normals(self) -> np.ndarray:
        """Draw n_simulations × n_factors standard normal matrix."""
        n = len(self._factor_names)
        z = self._rng.standard_normal((self.n_simulations, n))
        return z

    def generate_correlated_shocks(self) -> np.ndarray:
        """
        Return correlated factor shocks scaled by volatility and horizon.

        Returns
        -------
        np.ndarray
            Shape (n_simulations, n_factors). Each column corresponds to a
            risk factor in RISK_FACTORS order.
        """
        n = len(self._factor_names)
        z = self._generate_standard_normals()

        if self.copula == "student_t":
            chi2 = self._rng.chisquare(self.t_df, size=self.n_simulations)
            scale = np.sqrt(self.t_df / chi2)[:, np.newaxis]
            z = z * scale

        # Apply Cholesky to induce correlation
        correlated_z = z @ self._chol.T                   # (n_sim, n_factors)

        # Scale by vol × sqrt(horizon)
        sqrt_h = np.sqrt(self.horizon_years)
        shocks = correlated_z * self._vols[np.newaxis, :] * sqrt_h
        return shocks

    def factor_shock_dataframe(self) -> pd.DataFrame:
        shocks = self.generate_correlated_shocks()
        return pd.DataFrame(shocks, columns=self._factor_names)

    # ------------------------------------------------------------------
    # Portfolio loss computation
    # ------------------------------------------------------------------

    def run(self, portfolio, scenario_bias: Optional[Dict[str, float]] = None) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on a Portfolio object.

        Parameters
        ----------
        portfolio : Portfolio
            The institutional portfolio to stress-test.
        scenario_bias : dict, optional
            Mean shift applied to factor shocks to condition on a scenario
            (e.g. from a RecessionScenario). Keys are factor names.

        Returns
        -------
        MonteCarloResult
        """
        from ..portfolio.assets import (
            LoanAsset,
            BondAsset,
            EquityAsset,
            DerivativeAsset,
            RealEstateAsset,
        )

        shocks = self.generate_correlated_shocks()

        # Apply scenario mean bias
        if scenario_bias:
            idx_map = {f: i for i, f in enumerate(self._factor_names)}
            for factor, bias in scenario_bias.items():
                if factor in idx_map:
                    shocks[:, idx_map[factor]] += bias

        factor_idx = {f: i for i, f in enumerate(self._factor_names)}

        # Pre-extract factor paths
        eq_ret = shocks[:, factor_idx["equity_return"]]
        rate_shift = shocks[:, factor_idx["rate_shift_bps"]]
        fx_chg = shocks[:, factor_idx["fx_change"]]
        re_ret = shocks[:, factor_idx["real_estate_return"]]

        spread_cols = {
            "AAA": shocks[:, factor_idx["spread_aaa_bps"]],
            "AA":  shocks[:, factor_idx["spread_aa_bps"]],
            "A":   shocks[:, factor_idx["spread_a_bps"]],
            "BBB": shocks[:, factor_idx["spread_bbb_bps"]],
            "BB":  shocks[:, factor_idx["spread_bb_bps"]],
            "B":   shocks[:, factor_idx["spread_b_bps"]],
            "CCC": shocks[:, factor_idx["spread_ccc_bps"]],
        }
        funding_sp = shocks[:, factor_idx["funding_spread_bps"]]

        # Compute losses per asset type (loss = positive = bad)
        n = self.n_simulations
        loan_losses   = np.zeros(n)
        bond_losses   = np.zeros(n)
        equity_losses = np.zeros(n)
        deriv_losses  = np.zeros(n)
        re_losses     = np.zeros(n)

        for asset in portfolio.assets:
            if isinstance(asset, LoanAsset):
                # Stressed PD driven by equity-return (proxy for macro cycle)
                macro_mult = 1.0 + np.maximum(-eq_ret / 0.20, 0) * 2.0
                stressed_pd = np.minimum(asset.pd * macro_mult, 1.0)
                stressed_lgd = asset.lgd * (1.0 + np.maximum(-eq_ret / 0.20, 0) * 0.15)
                el = stressed_pd * stressed_lgd * asset.ead
                loan_losses += el

            elif isinstance(asset, BondAsset):
                rating_key = asset.rating.value
                sp = spread_cols.get(rating_key, spread_cols["BBB"])
                # Duration price change + spread impact
                dy = (rate_shift + sp) / 10_000.0
                dp = -asset.duration * dy + 0.5 * asset.convexity * dy ** 2
                loss = -dp * asset.market_value  # loss = negative return
                bond_losses += loss

            elif isinstance(asset, EquityAsset):
                total_ret = asset.beta * eq_ret
                loss = -total_ret * asset.market_value
                equity_losses += loss

            elif isinstance(asset, DerivativeAsset):
                underlying = asset.underlying.lower()
                if underlying in ("interest_rate", "rate"):
                    shock_pct = rate_shift / 10_000.0
                elif underlying == "fx":
                    shock_pct = fx_chg
                elif underlying == "credit":
                    shock_pct = spread_cols.get("BBB", np.zeros(n)) / 10_000.0 * 5.0
                else:
                    shock_pct = eq_ret

                pnl_pct = asset.delta * shock_pct
                if hasattr(asset, "gamma"):
                    pnl_pct = pnl_pct + 0.5 * asset.gamma * shock_pct ** 2
                pnl = pnl_pct * asset.notional
                if hasattr(asset, "vega"):
                    pnl = pnl + asset.vega * np.abs(eq_ret) * 0.5
                if not asset.is_long:
                    pnl = -pnl
                deriv_losses += -pnl

            elif isinstance(asset, RealEstateAsset):
                loss = -re_ret * asset.market_value
                re_losses += loss

        portfolio_losses = loan_losses + bond_losses + equity_losses + deriv_losses + re_losses

        return MonteCarloResult(
            n_simulations=n,
            factor_names=self._factor_names,
            factor_shocks=shocks,
            portfolio_losses=portfolio_losses,
            loss_by_type={
                "loan": loan_losses,
                "bond": bond_losses,
                "equity": equity_losses,
                "derivative": deriv_losses,
                "real_estate": re_losses,
            },
        )

    @staticmethod
    def scenario_bias_from_recession(scenario) -> Dict[str, float]:
        """Convert a RecessionScenario into a Monte Carlo mean-shift dict."""
        bias: Dict[str, float] = {
            "equity_return": scenario.equity_market_decline,
            "rate_shift_bps": scenario.yield_curve_shift_bps,
            "fx_change": scenario.fx_depreciation,
            "real_estate_return": scenario.real_estate_decline,
            "oil_return": scenario.oil_price_change,
            "funding_spread_bps": scenario.funding_spread_bps,
        }
        # Credit spread shocks
        rating_map = {
            "AAA": "spread_aaa_bps",
            "AA":  "spread_aa_bps",
            "A":   "spread_a_bps",
            "BBB": "spread_bbb_bps",
            "BB":  "spread_bb_bps",
            "B":   "spread_b_bps",
            "CCC": "spread_ccc_bps",
        }
        for rating, factor in rating_map.items():
            val = scenario.credit_spread_shock.get(rating, 0.0)
            bias[factor] = val
        return bias
