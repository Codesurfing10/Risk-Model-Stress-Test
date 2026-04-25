"""
Credit spread model for banking stress tests.

Computes mark-to-market P&L impact from credit spread widening on:
  - Corporate and sovereign bonds
  - Loan book (spread-to-price approximation)
  - CDS positions

Implements:
  1. Scenario-based instantaneous spread widening
  2. Rating-migration loss (downgrade probability × spread widening)
  3. Credit VaR from spread distribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..portfolio.assets import BondAsset, CreditRating, LoanAsset


# ---------------------------------------------------------------------------
# Rating transition matrix (Moody's long-run average, annual, simplified)
# Rows = from-rating, Cols = to-rating (AAA, AA, A, BBB, BB, B, CCC, D)
# ---------------------------------------------------------------------------
_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]

_TRANSITION_MATRIX = np.array(
    [
        # AAA    AA      A     BBB    BB      B     CCC      D
        [0.9066, 0.0826, 0.0066, 0.0006, 0.0014, 0.0006, 0.0000, 0.0016],  # AAA
        [0.0064, 0.9068, 0.0780, 0.0064, 0.0006, 0.0010, 0.0000, 0.0008],  # AA
        [0.0007, 0.0225, 0.9119, 0.0530, 0.0067, 0.0024, 0.0001, 0.0027],  # A
        [0.0004, 0.0026, 0.0591, 0.8693, 0.0530, 0.0117, 0.0012, 0.0027],  # BBB
        [0.0003, 0.0008, 0.0062, 0.0773, 0.8053, 0.0846, 0.0100, 0.0155],  # BB
        [0.0000, 0.0008, 0.0024, 0.0043, 0.0649, 0.8346, 0.0466, 0.0464],  # B
        [0.0010, 0.0000, 0.0029, 0.0059, 0.0147, 0.1059, 0.5800, 0.2896],  # CCC
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # D
    ]
)

# Base credit spreads (bps) per rating
_BASE_SPREADS: Dict[str, float] = {r: CreditRating[r].spread_base_bps for r in _RATINGS[:-1]}
_BASE_SPREADS["D"] = 3000.0


@dataclass
class CreditSpreadResult:
    """Results from credit spread stress analysis."""

    total_loss: float
    loss_by_asset: Dict[str, float]
    rating_migration_loss: float
    mark_to_market_loss: float

    def summary(self) -> pd.DataFrame:
        rows = [
            {"Component": "Mark-to-Market Spread Loss", "Loss (USD)": self.mark_to_market_loss},
            {"Component": "Rating Migration Loss", "Loss (USD)": self.rating_migration_loss},
            {"Component": "Total Credit Spread Loss", "Loss (USD)": self.total_loss},
        ]
        return pd.DataFrame(rows)


class CreditSpreadModel:
    """
    Models credit spread risk for bonds and loans in an institutional portfolio.

    Parameters
    ----------
    spread_shock : dict
        Mapping from rating string to spread shock in bps.
        If None, uses the scenario's credit_spread_shock.
    apply_rating_migration : bool
        Whether to include losses from rating downgrades.
    migration_horizon_years : float
        Horizon for rating migration calculation.
    stressed_transition : float
        Multiplier applied to off-diagonal transition probabilities (>1 = more downgrades).
    """

    def __init__(
        self,
        spread_shock: Optional[Dict[str, float]] = None,
        apply_rating_migration: bool = True,
        migration_horizon_years: float = 1.0,
        stressed_transition: float = 2.0,
    ) -> None:
        self.spread_shock = spread_shock or {}
        self.apply_rating_migration = apply_rating_migration
        self.migration_horizon_years = migration_horizon_years
        self.stressed_transition = stressed_transition

    # ------------------------------------------------------------------
    # Transition matrix under stress
    # ------------------------------------------------------------------

    def _stressed_transition_matrix(self) -> np.ndarray:
        """
        Generate a stressed transition matrix by amplifying off-diagonal
        (downgrade) probabilities.
        """
        tm = _TRANSITION_MATRIX.copy()
        n = len(_RATINGS)
        stressed = np.zeros_like(tm)
        for i in range(n):
            for j in range(n):
                if j <= i:   # on-diagonal or improvement: no stress
                    stressed[i, j] = tm[i, j]
                else:        # downgrade: amplify
                    stressed[i, j] = tm[i, j] * self.stressed_transition
            # Re-normalise row
            row_sum = stressed[i].sum()
            if row_sum > 0:
                stressed[i] /= row_sum
        return stressed

    # ------------------------------------------------------------------
    # Bond P&L from spread widening
    # ------------------------------------------------------------------

    def bond_spread_loss(self, bond: BondAsset, shock_override: Optional[float] = None) -> float:
        """
        Mark-to-market loss for a single bond from credit spread widening.

        Parameters
        ----------
        bond : BondAsset
        shock_override : float, optional
            Spread shock in bps; if None uses self.spread_shock.

        Returns
        -------
        float
            Loss (positive = adverse).
        """
        if shock_override is not None:
            delta_spread = shock_override
        else:
            delta_spread = self.spread_shock.get(bond.rating.value, 0.0)

        return -bond.price_change_from_spread_widening(delta_spread)

    # ------------------------------------------------------------------
    # Rating migration loss for loans / bonds
    # ------------------------------------------------------------------

    def migration_loss_for_asset(
        self, asset_rating: str, notional: float, duration: float = 3.0
    ) -> float:
        """
        Expected loss from rating migration over horizon.

        Computed as: sum over destination ratings d of
            P(migrate to d) × spread_change(d) × duration × notional / 10_000
        """
        if asset_rating not in _RATINGS:
            return 0.0
        from_idx = _RATINGS.index(asset_rating)
        tm = self._stressed_transition_matrix()

        loss = 0.0
        for to_idx, to_rating in enumerate(_RATINGS):
            prob = tm[from_idx, to_idx]
            if to_idx == from_idx:
                continue
            from_spread = _BASE_SPREADS.get(asset_rating, 0.0)
            to_spread = _BASE_SPREADS.get(to_rating, 0.0)
            spread_change = to_spread - from_spread  # positive = widening
            # Approximate P&L from spread change
            pnl = -spread_change / 10_000.0 * duration * notional
            loss += prob * (-pnl)  # convert to loss (positive=bad)

        return loss * self.migration_horizon_years

    # ------------------------------------------------------------------
    # Portfolio-level analysis
    # ------------------------------------------------------------------

    def compute(self, portfolio) -> CreditSpreadResult:
        """
        Compute total credit spread losses for a portfolio.

        Parameters
        ----------
        portfolio : Portfolio

        Returns
        -------
        CreditSpreadResult
        """
        from ..portfolio.assets import BondAsset, LoanAsset

        loss_by_asset: Dict[str, float] = {}
        mtm_loss = 0.0
        migration_loss = 0.0

        for asset in portfolio.assets:
            asset_loss = 0.0

            if isinstance(asset, BondAsset):
                shock = self.spread_shock.get(asset.rating.value, 0.0)
                bl = self.bond_spread_loss(asset)
                mtm_loss += bl
                asset_loss += bl

                if self.apply_rating_migration:
                    ml = self.migration_loss_for_asset(
                        asset.rating.value,
                        asset.market_value,
                        asset.duration,
                    )
                    migration_loss += ml
                    asset_loss += ml

            elif isinstance(asset, LoanAsset):
                if self.apply_rating_migration:
                    ml = self.migration_loss_for_asset(
                        asset.rating.value,
                        asset.ead,
                        asset.maturity_years / 2.0,  # approximate duration
                    )
                    migration_loss += ml
                    asset_loss += ml

            loss_by_asset[asset.asset_id] = asset_loss

        total_loss = mtm_loss + migration_loss

        return CreditSpreadResult(
            total_loss=total_loss,
            loss_by_asset=loss_by_asset,
            rating_migration_loss=migration_loss,
            mark_to_market_loss=mtm_loss,
        )

    # ------------------------------------------------------------------
    # Credit VaR via spread distribution
    # ------------------------------------------------------------------

    def credit_var(
        self,
        portfolio,
        confidence: float = 0.99,
        n_simulations: int = 10_000,
        seed: Optional[int] = 42,
    ) -> float:
        """
        Estimate Credit VaR by simulating spread shocks from a log-normal distribution.

        Returns
        -------
        float
            Credit VaR at the given confidence level.
        """
        rng = np.random.default_rng(seed)
        from ..portfolio.assets import BondAsset, LoanAsset

        losses = np.zeros(n_simulations)

        for asset in portfolio.assets:
            if isinstance(asset, BondAsset):
                base_spread = asset.spread_bps
                vol = base_spread * 0.50    # 50% vol on spread
                # Log-normal spread scenarios
                sim_spreads = base_spread * np.exp(
                    rng.normal(-0.5 * (vol / base_spread) ** 2, vol / base_spread, n_simulations)
                )
                delta_spreads = sim_spreads - base_spread
                dp = -asset.duration * delta_spreads / 10_000.0
                losses += -dp * asset.market_value

        return float(np.percentile(losses, confidence * 100))
