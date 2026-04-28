# Risk-Model-Stress-Test

A comprehensive **Banking Stress Test Simulator** for large institutional firms with complex multi-asset portfolios.

## Overview

This simulator combines four complementary methodologies:

| Approach | Module | Purpose |
|---|---|---|
| **Bottom-Up** | `models/bottom_up.py` | Instrument-level PD/LGD/EAD credit loss + Vasicek unexpected loss |
| **Top-Down** | `models/top_down.py` | Macro-factor regression on sector loss rates, CECL/IFRS-9 provisioning |
| **Monte Carlo** | `models/monte_carlo.py` | 13-factor correlated simulation (Gaussian or Student-t copula), VaR/ES |
| **Market Shock** | `models/market_shock.py` | Instantaneous scenario shocks across all asset classes |

Additional modules:
- **Credit Spreads** (`models/credit_spreads.py`) — spread widening P&L, rating migration losses, credit VaR
- **Leverage Risk** (`models/leverage_risk.py`) — Basel III/IV CET1 ratio, Tier-1 leverage, DFAST pass/fail, capital shortfall
- **Scenario Library** (`scenarios/recession_scenarios.py`) — 7 pre-built macro scenarios

---

## Quick Start

### Web Dashboard (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### Command-line example

```bash
python examples/run_stress_test.py
```

---

```
stress_test/
├── portfolio/
│   ├── assets.py          # LoanAsset, BondAsset, EquityAsset, DerivativeAsset, RealEstateAsset
│   └── portfolio.py       # Portfolio container — aggregation, RWA, HHI, CET1
├── scenarios/
│   └── recession_scenarios.py  # RecessionScenario dataclass + 7 named scenarios
├── models/
│   ├── bottom_up.py       # Vasicek ASRF credit loss model
│   ├── top_down.py        # Macro-factor regression + CECL/IFRS-9 provisioning
│   ├── monte_carlo.py     # Cholesky-correlated Monte Carlo engine
│   ├── credit_spreads.py  # Spread widening MTM + rating transition matrix
│   ├── market_shock.py    # Instantaneous scenario shock propagation
│   └── leverage_risk.py   # Basel III capital adequacy + DFAST assessment
├── reporting/
│   └── reports.py         # Text reports + Excel export
└── simulator.py           # Orchestrator — runs all modules, reverse stress test
examples/
└── run_stress_test.py     # End-to-end demo with synthetic ~$360B portfolio
tests/
└── test_stress_test.py    # 47 unit + integration tests
```

---

## Pre-Built Scenarios

| Scenario Key | Name | Severity | Equity | GDP | BBB Spread |
|---|---|---|---|---|---|
| `gfc_2008` | 2008 Global Financial Crisis | Extreme | -55% | -4.5% | +500bps |
| `covid_2020` | COVID-19 Pandemic Shock | Severe | -34% | -6.5% | +400bps |
| `stagflation` | Stagflation | Severe | -35% | -3.0% | +350bps |
| `sovereign_debt_crisis` | Sovereign Debt Crisis | Severe | -30% | -2.5% | +350bps |
| `moderate_recession` | Moderate Recession | Moderate | -20% | -2.0% | +200bps |
| `rate_shock` | Rapid Rate Rise Shock | Moderate | -15% | -1.0% | +200bps |
| `extreme_tail` | Extreme Tail Risk | Extreme | -70% | -10% | +1000bps |

---

## Usage

### Basic usage

```python
from stress_test.portfolio.assets import LoanAsset, BondAsset, EquityAsset, CreditRating
from stress_test.portfolio.portfolio import Portfolio
from stress_test.simulator import StressTestSimulator
from stress_test.reporting.reports import StressTestReport

# Build portfolio
portfolio = Portfolio(
    name="My Bank",
    tier1_capital=50_000_000_000,
    tier2_capital=10_000_000_000,
    total_liabilities=440_000_000_000,
)
portfolio.add_asset(LoanAsset(
    asset_id="L1", name="Corp Loan",
    asset_type=None, notional=20e9, market_value=20e9,
    sector="corporate", rating=CreditRating.BBB, lgd=0.40, maturity_years=5,
))
# ... add more assets ...

# Run stress tests
simulator = StressTestSimulator(portfolio, n_simulations=10_000, copula="student_t", seed=42)
results = simulator.run_all_scenarios()

# Print report
report = StressTestReport(results, portfolio, scale=1e9, scale_suffix="B")
report.print_report()

# Export to Excel
report.to_excel("stress_test_results.xlsx")
```

### Custom scenario

```python
from stress_test.scenarios.recession_scenarios import RecessionScenario

scenario = RecessionScenario(
    name="Geopolitical Energy Crisis",
    description="Oil surge + credit tightening",
    gdp_shock_pct=-0.035,
    unemployment_rise_pp=3.5,
    equity_market_decline=-0.28,
    yield_curve_shift_bps=150,
    credit_spread_shock={"BBB": 280, "BB": 520, "B": 900},
    real_estate_decline=-0.12,
    oil_price_change=1.20,
    pd_multiplier=2.8,
    severity="severe",
)
result = simulator.run_custom_scenario(scenario)
print(result.executive_summary())
```

### Reverse stress test

```python
# Find the minimum recession severity that breaches CET1 = 4.5%
reverse = simulator.reverse_stress_test(
    target_cet1_ratio=0.045,
    base_scenario_name="moderate_recession",
    severity_steps=25,
)
print(f"Breach at {reverse['critical_multiplier']:.2f}x severity")
print(f"Critical loss: ${reverse['critical_loss'] / 1e9:.1f}B")
```

---

## Risk Factors (Monte Carlo)

The Monte Carlo engine simulates 13 correlated risk factors using a Cholesky decomposition of the empirical correlation matrix:

- Equity returns
- Interest rate parallel shift
- Credit spreads: AAA / AA / A / BBB / BB / B / CCC
- FX rates
- Real-estate prices
- Oil / commodity prices
- Interbank funding spread

Supports **Gaussian** and **Student-t** copulas. The Student-t copula (default, df=5) captures fat-tail dependence and is appropriate for institutional stress testing.

---

## Capital Adequacy Framework

The leverage risk module implements a simplified Basel III / DFAST framework:

| Ratio | Minimum |
|---|---|
| CET1 | 4.5% |
| Tier-1 | 6.0% |
| Total Capital | 8.0% |
| Tier-1 Leverage | 3.0% |
| Capital Conservation Buffer | +2.5% |

Post-stress capital is computed as:

```
Net Capital Impact = max(Stressed Loss − PPNR over 9 quarters, 0)
Post-Stress Tier-1 = Pre-Stress Tier-1 − Net Capital Impact
```

where PPNR (pre-provision net revenue) is estimated from the stressed net interest margin on the loan book.

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/test_stress_test.py -v
```

47 tests covering all modules: asset Greeks, portfolio aggregation, scenario scaling, Monte Carlo correlation structure, VaR/ES ordering, bottom-up Vasicek formula, top-down sector losses, credit spread MTM, market shock propagation, Basel III capital ratios, and full end-to-end simulator integration.
