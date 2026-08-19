# Worked examples: Statistical Mean Reversion Testing

Companion code for [`Statistical Mean Reversion Testing.md`](../Statistical%20Mean%20Reversion%20Testing.md).
Each script is self-contained, runnable independently, and prints a step-by-step
worked example before running the full calculation, so you can see the math
happening rather than just trusting a library call.

## Setup

```bash
python3 -m venv .venv          # from the afistaj/ directory
source .venv/bin/activate
pip install -r examples/requirements.txt
```

`arch` and `yfinance` (used by scripts 04-05) are not pinned in
`requirements.txt` by default — install them the same way if you don't
already have them: `pip install arch yfinance`.

## Scripts, in order

| Script | Note section | What it calculates |
|---|---|---|
| [`01_ornstein_uhlenbeck_process.py`](01_ornstein_uhlenbeck_process.py) | Testing for Mean Reversion | Simulates the OU SDE via Euler-Maruyama; checks empirical mean/variance against the closed-form stationary distribution `sigma^2/(2*theta)`; generates the three example series (mean-reverting, random walk, trending) the next two scripts test. |
| [`02_adf_test.py`](02_adf_test.py) | Augmented Dickey-Fuller Test | Implements the ADF regression and `DF_tau` statistic from scratch with plain OLS, cross-checked against `statsmodels.adfuller`. Includes a pairs-trading spread example — the practical reason traders run this test at all. |
| [`03_hurst_exponent.py`](03_hurst_exponent.py) | Hurst Exponent | Implements the variance-scaling calculation (`log(Var(tau)) = 2H*log(tau) + c`) from scratch. Includes a live demonstration of the estimator's known finite-sample bias, and why `max(tau)` should stay well below the series length. |
| [`04_garch_volatility.py`](04_garch_volatility.py) | *(not in the note — a complementary tool)* | GARCH(1,1) volatility-clustering model: from-scratch MLE fit via `scipy.optimize`, cross-checked against the `arch` package, applied to real SPY returns. |
| [`05_real_world_pairs_trading.py`](05_real_world_pairs_trading.py) | *(combines all four)* | Downloads real data (Visa/Mastercard), runs the Engle-Granger cointegration test (OLS hedge ratio + ADF), confirms with the Hurst Exponent, then uses GARCH-fitted volatility to scale position size in a simple illustrative backtest. |
| [`06_walk_forward_bist_pairs.py`](06_walk_forward_bist_pairs.py) | *(fixes script 05's known gaps)* | Same pipeline, but on BIST (Borsa Istanbul) large-caps, with: in-sample pair *selection* vs. out-of-sample *testing*, a hedge ratio re-estimated daily instead of fixed forever, periodic re-validation of the cointegration evidence with a forced flatten if it breaks down, and a hard z-score stop-loss. Compares against script 05's original "fit once, trade forever" approach on the same data. |
| [`07_bootstrap_significance.py`](07_bootstrap_significance.py) | *(from "Bootstrap Aggregation..." note)* | Implements the moving block bootstrap from scratch to ask whether script 06's naive-vs-walk-forward comparison is statistically robust or a single lucky draw — resamples the two strategies' daily PnL (in correlated blocks, not independent days) thousands of times to build confidence intervals on Sharpe ratio and volatility. |

Run any script directly, e.g.:

```bash
python examples/01_ornstein_uhlenbeck_process.py
```

Scripts 02 and 03 load `examples/ou_example_series.npz`, written by script 01 —
run `01` at least once first. Scripts 04-07 need network access to fetch
real market data (SPY, then V/MA, then 16 BIST large-caps); 04-05 skip
that step gracefully if the download fails, while 06 and 07 need it to
run at all. Script 07 loads `06_walk_forward_bist_pairs.py` directly via
`importlib` (its filename starts with a digit, so it can't be a normal
`import` target) to reuse its backtest — no need to run 06 first, 07
re-runs it internally.

## The throughline

- **01-03** ask one question: *does the price level revert to a mean?*
  (OU gives the model, ADF gives the hypothesis test, Hurst gives an
  independent cross-check.)
- **04** asks a different question: *does volatility cluster over time?*
  A series can be perfectly mean-reverting and still have calm and
  turbulent regimes.
- **05** shows why real systematic strategies need both: ADF/Hurst decide
  *whether* to trade a spread; GARCH informs *how much* to bet at each
  point in time.
- **06** shows why a single fit isn't enough even when 01-05's logic is
  correct: markets aren't stationary over years, so the hedge ratio,
  the cointegration evidence, and the volatility model all need to be
  re-estimated over time, with a way to stop trading when the statistical
  basis itself breaks down.
- **07** asks the question none of the above answer: how much would 06's
  results have changed if history had unfolded slightly differently? It
  turns one backtest into a distribution of plausible backtests.

Every script's summary table ends with a plain-language paragraph
explaining what the numbers should show and why — read that even if you
skim the code above it.
