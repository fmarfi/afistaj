# BIST30 Strategy Lab — workflow, thresholds, methodology

## Why this mission exists

Two prior missions in this repo tried pairs trading (`examples/06_walk_forward_bist_pairs.py`, then `intraday_pairs_trading/_engine.py`) and, after that, volatility mean reversion (`intraday_pairs_trading/_engine_vol.py`) — both on an ad hoc 16-ticker universe. Neither made money reliably enough. Rather than tuning a third variant of the same idea, this mission asks a broader question: across the **real** BIST30 index, using several genuinely different strategy families and a validation harness rigorous enough to catch the exact overfitting trap already hit once this session (tuning one strategy's parameters on one ticker looked great, didn't generalize) — which approach is actually the most credible?

This is a self-contained mission, sibling to `examples/` and `intraday_pairs_trading/`, sharing no code with either (per this project's "don't mix missions" rule) beyond the same hand-rolled statistical primitives (ADF, Hurst) that appear, independently reimplemented, in all three.

## Files

| File | Role |
|---|---|
| `universe.py` | The real, reconciled BIST30 ticker list and the hourly data-download layer. |
| `_common.py` | Shared math/sizing utilities used by all 3 engines (session handling, ADF/Hurst, OU half-life, no-leverage capital sizing, performance stats). |
| `_engine_meanrev.py` | Strategy 1: OU-half-life pairs mean reversion, screened across the full BIST30 universe. |
| `_engine_momentum.py` | Strategy 2: cross-sectional momentum/trend rotation, long-only. |
| `_engine_pca.py` | Strategy 3: PCA-based basket statistical arbitrage. |
| `_ou_calibration.py` | Optional add-on to strategy 1: Monte Carlo calibration of entry_z/stop_z per pair, from that pair's own fitted OU parameters and an assumed trading cost, instead of the fixed 1.5/3.5 thresholds. |
| `validation.py` | The bias-correction layer: block bootstrap, Deflated Sharpe Ratio, cross-universe consistency, parameter-perturbation sensitivity, the leaderboard. |
| `02_run_full_universe_backtests.py` | Batch: download once, evaluate all 3 families (+ the cost-calibrated mean-reversion variant) across the full universe, cache to `_universe_backtest_cache.pkl`. |
| `03_run_validation_leaderboard.py` | Batch: build the DSR-corrected leaderboard + bootstrap + sensitivity check from the cache, save `_leaderboard.csv` / `_leaderboard_detail.pkl`. |
| `04_interactive_dashboard.py` | Flask dashboard (port 5100): a **Leaderboard** tab (reads 03's cache) and an **Explore** tab (live single-backtest parameter tweaking per family). |
| `test_*.py` | pytest regression suites, one per module, synthetic data only (no network) — see each file for what's covered. |

## The workflow

```
python bist30_strategy_lab/02_run_full_universe_backtests.py   # ~10s, needs network
python bist30_strategy_lab/03_run_validation_leaderboard.py    # ~1s, reads the cache
python bist30_strategy_lab/04_interactive_dashboard.py         # open http://127.0.0.1:5100
```
Run 02 again (and then 03) whenever you want the leaderboard tab to reflect fresh market data — the dashboard never recomputes the full-universe comparison itself, only the Explore tab's single live backtest.

## Universe & data

### The BIST30 list

`universe.BIST30_TICKERS` (30 names) was cross-validated against three independent sources (Aug 2026) and is the canonical list for this mission — deliberately **not** shared with `dashboard/bist30_dashboard.py` / `dashboard/bist30_webapp.py`, whose own hardcoded lists are older and were left untouched (out of scope for this mission; see the plan file this mission was built from).

Two real caveats, verified by direct download, not assumed:

- **`TRALT.IS`** is the renamed ticker for the former `KOZAL.IS` (Koza Altın İşletmeleri → Türk Altın İşletmeleri A.Ş.). yfinance only carries `TRALT.IS` history from **2025-11-24** onward — about 9 months as of 2026-08. Older bars under the `KOZAL.IS` name are not automatically included.
- **`DSTKF.IS`** IPO'd January 2025 — about 1.5 years of history, and noticeably higher volatility than the rest of the universe (reportedly BIST30's best YTD-2026 performer).

Both are in `universe.SHORT_HISTORY_TICKERS`. Neither is excluded from the universe outright — every engine's walk-forward loop naturally skips a ticker/pair wherever it lacks enough trailing history (`_common.min_history_guard`), rather than hardcoding an exclusion list. **`universe.download_universe` deliberately does not forward-fill or drop rows for a single short-history ticker's sake** — an earlier version of this function did a panel-wide `.dropna()`, which silently truncated the *entire* 30-ticker, multi-year panel down to whichever ticker IPO'd most recently (DSTKF.IS). Caught before this mission's engines were built, fixed to leave individual columns' pre-listing cells as NaN instead.

### Why hourly bars, not daily or 5-minute intraday

yfinance's `60m` interval allows roughly 2+ years of history per pull (a real download during this build returned 6,143 hourly bars spanning 2023-09-28 to 2026-08-18 — somewhat more than the commonly-cited 730-day guideline), versus a hard 60-day cap on 5m/15m bars and many-year depth for daily bars. This was a deliberate choice, made after weighing daily-bar depth (better for PCA factor stability and Deflated-Sharpe statistical power) against hourly granularity (closer to the "day trading" spirit of the earlier intraday mission) — hourly won as the practical middle ground.

**The tradeoff is real, not hidden**: ~2-3 years of hourly history is less calendar depth than a multi-decade daily study would give PCA factor stability or the Deflated Sharpe Ratio to work with. Results here should be read as "the best-supported answer given this data window," not as a claim with daily-study-grade statistical power.

## The three strategy families

### 1. Mean reversion (`_engine_meanrev.py`)

Engle-Granger two-step cointegration (OLS hedge ratio + ADF on the residual spread, session-gap-aware) screened across **all `C(30,2)=435` candidate pairs** in the full universe, not a hand-picked subset. The upgrade over both prior missions: entry/exit z-score windows are **derived per pair** from that pair's own fitted OU half-life (`_common.ou_half_life` → `derive_entry_exit_z`) instead of one fixed constant borrowed across every pair regardless of how fast or slowly it actually reverts. Position sizing uses a realized-volatility target (not a full GARCH refit — refitting `arch_model` per rebalance across up to 435 candidate pairs was needless cost for a signal that isn't what drives entries/exits). No forced end-of-session flatten — half-life-derived windows routinely span many sessions, so multi-day holds are expected, same as the daily mission (`examples/06`), not the day-trading one.

#### Optional: cost-calibrated entry_z (`_ou_calibration.py`)

The fixed `entry_z=1.5`/`stop_z=3.5` thresholds are borrowed constants — reasonable, but not derived from anything about the pair. `params["derive_entry_z_from_cost"]=True` (off by default) replaces them, at every rebalance, with a threshold calibrated via Monte Carlo simulation of **that pair's own fitted OU process**: simulate its (κ, σ) forward many times, trade a grid of candidate entry_z values (stop_z scaled to keep the same 1.5:3.5 ratio) against an assumed round-trip cost (`cost_per_round_trip_bps`, default 5 bps), and keep whichever threshold maximized simulated net PnL.

Why simulation instead of a textbook closed-form: the classic no-cost OU result says the optimal entry threshold is zero (capture every fluctuation) — but that assumes no stop-loss. Once a proportionally-scaled stop is added (as this engine always uses), a too-tight entry also means a too-tight stop that triggers on ordinary noise before a position can revert. Simulation confirmed this directly: there's a genuine **interior optimum even at zero transaction cost** (see `test_ou_calibration.py`), not a corner solution — exactly the kind of result that's easy to get wrong deriving by hand and worth checking by simulation instead.

**Real result, not illustrative**: running this on live BIST30 data (`03_run_validation_leaderboard.py`) produced a materially different leaderboard entry — DSR jumped from 0.53 (fixed thresholds) to 0.96 (calibrated), cross-pair consistency from 50% to 75% profitable, and the bootstrap 90% CI on Sharpe went from barely-clearing-zero (0.003 to 2.14) to clearly positive (0.32 to 3.55). That's a real improvement in this specific test, not a guaranteed one — checking a single pair in isolation (AKBNK.IS/TCELL.IS) actually showed the *opposite*, calibration hurt that pair's Sharpe (1.08 → 0.38). The lesson: calibration changes *which* pair looks best and by how much; it isn't uniformly better or worse, which is exactly why it's evaluated as its own separate leaderboard entry (`mean_reversion_ou_cost_calibrated`) through the same DSR/bootstrap machinery, not just adopted on the strength of one promising number. `validation.N_TRIALS` was bumped 12 → 14 to account for having tried this idea.

### 2. Momentum (`_engine_momentum.py`)

Genuinely new to this repo — the untried complementary family to mean reversion, which has now failed twice. Cross-sectional: rank all tickers with enough trailing history by rate-of-change, keep only those a fast/slow MA crossover confirms are actually trending (a whipsaw guard), hold the top N equally weighted until the next rebalance. **Long-only** — no shorting, consistent with the no-leverage philosophy and avoiding BIST margin/short-sale mechanics.

### 3. PCA basket stat-arb (`_engine_pca.py`)

Addresses the "small ad hoc pair-candidate set" problem directly by not picking a pair at all: every ticker's return is decomposed (from-scratch `numpy.linalg.eigh` on the trailing return covariance, no scikit-learn) into a common-factor part and an idiosyncratic residual, and the **residual** — not the raw price — is traded as mean-reverting, market/factor-neutral by construction. Long/short (a name can be shorted against the basket), equal-weighted across open residual positions.

### Shared sizing contract

All three engines' `run_full_backtest_*` return the same shape (`results` DataFrame with a `pnl` column, `trades` list of pnl-keyed dicts), so `_common.add_capital_pnl` works unmodified across all of them. No leverage, ever, and no manual risk dial:
- Mean reversion declares `scale_max=3.0` (a GARCH-style vol-targeting scale, same bound as `intraday_pairs_trading/_engine.py`).
- Momentum and PCA declare `scale_max=1.0` — never more than 100% of capital invested in aggregate — which makes `pnl` (a fractional return) convert to currency as `pnl * capital`, exactly right without a bespoke sizing function.

## Validation methodology

The whole point of this mission. Four independent checks, read together, not any one alone:

1. **Cross-universe consistency** (`validation.evaluate_meanrev_universe` / `evaluate_basket_universe`): mean reversion is re-run across the top 15 qualifying pairs (not just the single best one); momentum and PCA already trade the whole basket in one run, so consistency comes from what fraction of individually-traded names had net-positive pnl. A strategy profitable on 2 of 30 names is very likely noise.
2. **Moving block bootstrap** (`validation.bootstrap_significance`, mechanics reused directly from `examples/07_bootstrap_significance.py`): resamples the top-ranked family's actual OOS pnl in contiguous blocks (preserving trade-to-trade correlation) to get a 90% CI on Sharpe and P(Sharpe > 0) — is this one specific history's result distinguishable from noise?
3. **Deflated Sharpe Ratio** (`validation.deflated_sharpe_ratio`, Bailey & López de Prado 2014, implemented from scratch): discounts the observed Sharpe for the number of trials (`n_trials`) it took to find it. **`n_trials=12`** in `03_run_validation_leaderboard.py` — the accounting: 3 strategy families × roughly 4 meaningfully-different parameter configurations explored while building each engine (deliberately conservative/generous — it does *not* include the extensive parameter tuning that happened on the two now-abandoned missions earlier this session, only the tuning within this mission's own build). DSR is the leaderboard's primary sort key, specifically because raw Sharpe rewards exactly the kind of single-lucky-result overfitting this mission exists to avoid.
4. **Parameter-perturbation sensitivity** (`validation.perturbation_sensitivity`): nudges the top family's key parameters ±10%, reports the Sharpe delta. A result that collapses under a small nudge is a red flag for overfitting — the same lesson learned earlier this session when a single-ticker-tuned pairs strategy didn't generalize.

Run `03_run_validation_leaderboard.py` to see the current numbers; the dashboard's Leaderboard tab renders the same cached result.

## Key thresholds and defaults

| Constant | Value | Rationale |
|---|---|---|
| `universe.BIST30_INTERVAL` / `PERIOD` | `60m` / `730d` | Hourly middle ground — see "Why hourly bars" above. |
| `_engine_meanrev.DEFAULTS.session_lookback` | 20 sessions | Trailing window for hedge-ratio refit and regime re-check. |
| `_engine_meanrev.DEFAULTS.rebalance_sessions` | 5 sessions | How often the hedge ratio / half-life / z-window are re-derived. |
| `_engine_meanrev` z-window | derived from OU half-life | Not a fixed constant — see strategy 1 above. |
| `_engine_momentum.DEFAULTS.lookback_mom` | 180 bars (~20 sessions) | Trailing ROC-ranking window. |
| `_engine_momentum.DEFAULTS.ma_fast` / `ma_slow` | 45 / 180 bars | Trend-filter whipsaw guard. |
| `_engine_momentum.DEFAULTS.top_n` | 5 | Basket size. |
| `_engine_pca.DEFAULTS.fit_window` | 500 bars | Trailing window for each PCA refit. |
| `_engine_pca.DEFAULTS.n_factors` | 3 | Principal components removed before residualizing. |
| `_engine_pca.DEFAULTS.z_window` | 90 bars | Rolling window for the residual z-score. |
| `entry_z` / `stop_z` (meanrev, PCA) | 1.5 / 3.5 | Same interpretable meaning as both prior missions. |
| `validation.N_TRIALS` (in `03_...py`) | 14 | See "Deflated Sharpe Ratio" above; bumped from 12 for the cost-calibration variant. |
| `_engine_meanrev.DEFAULTS.cost_per_round_trip_bps` | 5 | Assumed round-trip cost for cost-calibrated entry_z (opt-in, see above). |
| `_common.SCALE_MIN` / `SCALE_MAX` | 0.2 / 3.0 | No-leverage bound, same as `intraday_pairs_trading/_engine.py`. |

## Future extensions (deliberately deferred, not built in v1)

- **ML classifiers** (gradient boosting / random forest on engineered technical features). Needs `scikit-learn` (not currently in `.venv`) and, more importantly, Combinatorial Purged Cross-Validation to be trustworthy — a substantial v2 build on its own, and the highest overfitting risk of everything considered for this mission.
- **HMM / Markov regime-switching** (trade momentum when trending, mean reversion when ranging, gated by a fitted hidden Markov model rather than a simple Hurst/ADX threshold). A lighter non-ML regime-aware ensemble using Hurst or ADX as the gate is a reasonable stretch goal once the 3 core families are proven out, but full HMM fitting was judged not worth its implementation cost for v1.
- **Kalman-filter dynamic hedge ratios**, **copula-based pair selection**, **ARIMA/SARIMA forecasting** — all surfaced during research as legitimate techniques, none included in v1 to keep scope bounded; see the plan file this mission was built from for the full research menu.

## What this workflow deliberately does NOT do

- No transaction costs baked into the headline Sharpe/return numbers (the cost-sensitivity table in the Explore tab shows the effect of adding them).
- No real order execution, no slippage model, no full per-leg share/lot-level accounting.
- Does not escape the free-data ceiling: yfinance's hourly cap genuinely limits how much history-hungry techniques (PCA, DSR) have to work with — flagged, not solved.
- `n_trials=12` for the Deflated Sharpe Ratio is this mission's own best-effort accounting, not an audited number — treat DSR as directionally informative, not a precise p-value.
