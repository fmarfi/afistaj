# Intraday pairs trading — workflow, thresholds, strategy

Adapts `examples/06_walk_forward_bist_pairs.py` (a daily-bar, walk-forward
BIST pairs-trading backtest) down to 5-minute bars and day-trading rules.
Read this file top to bottom before running anything — it explains *why*
each step exists, not just what to type.

## Why this needed its own folder, not just new parameters

Going from daily bars to 5-minute bars isn't a config change, it's a
different problem:

- **A relationship that holds daily may not hold intraday, or vice versa.**
  Daily cointegration is driven by slow forces (earnings, sector flows,
  macro). Intraday co-movement is driven by fast forces (order flow,
  market-maker inventory, correlated algo activity). There's no reason to
  assume the same pair wins at both timescales — Step 1 tests this
  directly instead of assuming it.
- **Two different clocks exist at once.** A daily backtest has one clock:
  one bar = one day = one recalibration step. Intraday, a session contains
  ~90 bars — recalibrating the hedge ratio and re-validating cointegration
  every bar would be both wasteful and statistically unstable (see
  Checkpoint 3), so this workflow explicitly splits a *slow* clock
  (once-per-session recalibration) from a *fast* clock (bar-by-bar z-score).
- **Sessions have edges.** The gap between one day's close and the next
  day's open is not a 5-minute price move; treating it as one would
  contaminate every statistic that touches price differences (the
  cointegration test, the GARCH fit). Day trading also means the position
  itself must never carry that gap: every position is closed by the last
  bar of its session, no exceptions.
- **Costs matter an order of magnitude more.** More trades, smaller moves
  per trade — see Checkpoint 4, which is arguably the single most
  important number this workflow produces.

## Files

- **`_engine.py`** — the actual logic (screening, walk-forward, trade log,
  cost sensitivity), parameterized, with no leading digit so it's a normal
  `import`able module. Every other file below calls into this one, so the
  batch scripts and the interactive dashboard can never disagree with
  each other about how a backtest is computed.
- **`01` / `02` / `03`** — the step-by-step CLI pipeline, using
  `_engine.py`'s DEFAULT thresholds. Each prints `>>> CHECKPOINT` guidance.
- **`04_interactive_dashboard.py`** — a local Flask app for changing
  thresholds (backtest AND statistical-test parameters), the data length
  (bar interval / history window), and capital & position sizing, then
  re-running live, including a full positions/trade log denominated in
  actual currency. See "Interactive parameter exploration" below.
- **`test_engine.py`** — a pytest suite for `_engine.py`, no network
  needed. See "Testing" below — this exists because two real bugs were
  found by hand during development and are now regression-tested.
- **`run_all.py`** — convenience wrapper that runs 01 → 02 → 03 in
  sequence for a quick end-to-end refresh with default parameters.

## The workflow

Run the batch pipeline in order (each writes files the next one reads):

```
cd intraday_pairs_trading
python 01_screen_intraday_pairs.py     # -> screening_results.csv, _selected_pair.txt
python 02_intraday_walk_forward.py     # -> _backtest_summary.csv, _cost_sensitivity.csv, _trades.csv
python 03_generate_dashboard.py        # -> dashboard.html (open in a browser)
```

or run everything in one shot with `python run_all.py`. Either way, run
`python -m pytest test_engine.py -v` at least once after pulling any
change to `_engine.py` — see "Testing" below.

This is meant to be a **stop-and-look pipeline, not a black box** — each
step is a separate script specifically so you can pause after it, read the
console output (or the dashboard), and decide whether to change a
parameter before moving on. That's what the ">>> CHECKPOINT" lines printed
by each script are for. The checkpoints below are the ones worth treating
as mandatory reading, not optional:

### Checkpoint 1 — after Step 1: did a real pair emerge?

Look at the screening table and the "qualifies" count. In the run this
workflow was built against: 13/120 candidate pairs qualified, and the
winner (`GARAN.IS/ISCTR.IS`, two BIST banks) was a **different pair** than
`examples/06`'s daily winner (`SAHOL.IS/YKBNK.IS`) — which still qualified
intraday, but far more weakly (ADF stat -3.22 vs. the new winner's -5.73).
**If you rerun this on a different day**, the winner may well change again
— that's expected, not a bug, given only ~60 days of history (see the data
caveat below). If *zero* pairs qualify, don't force a trade; that's the
pipeline correctly telling you there's no signal to act on right now.

### Checkpoint 2 — after Step 2: does the strategy actually trade?

Check `trades entered` and `% time disqualified`. Too few trades (near
zero) usually means `entry_z` is too high for this pair's actual intraday
z-score range — check the dashboard's z-score panel: if it rarely reaches
1.5, lower `entry_z`. Consistently near-100% disqualified means
`session_lookback` is too short to get a stable cointegration read each
session (see Checkpoint 3) — raise it.

### Checkpoint 3 — the automated red-flag block

Step 2 prints a `DIAGNOSTIC CHECKS` section that fires automatically, not
just a caveat you have to remember to reread:

- **Sharpe > 3** triggers a bid-ask-bounce warning. In the reference run,
  the no-cost Sharpe came out at **3.75** — a number that should make you
  suspicious, not pleased. 5-minute bars with zero transaction costs
  modeled are the textbook setup for accidentally backtesting the bid-ask
  spread bouncing back and forth as if it were real mean reversion.
- **Hedge-ratio instability** (wide swings, or a sign flip across
  sessions) triggers a warning that `session_lookback` may be too short.
  In the reference run, beta ranged from **-0.18 to +1.09** across only 23
  out-of-sample sessions — a genuinely unstable estimate, most likely
  because 5 sessions (~475 bars) just isn't much data for pinning down an
  intraday relationship, compared to the daily version's 252-*day* window.

### Checkpoint 4 — the cost-sensitivity table (the most important one)

Step 2 also prints a table of Sharpe ratio as a function of a per-trade
cost, in basis points, charged every time the position changes. This
directly tests Checkpoint 3's bid-ask-bounce suspicion instead of just
asserting it. In the reference run:

| bps / switch | Sharpe |
|---|---|
| 0 | 3.75 |
| 1 | 2.87 |
| 2 | 1.98 |
| 5 | -0.71 |
| 10 | -5.14 |
| 20 | -12.96 |

Sharpe crosses zero **between 2 and 5 bps**. A real BIST large-cap
round-trip (crossing the spread on both legs, open and close) commonly
runs 4-20 bps depending on the names and how the order is worked — meaning
this backtest's edge would most likely NOT have survived real trading
costs, not just "might not." **This is the single most useful output of
this entire workflow**: it turns "the backtest looks great" into "the
backtest looks great IF your real trading costs are under roughly 2-5 bps
per switch, otherwise it loses money" — a falsifiable, checkable claim
instead of a vibe.

### Checkpoint 5 — the positions table: read individual trades, not just aggregates

Step 2 also prints a full trade log (and saves it to `_trades.csv` /
`dashboard.html`'s "Positions opened" table): one row per completed
position, with entry/exit time, direction, entry/exit z-score, how many
bars it was held, and — critically — **why it closed**
(`reversion` / `hard stop` / `end of day` / `regime break`). Aggregate
stats can hide a lot: in the reference run, 42 trades broke down as 26
`reversion` (the intended, healthy outcome), 10 `end of day`, and 6
`hard stop`. A strategy dominated by `hard stop` exits is telling you
`entry_z`/`stop_z` are miscalibrated for this pair's actual volatility,
something no aggregate Sharpe number would surface on its own.

### Checkpoint 6 — the dashboard

`dashboard.html` is meant as the fast way to re-run Checkpoints 1-5
visually: KPI tiles up top (with a red banner if the Sharpe flag fired),
cumulative PnL at both zero cost and the approximate breakeven cost side
by side, the full positions table, the z-score with disqualified periods
shaded, hedge-ratio stability, and the same cost-sensitivity and
screening tables — every chart and table has a "How to read this" guide
box underneath it. It's deliberately basic — static charts generated once
per run, not a live monitor — regenerate it any time you rerun Step 2
with different parameters, or use the interactive dashboard below instead.

## Interactive parameter exploration

`03`'s dashboard is a snapshot of ONE parameter set (`_engine.DEFAULTS`).
`04_interactive_dashboard.py` is a local Flask app for trying different
values live, across three groups:

```
python 04_interactive_dashboard.py
# then open http://127.0.0.1:5050
```

- **Statistical tests & backtest rules**: `in_sample_fraction`, `adf_crit`,
  `hurst_cutoff`, `session_lookback`, `z_window`, `entry_z`, `stop_z` —
  the same thresholds documented above, including the statistical-test
  parameters, not just the trading rules. Number inputs use `step="any"`
  deliberately: HTML's native step validation compares against exact
  float multiples of the declared step from `min`, and binary floating
  point can't represent values like `0.05` exactly — that was rejecting
  legitimate values (e.g. an ADF critical value of `-2.5`) with a
  spurious "enter a valid value" error. `step="any"` disables that
  browser-side check entirely; `min`/`max` still apply.
- **Data ("how long")**: `interval` (5m/15m/30m/60m) and `period`, each
  constrained to what yfinance actually allows for that interval (5m/15m/30m
  top out at 60 days; 60m goes back up to 730 days — see the thresholds
  table's `BIST_INTERVAL`/`BIST_PERIOD` row for why this ceiling exists at
  all). Switching interval re-downloads data (a different bar size is a
  different dataset, and a different pair may win the screening step
  entirely at a coarser resolution — the same "timescale changes the
  answer" point Checkpoint 1 makes about 5m vs. daily, now visible at 5m
  vs. 60m too).
- **Capital & position sizing ("how much money")**: `starting_capital`
  (TRY) and `risk_per_trade_pct`. These convert the engine's abstract
  ±1-unit, GARCH-vol-scaled position into an actual currency P&L via
  `_engine.dollar_per_unit()`: a single conversion factor (`capital ×
  risk% ÷ typical forecast volatility`) applied uniformly across the
  run, sized so a position experiencing *typical* volatility risks
  roughly `risk_per_trade_pct`% of capital. This is deliberately a
  simplification — one constant scaling factor, not full per-leg
  share-count/lot-size accounting or separate capital allocation for the
  long vs. short leg — so treat the resulting equity curve and
  "Final capital" tile as "roughly how much money," not an exact trade
  blotter. It answers "how much would this have made or lost," not
  "exactly how many shares of each stock to hold."

Price data is downloaded once per (interval, period) combination and
cached in-process; changing a threshold or capital parameter and clicking
"Re-run backtest" re-executes the full screen → select → walk-forward →
trade-log pipeline (the same `_engine.run_full_backtest` +
`_engine.add_capital_pnl` calls) against the cached prices, so results
update in a second or two, not a fresh network fetch each time (only an
interval/period change triggers a new download). The page shows the same
KPI tiles, charts, cost table, and full positions log as the static
dashboard — now in currency terms — recomputed for whatever you just
typed in.

**Two concrete examples of what this is for**:
- Raising `session_lookback` from 5 to 10 sessions (double the
  calibration window) drops the no-cost Sharpe from 3.75 to well negative
  territory in this dataset — a vivid, interactive demonstration of
  Checkpoint 3's point that this pair's hedge ratio is genuinely unstable
  with too little data, not a robust result that happens to look good at
  one specific setting.
- Switching `interval` from 5m to 60m (with `period=730d`) picks a
  *different* winning pair entirely (`AKBNK.IS/TCELL.IS` instead of
  `GARAN.IS/ISCTR.IS` in one run) — the same lesson as Checkpoint 1
  (daily vs. 5m can disagree on the best pair), now shown between two
  different intraday resolutions.

## Testing

`test_engine.py` is a pytest suite for `_engine.py`, built entirely on
small synthetic price series (no network access needed):

```
python -m pytest test_engine.py -v
```

**Why this exists**: real bugs were found by manually inspecting backtest
output during development — reading the actual trade log and checking
individual trades against real quoted prices, not just trusting aggregate
stats — all now permanently regression-tested so they can't silently come back:

1. **Overnight-gap z-score spikes.** The rolling z-score window originally
   could span a session boundary. A real overnight price gap, viewed
   through a freshly re-estimated (session-to-session) hedge ratio,
   produced z-scores in the hundreds right at session open — not a data
   error, an artifact of computing a "recent average" across a boundary
   that shouldn't be crossed. Fixed by capping the window at the current
   session's start. Guarded by
   `test_walk_forward_zscore_bounded_despite_large_overnight_jump`.
2. **Thin-sample variance blowup.** Even after fix #1, the first couple of
   bars in a session had so few same-session observations that the
   standard deviation estimate itself was unstable, occasionally
   producing a z-score over 400 on the second bar of a session. Fixed by
   `MIN_WINDOW_BARS` (see the thresholds table): the z-score is held at
   exactly 0 (no signal, no possible entry) until enough same-session
   bars have accumulated. Guarded by
   `test_walk_forward_zscore_zero_during_min_window_warmup`.
3. **Trading on a degenerate (negative) hedge ratio.** With
   `session_lookback=5`, the daily beta re-estimation occasionally produced
   a NEGATIVE hedge ratio for GARAN.IS/ISCTR.IS — economically nonsensical
   for two same-sector banks (it implies "when one rises, the other should
   fall"), a symptom of too little data for a stable regression, not a real
   relationship. The strategy traded on it anyway: on 2026-08-10 this
   produced a trade whose reported P&L didn't reconcile against the two
   stocks' actual price moves at all when checked by hand — the "spread"
   had stopped meaning anything. Fixed by `BETA_SANITY_BOUNDS` (see the
   thresholds table): a session whose re-estimated beta falls outside a
   sane range is disqualified, same as a failed ADF/Hurst check. Guarded by
   `test_walk_forward_disqualifies_session_with_negative_beta`.

Also worth internalizing from bug #3's discovery process: the position-
sizing conversion (`dollar_per_unit_for_trade`) was separately found to be
oversizing every multi-bar trade, by calibrating to a single 5-minute
bar's volatility instead of the specific trade's actual distance to its
stop-loss. A 52-bar trade lost 18,838 TRY against an intended 3,750 TRY
risk budget — ~5x oversized — before the fix. Both of these were caught
the same way: reading the trade log line by line and checking it against
real prices, not by staring at an aggregate Sharpe ratio. That's the
actual argument for Checkpoint 5 above, demonstrated twice.

Both were the kind of bug that doesn't crash anything and doesn't look
obviously wrong in a quick glance at aggregate stats — they only surfaced
by reading the actual trade log (Checkpoint 5), which is itself a good
argument for always checking the positions table, not just the summary
numbers. If you modify `_engine.py`, re-run the suite before trusting new
output from any of the scripts.

## Thresholds — what each one does and why it's set where it is

All of these live in `_engine.DEFAULTS` and (aside from the two data-ceiling
rows) are adjustable live in `04_interactive_dashboard.py` without editing
code.

| Parameter | Value | Rationale | What changing it trades off |
|---|---|---|---|
| `BIST_INTERVAL` | `5m` | 1-minute bars are only available for 7 days from yfinance (too short for any in-sample/out-of-sample split); 5m is the finest resolution with a usable amount of history. | Coarser (15m) → more statistical power per fit, fewer, slower signals. Finer (1m) → far less history, much noisier microstructure. *(not exposed in 04 — changing it means re-downloading data)* |
| `BIST_PERIOD` | `60d` | yfinance's maximum lookback for 5-minute bars. Not a design choice — a hard data ceiling. | None available via this data source; a paid intraday data vendor would remove this ceiling entirely and is the real fix if this strategy looks promising. *(not exposed in 04)* |
| `in_sample_fraction` | `0.6` | Same 60/40 split as `examples/06`, for consistency — pair *selection* must never see the *test* period. | Higher → more reliable pair selection, less out-of-sample test data (already thin at 23 sessions). |
| `session_lookback` | `5` sessions | Enough bars (~475) for an ADF/GARCH fit to run at all, while staying short enough to be "fast" relative to a 56-session dataset. Flagged in Checkpoint 3 as possibly still too short — see the Sharpe-collapse example in "Interactive parameter exploration." | Higher → more stable hedge ratio and regime read, slower to adapt to a genuinely shifting relationship. |
| `z_window` | `60` bars (~5 hours) | The "fast" clock: long enough to smooth out single-bar noise, short enough to update meaningfully within a session. | Shorter → more entries, more whipsaw. Longer → fewer, slower signals; starts to blend into `session_lookback`'s territory. |
| `MIN_WINDOW_BARS` | `10` bars | Guards the z-score against the thin-sample variance blowup described in "Testing" above — with fewer than this many same-session bars, the z-score is forced to 0 (no signal) rather than trusting a standard-deviation estimate from almost no data. *(not exposed in 04 — a correctness floor, not a strategy knob)* | Higher → more of each session's opening bars are untradeable, but a more reliable z-score once trading starts. Lower → risks reintroducing the exact blowup this constant exists to prevent. |
| `BETA_SANITY_BOUNDS` | `(0.0, 3.0)` | Guards against the degenerate-hedge-ratio bug described in "Testing" above — a session whose re-estimated beta falls outside this range is disqualified rather than traded, since a non-positive (or wildly large) beta between two same-sector stocks signals an unstable regression, not a real relationship. *(not exposed in 04 — a correctness floor, not a strategy knob)* | Wider → more sessions allowed to trade even with a shakier beta estimate. Narrower → stricter, more time disqualified, but a stronger guarantee the "spread" traded actually means something. |
| `entry_z` | `1.5` | Same value as `examples/06`, kept as a starting point — **not re-tuned for intraday's different z-score behavior**, see Checkpoint 2. | Lower → more trades, more false starts. Higher → fewer, higher-conviction entries. |
| `stop_z` | `3.5` | Wide enough not to trigger on normal reversion noise, tight enough to actually cap the worst-case single-trade loss. | Lower → more stop-outs on noise. Higher → larger max per-trade loss before the safety net engages. |
| `adf_crit` / `hurst_cutoff` | `-2.86` / `0.45` | Same ADF 5% critical value and Hurst cutoff as `examples/06`'s screening step. The walk-forward's periodic in-loop regime re-check loosens the Hurst side by +0.05 internally (i.e. effectively 0.5) since a ~475-bar Hurst estimate is noisier than the in-sample screening estimate — see `examples/03`'s own documented bias with short windows. | Stricter (more negative `adf_crit`, lower `hurst_cutoff`) → more time disqualified (flat), fewer sessions traded, higher-confidence qualification. Looser → risk trading through a relationship that's actually broken down. |
| Cost-sensitivity grid | `0, 1, 2, 5, 10, 20` bps | Spans from "frictionless" to "wide small-cap-like spread," bracketing the realistic 4-20 bps round-trip range for liquid BIST large-caps. | N/A — this is a diagnostic sweep, not a strategy parameter. *(not exposed in 04)* |

## The strategy itself, end to end

1. **Universe**: the same 16 curated liquid BIST large-caps as
   `examples/06` (banks, holdings, industrials, telecom, airlines, autos,
   retail, glass) — not an official index scrape, just tickers likely to
   have real sector relationships.
2. **Selection (in-sample only)**: for every pair, regress
   `log(price_A)` on `log(price_B)` (OLS hedge ratio), test the residual
   spread for a unit root (session-aware ADF, so overnight gaps don't leak
   in) and cross-check with the Hurst Exponent. Rank by ADF statistic and
   take the strongest qualifying pair.
3. **Out-of-sample walk-forward, day by day**: at the start of every
   session, using only the preceding `session_lookback` sessions:
   recompute the hedge ratio, re-validate cointegration (flatten and
   pause trading for the session if it fails), and refit GARCH(1,1) on
   the spread's changes, forecasting volatility forward for that session
   (never reusing an in-sample fit as if it were known ahead of time).
4. **Signal, bar by bar within the session**: a rolling `z_window`-bar
   z-score of the spread (using that session's fixed hedge ratio). Enter
   when `|z| > entry_z`, exit when `z` crosses back through 0, hard-stop
   if `|z| > stop_z` even after entering.
5. **Position sizing**: scaled inversely to the session's GARCH-forecast
   volatility — smaller bets when the spread is turbulent, larger when calm.
6. **Risk discipline unique to day trading**: every position is force-flattened
   at the last bar of its session. No exceptions, regardless of the z-score.
7. **Before trusting any of it**: check the cost-sensitivity table
   (Checkpoint 4). This strategy's fate is decided there, not in the
   headline Sharpe ratio.

## What this workflow deliberately does NOT do

- **Place real orders.** This is a research/backtesting pipeline only.
- **Model transaction costs in the headline numbers** (only in the
  separate sensitivity table) — real deployment needs an actual cost
  model built in from the start, not bolted on as an afterthought.
- **Do full share-level position sizing.** The interactive dashboard's
  "how much money" figures (starting capital, final capital, per-trade
  currency P&L) come from one constant conversion factor
  (`_engine.dollar_per_unit`), not per-leg share counts, lot-size
  rounding, or separate long/short capital allocation. Good for "roughly
  how much money," not a real trade blotter.
- **Escape the free-data ceiling by default.** `01`/`02`/`03` use 5-minute
  bars capped at 60 days (yfinance's limit for that resolution) — treat
  every statistic they produce as lower-confidence than the daily
  version's 5-year backtest by roughly an order of magnitude.
  `04_interactive_dashboard.py` can trade this off (coarser bars, e.g.
  60-minute, unlock up to ~730 days), but that's a different, coarser
  timescale, not "more data at the same resolution" — there's no way to
  get more than 60 days of genuine 5-minute BIST history for free.
- **Bootstrap-test significance**, the way `examples/07` did for the daily
  version. With only 23 out-of-sample sessions at the default settings, a
  block bootstrap here would have very few independent blocks to draw
  from and its confidence intervals would be extremely wide — worth doing
  once more intraday history is available (e.g. via the 60-minute/730-day
  option), not particularly informative yet at 5-minute/60-day sample sizes.
