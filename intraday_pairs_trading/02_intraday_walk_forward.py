"""
STEP 2 of the intraday workflow -- see WORKFLOW.md for the full picture.

Walk-forward DAY-TRADING backtest on the pair 01_screen_intraday_pairs.py
selected, at 5-minute resolution, using the DEFAULT thresholds (see
_engine.py's DEFAULTS dict). Use 04_interactive_dashboard.py instead if
you want to try different thresholds without editing code -- this script
and that dashboard call the exact same underlying functions in
_engine.py, so results are always consistent between them.

This is the intraday analogue of examples/06_walk_forward_bist_pairs.py,
with the adaptations the daily version doesn't need:

  - TWO lookback windows instead of one: SESSION_LOOKBACK (a handful of
    trailing SESSIONS) recalibrates the hedge ratio / cointegration check /
    GARCH model once per session -- the "slow" process, directly analogous
    to examples/06's daily rolling window. Z_WINDOW (a number of BARS)
    drives the actual entry/exit z-score, updating continuously within the
    session -- the "fast" process this strategy trades off of.
  - FORCED FLATTEN AT SESSION END: day trading means no overnight
    exposure, full stop -- the position is closed at the last bar of every
    session regardless of the z-score, no exceptions.
  - Overnight-gap bars (the first bar of each new session) are excluded
    from every statistical fit -- see _engine.py's mask_overnight_gaps.

>>> CHECKPOINT (read this after running): look at "time spent
    DISQUALIFIED" and the number of trades. With only ~23 out-of-sample
    sessions (see WORKFLOW.md's data caveat), a handful of trades is
    normal, not a bug -- but if there are close to ZERO trades, ENTRY_Z is
    probably too high for this pair's intraday z-score range; check the
    dashboard's z-score panel and consider lowering it. If disqualified
    time is near 100%, SESSION_LOOKBACK may be too short to get a stable
    cointegration read -- try raising it. 04_interactive_dashboard.py lets
    you try both without editing this file.
"""

import numpy as np
import pandas as pd
import _engine as eng


def main():
    print("=" * 70)
    print("Loading screened pair + intraday prices from step 1")
    print("=" * 70)
    try:
        prices = pd.read_pickle("intraday_pairs_trading/_intraday_prices.pkl")
        with open("intraday_pairs_trading/_selected_pair.txt") as f:
            ticker_a, ticker_b = f.read().strip().split(",")
    except FileNotFoundError:
        print("Missing step-1 outputs -- run 01_screen_intraday_pairs.py first.")
        return

    print(f"pair: {ticker_a}/{ticker_b}\n")
    log_prices = np.log(prices)
    log_a = log_prices[ticker_a].to_numpy()
    log_b = log_prices[ticker_b].to_numpy()

    p = eng.DEFAULTS
    split_idx, sess = eng.split_by_session(prices, p["in_sample_fraction"])
    n_sessions = sess[-1] + 1
    print(f"out-of-sample: {n_sessions - sess[split_idx]} sessions "
          f"({prices.index[split_idx]} to {prices.index[-1]})\n")

    print("=" * 70)
    print(f"WALK-FORWARD DAY-TRADING BACKTEST "
          f"(session_lookback={p['session_lookback']}, z_window={p['z_window']} bars, "
          f"entry_z={p['entry_z']}, stop_z={p['stop_z']})")
    print("=" * 70)
    beta_path, z_path, qualified_path, sigma_path, spread_path, is_last_bar, sd_path = eng.walk_forward_intraday(
        log_a, log_b, sess, split_idx,
        p["session_lookback"], p["z_window"], p["adf_crit"], p["hurst_cutoff"],
    )
    pnl, position, trades, n_stops, n_regime_flat, n_eod_flat, n_trades = eng.run_signal_pnl_and_trades(
        z_path, sigma_path, qualified_path, spread_path, is_last_bar, prices.index.to_numpy(),
        split_idx, len(prices), p["entry_z"], p["stop_z"],
    )
    trades = eng.attach_trade_pnl(trades, pnl, prices.index.to_numpy())
    trades = eng.attach_trade_prices(trades, prices, ticker_a, ticker_b)
    stats = eng.performance_stats(pnl, split_idx)

    oos_mask = np.arange(len(prices)) >= split_idx
    pct_disqualified = (~qualified_path[oos_mask]).mean() * 100
    beta_by_session = pd.Series(beta_path[oos_mask]).groupby(sess[oos_mask]).first()
    print(f"hedge ratio: recalibrated every session, ranged "
          f"{beta_by_session.min():.3f} to {beta_by_session.max():.3f} "
          f"(std across sessions: {beta_by_session.std():.3f})")
    print(f"time spent DISQUALIFIED (flat, no new trades): {pct_disqualified:.1f}% of out-of-sample bars")
    print(f"trades entered: {n_trades} | regime-break flattenings: {n_regime_flat} | "
          f"hard stop-losses: {n_stops} | end-of-day flattenings: {n_eod_flat}")
    print(pd.DataFrame([stats]).to_string(index=False))
    print(
        "\nRemember: no transaction costs, no bid-ask spread, no slippage are\n"
        "modeled in the headline numbers -- intraday strategies are far more\n"
        "sensitive to these than daily ones. See the cost-sensitivity table below.\n"
    )

    print("=" * 70)
    print("POSITIONS OPENED (trade log)")
    print("=" * 70)
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["direction"] = trades_df["direction"].apply(
            lambda d: eng.direction_label(d, ticker_a, ticker_b)
        )
        print(trades_df.to_string(index=False))
        print(f"\n{len(trades_df)} trades | win rate: {(trades_df['pnl'] > 0).mean()*100:.0f}% | "
              f"avg hold: {trades_df['hold_bars'].mean():.0f} bars | "
              f"avg pnl/trade: {trades_df['pnl'].mean():.5f}")
        print("\nexit_reason breakdown:")
        print(trades_df["exit_reason"].value_counts().to_string())
    else:
        print("No trades were entered -- see Checkpoint 2 in WORKFLOW.md.")

    # --- CHECKPOINT: automated red flags, not just a caveat paragraph ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC CHECKS -- read these before trusting the numbers above")
    print("=" * 70)
    flags = []

    if stats["sharpe (naive)"] > 3.0:
        flags.append(
            f"Sharpe of {stats['sharpe (naive)']} is implausibly high for a live strategy. The\n"
            f"  most likely explanation is NOT genuine edge: with {n_trades} trades on 5-minute\n"
            "  bars and zero transaction costs modeled, this backtest is almost certainly\n"
            "  capturing bid-ask BOUNCE -- the price mechanically wobbling between the bid\n"
            "  and ask as different order types execute -- rather than real mean reversion.\n"
            "  That bounce evaporates the moment you pay a real spread to trade it. This is\n"
            "  the single most common failure mode in naive intraday backtests. TREAT THIS\n"
            "  RESULT AS A WARNING SIGN, not a strategy to deploy, until a transaction-cost\n"
            "  model (at minimum: subtract the historical bid-ask spread per round-trip) is\n"
            "  added and the Sharpe is re-checked."
        )

    beta_range = beta_by_session.max() - beta_by_session.min()
    if beta_range > 0.5 or (beta_by_session.min() < 0 < beta_by_session.max()):
        flags.append(
            f"Hedge ratio swung from {beta_by_session.min():.2f} to {beta_by_session.max():.2f} across\n"
            f"  sessions (even changing SIGN) -- session_lookback={p['session_lookback']} sessions is probably\n"
            "  too short a window to pin down a stable intraday relationship for this pair;\n"
            "  the daily version's 252-day window had ~50x more data per fit. Consider\n"
            "  raising session_lookback (trades off responsiveness for stability), or accept\n"
            "  that this pair's intraday relationship may just be genuinely unstable, in\n"
            "  which case day-trading it on a re-estimated hedge ratio is intrinsically\n"
            "  riskier than the daily version, no matter how the other parameters are tuned."
        )

    if flags:
        for i, f in enumerate(flags, 1):
            print(f"[{i}] {f}\n")
    else:
        print("No automated red flags triggered.\n")

    print("=" * 70)
    print("COST SENSITIVITY: does the edge survive realistic transaction costs?")
    print("=" * 70)
    cost_table = eng.cost_sensitivity_table(pnl, position, split_idx)
    print(cost_table.to_string())
    breakeven = cost_table[cost_table["sharpe (naive)"] <= 0]
    if len(breakeven) > 0:
        breakeven_cost = breakeven.index[0]
        print(
            f"\nSharpe crosses zero somewhere at or before {breakeven_cost} bps/switch. A liquid\n"
            "BIST large-cap round-trip (crossing the spread on both legs, open and close) is\n"
            "commonly in the 4-20 bps range depending on the names and how the order is worked --\n"
            f"if that's above {breakeven_cost} bps, this backtest's edge would not have survived\n"
            "actually trading it. Compare this table's breakeven point against a real quoted\n"
            f"spread for {ticker_a}/{ticker_b} before taking the no-cost Sharpe seriously.\n"
        )
    else:
        print(
            "\nSharpe stays positive across the whole cost grid tested (0-20 bps/switch) --\n"
            "more encouraging than a typical naive intraday result, but still only as good as\n"
            "the grid tested; push cost_bps_grid higher in _engine.cost_sensitivity_table() if\n"
            "actual spreads for this pair run wider.\n"
        )

    results = pd.DataFrame({
        "timestamp": prices.index,
        "session": sess,
        "beta": beta_path,
        "z": z_path,
        "qualified": qualified_path,
        "sigma": sigma_path,
        "spread": spread_path,
        "sd": sd_path,
        "position": position,
        "pnl": pnl,
        "is_last_bar_of_session": is_last_bar,
    })
    results.to_pickle("intraday_pairs_trading/_walk_forward_results.pkl")
    stats["pair"] = f"{ticker_a}/{ticker_b}"
    stats["trades"] = n_trades
    stats["regime_flattenings"] = n_regime_flat
    stats["stop_losses"] = n_stops
    stats["pct_disqualified"] = round(pct_disqualified, 1)
    pd.DataFrame([stats]).to_csv("intraday_pairs_trading/_backtest_summary.csv", index=False)
    cost_table.to_csv("intraday_pairs_trading/_cost_sensitivity.csv")
    if trades:
        pd.DataFrame(trades).to_csv("intraday_pairs_trading/_trades.csv", index=False)
    print("\nSaved: _walk_forward_results.pkl, _backtest_summary.csv, _cost_sensitivity.csv, _trades.csv")
    print(">>> CHECKPOINT: review the numbers above, then run 03_generate_dashboard.py")
    print("    (or run 04_interactive_dashboard.py to try different thresholds live)")


if __name__ == "__main__":
    main()
