"""
STEP 1 of the intraday workflow -- see WORKFLOW.md for the full picture.

Screens the same BIST universe from examples/06_walk_forward_bist_pairs.py
for cointegration at 5-MINUTE bar resolution instead of daily bars, and
explicitly checks whether the pair that won at the daily timescale
(SAHOL.IS/YKBNK.IS) still qualifies intraday.

WHY THIS STEP EXISTS ON ITS OWN: cointegration is timescale-specific. Two
stocks can share a slow-moving daily equilibrium (driven by earnings,
sector rotation, macro flows) while showing no fast intraday relationship
at all (driven by order-flow noise, market-maker inventory, bid-ask
bounce) -- or vice versa. You cannot assume the daily pair carries over;
it has to be re-tested at the new timescale, which is exactly what this
script does.

DATA CONSTRAINT: yfinance only serves 5-minute bars for the trailing 60
calendar days (~93 trading sessions for BIST). That's a real limitation --
the daily backtest (examples/06) had 5+ years of history; this has ~2
months. Every conclusion here carries much wider uncertainty than the
daily version's. This is flagged again in WORKFLOW.md's caveats section.

The actual screening logic lives in _engine.py (shared with 02 and the
interactive dashboard, 04) -- this script is a thin CLI wrapper around it
using the DEFAULT thresholds. Use 04_interactive_dashboard.py instead if
you want to try different thresholds without editing code.

>>> CHECKPOINT (read this after running): see the "qualifies" column and
    the SAHOL/YKBNK comparison printed at the end. If fewer than ~10% of
    pairs qualify, or the top pair's p-value is only marginally under 0.05,
    treat the intraday edge as weak evidence, not a green light -- consider
    a longer intraday history (BIST_PERIOD), a coarser bar (15m instead of
    5m, trading off responsiveness for statistical power), or accept that
    this universe may just not have a strong intraday relationship yet.
"""

import numpy as np
import _engine as eng


def main():
    print("=" * 70)
    print(f"Downloading {eng.BIST_INTERVAL} bars, {eng.BIST_PERIOD} history, {len(eng.BIST_UNIVERSE)} BIST tickers")
    print("=" * 70)
    try:
        prices = eng.download_intraday_universe(eng.BIST_UNIVERSE)
    except Exception as e:
        print(f"Could not download data ({e}); this script needs network access.")
        return

    split_idx, sess = eng.split_by_session(prices, eng.DEFAULTS["in_sample_fraction"])
    n_sessions = sess[-1] + 1
    print(f"{len(prices)} bars across {n_sessions} sessions, "
          f"{prices.index.min()} to {prices.index.max()}")
    print(f"in-sample: bars 0:{split_idx} (sessions 0:{sess[split_idx]}) -- for SCREENING only")
    print(f"out-of-sample: bars {split_idx}:{len(prices)} (sessions {sess[split_idx]}:{n_sessions}) "
          f"-- reserved for 02_intraday_walk_forward.py\n")

    log_prices = np.log(prices)
    insample_log = log_prices.iloc[:split_idx]
    insample_sess = sess[:split_idx]

    print("=" * 70)
    print(f"Screening {len(eng.BIST_UNIVERSE)*(len(eng.BIST_UNIVERSE)-1)//2} pairs at "
          f"{eng.BIST_INTERVAL} resolution, IN-SAMPLE ONLY "
          f"(ADF crit={eng.DEFAULTS['adf_crit']}, Hurst cutoff={eng.DEFAULTS['hurst_cutoff']})")
    print("=" * 70)
    screening = eng.screen_pairs(insample_log, insample_sess, prices.columns,
                                  eng.DEFAULTS["adf_crit"], eng.DEFAULTS["hurst_cutoff"])
    print(screening.head(10)[["pair", "beta", "adf_stat", "H", "qualifies"]].to_string(index=False))
    n_qualify = screening["qualifies"].sum()
    print(f"\n{n_qualify}/{len(screening)} pairs qualify intraday.\n")

    print("=" * 70)
    print("COMPARISON: does the DAILY-selected pair still hold up intraday?")
    print("=" * 70)
    a, b = eng.DAILY_SELECTED_PAIR
    daily_row = screening[
        ((screening.ticker_a == a) & (screening.ticker_b == b)) |
        ((screening.ticker_a == b) & (screening.ticker_b == a))
    ]
    print(f"{a}/{b} (winner of examples/06's daily screen, ADF p=0.0005 there):")
    print(daily_row[["pair", "beta", "adf_stat", "H", "qualifies"]].to_string(index=False))

    best = screening.iloc[0]
    if (best.ticker_a, best.ticker_b) != eng.DAILY_SELECTED_PAIR and (best.ticker_b, best.ticker_a) != eng.DAILY_SELECTED_PAIR:
        print(
            f"\n-> The intraday winner ({best.pair}) is a DIFFERENT pair than the daily\n"
            f"winner. This is the point flagged in WORKFLOW.md: a relationship that's\n"
            "strong at one timescale is not guaranteed to be the strongest (or even\n"
            "present) at another. Trading this strategy intraday means re-selecting\n"
            "the pair for that timescale, not reusing the daily choice.\n"
        )
    else:
        print("\n-> Same pair wins at both timescales here -- a reassuring, though not guaranteed, sign.\n")

    print(f"Selected pair for the walk-forward backtest: {best.pair} "
          f"(ADF stat={best.adf_stat:.3f}, H={best.H:.3f})")

    prices.to_pickle("intraday_pairs_trading/_intraday_prices.pkl")
    screening.to_csv("intraday_pairs_trading/screening_results.csv", index=False)
    with open("intraday_pairs_trading/_selected_pair.txt", "w") as f:
        f.write(f"{best.ticker_a},{best.ticker_b}")
    print("\nSaved: _intraday_prices.pkl, screening_results.csv, _selected_pair.txt")
    print(">>> CHECKPOINT: review the table above, then run 02_intraday_walk_forward.py")


if __name__ == "__main__":
    main()
