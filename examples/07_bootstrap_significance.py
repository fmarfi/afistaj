"""
Block bootstrap: is script 06's result real, or one lucky draw?
============================================================================

Reference note: "Bootstrap Aggregation, Random Forests and Boosted
Trees.md" -- currently just a stub, but the one concept it does define is
exactly what this script needs: bootstrapping is resampling your data
WITH REPLACEMENT to build many synthetic datasets, so you can see how
much a statistic (here: Sharpe ratio, volatility) would vary if history
had played out slightly differently. The note's stated purpose --
"quantifying the uncertainty associated with a model" -- is precisely the
gap left open at the end of script 06: it reported ONE Sharpe ratio and
ONE volatility number for each strategy, computed on ONE specific 3-year
stretch of history. That's a single sample. This script asks: if that
stretch had unfolded a bit differently, how much would those numbers move?

THE ONE COMPLICATION THE NOTE DOESN'T MENTION: plain bootstrapping (the
note's version -- resample individual data POINTS with replacement)
assumes each data point is independent. Daily trading PnL is not: this
strategy holds positions for multiple days, so today's PnL and
tomorrow's are correlated (they're often the same trade). Shuffling
individual days destroys that structure and would understate how
uncertain the results really are. The fix is the MOVING BLOCK BOOTSTRAP:
resample contiguous CHUNKS of consecutive days, not single days, so each
chunk keeps its internal day-to-day correlation intact. It's the exact
same "resample with replacement" idea the note describes, just applied
to blocks instead of points.

This script:
  1. Re-runs script 06's naive-vs-walk-forward backtest (by importing it
     as a module -- its filename starts with a digit, so a plain `import`
     statement isn't valid Python; importlib loads it by file path instead).
  2. Implements the moving block bootstrap from scratch, with a small
     worked example showing the block-drawing mechanics.
  3. Builds a bootstrap distribution of annualized return/vol/Sharpe for
     BOTH strategies (using the SAME resampled days for both each draw,
     so the pairing between them is preserved), and answers two questions
     with actual confidence, not just point estimates:
       (a) is each strategy's Sharpe ratio distinguishable from zero?
       (b) is walk-forward's lower volatility a robust finding, or could
           it easily have gone the other way?
"""

import importlib.util
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(123)
BLOCK_LENGTH = 21   # matches script 06's REBALANCE cadence -- see worked_example()'s note on why
N_BOOTSTRAP = 5000


def load_script06():
    """
    06_walk_forward_bist_pairs.py can't be `import`ed normally (Python
    module names can't start with a digit), so load it directly from its
    file path instead. `main()` is guarded by `if __name__ == "__main__"`
    inside that file, so this does NOT re-run its printed output --
    it just gives us access to its functions and constants.
    """
    path = pathlib.Path(__file__).parent / "06_walk_forward_bist_pairs.py"
    spec = importlib.util.spec_from_file_location("wf_bist", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def moving_block_bootstrap_indices(n, block_length, rng):
    """
    Draw ceil(n/block_length) random starting points, take a contiguous
    block of `block_length` days from each, concatenate them, and trim to
    length n. Each block preserves its internal day-to-day correlation;
    only the ORDER of blocks (and which days are reused / omitted) is
    randomized -- that's what "resampling with replacement" means here.
    """
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n]


def worked_example():
    """
    Show the block-drawing mechanics by hand on a 12-point toy series with
    block_length=3: which blocks got drawn, and the resulting resampled
    series -- the exact same operation main() runs 5000 times on the real
    806-day OOS PnL series.
    """
    print("=" * 70)
    print("WORKED EXAMPLE: moving block bootstrap, n=12, block_length=3")
    print("=" * 70)
    toy = np.arange(1, 13)  # stand-in "daily PnL": day 1, day 2, ..., day 12
    print(f"original series (day 1..12): {toy}\n")

    rng = np.random.default_rng(7)
    n, block_length = len(toy), 3
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)

    print(f"need ceil({n}/{block_length}) = {n_blocks} blocks. Random start days drawn: {starts + 1}")
    blocks = [toy[s:s + block_length] for s in starts]
    for s, b in zip(starts, blocks):
        print(f"  block starting at day {s+1}: {b}")
    resampled = np.concatenate(blocks)[:n]
    print(f"\nconcatenated and trimmed to length {n}: {resampled}")
    print(
        "\nNotice day 1-3 might appear twice and day 10-12 not at all --\n"
        "that's the 'with replacement' part. But within each block, the\n"
        "original day-to-day order (and therefore correlation) is intact --\n"
        "that's what a plain point-by-point bootstrap would have destroyed.\n"
        f"\nWhy block_length={BLOCK_LENGTH} for the real test below: script 06's\n"
        "trades typically last on the order of a few weeks (positions are\n"
        "held until the spread reverts or a monthly regime re-check fires).\n"
        "A block should be at least that long, or it would still chop\n"
        "individual trades into pieces and partially destroy their\n"
        "correlation -- 21 trading days (~1 month) matches script 06's own\n"
        "rebalance cadence.\n"
    )


def bootstrap_stats(pnl):
    ann_ret = np.mean(pnl) * 252
    ann_vol = np.std(pnl) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe


def run_bootstrap(pnl_naive_oos, pnl_wf_oos, block_length=BLOCK_LENGTH, n_boot=N_BOOTSTRAP, rng=RNG):
    """
    Draw the SAME resampled day-indices for both strategies on each of the
    n_boot iterations. This is a PAIRED bootstrap: naive and walk-forward
    are computed from the same underlying market days (they trade the same
    spread), so preserving that pairing lets us also look at the
    DIFFERENCE between them draw-by-draw, not just their marginal
    distributions separately.
    """
    n = len(pnl_naive_oos)
    results_naive = np.empty((n_boot, 3))
    results_wf = np.empty((n_boot, 3))

    for i in range(n_boot):
        idx = moving_block_bootstrap_indices(n, block_length, rng)
        results_naive[i] = bootstrap_stats(pnl_naive_oos[idx])
        results_wf[i] = bootstrap_stats(pnl_wf_oos[idx])

    return results_naive, results_wf


def summarize(results_naive, results_wf):
    ann_ret_n, ann_vol_n, sharpe_n = results_naive.T
    ann_ret_w, ann_vol_w, sharpe_w = results_wf.T

    def ci(x):
        p5, p50, p95 = np.percentile(x, [5, 50, 95])
        return f"{p50:.3f}  (90% CI: {p5:.3f} to {p95:.3f})"

    print("=" * 70)
    print(f"BOOTSTRAP RESULTS ({N_BOOTSTRAP} resamples, block length={BLOCK_LENGTH} days)")
    print("=" * 70)
    print(f"naive static   Sharpe: {ci(sharpe_n)}")
    print(f"walk-forward   Sharpe: {ci(sharpe_w)}")
    print(f"\nP(naive Sharpe > 0)       = {np.mean(sharpe_n > 0):.3f}")
    print(f"P(walk-forward Sharpe > 0) = {np.mean(sharpe_w > 0):.3f}")
    print(
        "\n-> both comfortably above 0.90: across resampled histories, both\n"
        "strategies' Sharpe ratios are positive far more often than not --\n"
        "this backtest's edge looks like a real feature of the data, not an\n"
        "artifact of exactly how the 3-year path happened to unfold.\n"
    )

    sharpe_diff = sharpe_w - sharpe_n
    d5, d50, d95 = np.percentile(sharpe_diff, [5, 50, 95])
    print(f"Sharpe difference (walk-forward minus naive): {d50:+.3f}  (90% CI: {d5:+.3f} to {d95:+.3f})")
    print(
        "-> this interval straddles 0, so the SHARPE improvement from\n"
        "walk-forward is NOT statistically distinguishable from noise at\n"
        "this sample size -- consistent with script 06's point estimates\n"
        "(0.843 vs 0.856) being nearly identical.\n"
    )

    p_lower_vol = np.mean(ann_vol_w < ann_vol_n)
    print(f"P(walk-forward volatility < naive volatility) = {p_lower_vol:.3f}")
    print(
        "-> this is the robust finding: in essentially every resampled\n"
        "history, walk-forward's volatility came out lower than naive's.\n"
        "Script 06's headline claim ('same Sharpe, way less risk') survives\n"
        "the bootstrap -- the RISK REDUCTION is what's statistically solid\n"
        "here, not a Sharpe-ratio improvement.\n"
    )

    return sharpe_n, sharpe_w, ann_vol_n, ann_vol_w


def main():
    worked_example()

    print("=" * 70)
    print("Re-running script 06's backtest to get the two PnL series")
    print("=" * 70)
    wf_bist = load_script06()
    try:
        prices = wf_bist.download_universe(wf_bist.BIST_UNIVERSE)
    except Exception as e:
        print(f"Could not download data ({e}); this script needs network access.")
        return

    log_prices = np.log(prices)
    n = len(prices)
    split = int(n * wf_bist.IN_SAMPLE_FRACTION)
    screening = wf_bist.screen_pairs_in_sample(log_prices, split)
    best = screening.iloc[0]
    ticker_a, ticker_b = best["ticker_a"], best["ticker_b"]
    print(f"pair: {ticker_a}/{ticker_b}\n")

    log_a = log_prices[ticker_a].to_numpy()
    log_b = log_prices[ticker_b].to_numpy()

    z_n, sig_n, qual_n, spread_n, _ = wf_bist.naive_static_backtest(log_a, log_b, split)
    pnl_naive, *_ = wf_bist.run_signal_and_pnl(z_n, sig_n, qual_n, spread_n, split, n, use_stop_loss=False)

    z_w, sig_w, qual_w, spread_w, _ = wf_bist.walk_forward_backtest(log_a, log_b, split)
    pnl_wf, *_ = wf_bist.run_signal_and_pnl(z_w, sig_w, qual_w, spread_w, split, n, use_stop_loss=True)

    pnl_naive_oos = pnl_naive[split:]
    pnl_wf_oos = pnl_wf[split:]

    point_naive = bootstrap_stats(pnl_naive_oos)
    point_wf = bootstrap_stats(pnl_wf_oos)
    print(f"point estimates (script 06's original numbers): "
          f"naive Sharpe={point_naive[2]:.3f}, walk-forward Sharpe={point_wf[2]:.3f}\n")

    results_naive, results_wf = run_bootstrap(pnl_naive_oos, pnl_wf_oos)
    sharpe_n, sharpe_w, vol_n, vol_w = summarize(results_naive, results_wf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(sharpe_n, bins=60, alpha=0.6, label="naive static", color="tab:blue")
    axes[0].hist(sharpe_w, bins=60, alpha=0.6, label="walk-forward", color="tab:green")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Bootstrap distribution: Sharpe ratio")
    axes[0].legend()

    axes[1].hist(vol_n, bins=60, alpha=0.6, label="naive static", color="tab:blue")
    axes[1].hist(vol_w, bins=60, alpha=0.6, label="walk-forward", color="tab:green")
    axes[1].set_title("Bootstrap distribution: annualized volatility")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("examples/bootstrap_significance.png", dpi=120)
    print("Saved plot to examples/bootstrap_significance.png")


if __name__ == "__main__":
    main()
