"""
Putting it together: ADF + Hurst + GARCH on real market data
============================================================================

Scripts 01-03 taught the note's mean-reversion tools (OU process, ADF
test, Hurst Exponent). Script 04 taught GARCH, which answers a different
question (does volatility cluster?). This script shows how a real
systematic pairs-trading workflow actually combines them on live data --
two DIFFERENT jobs, not one bigger test:

  JOB 1 -- "Is there a tradeable mean-reverting relationship at all?"
    Engle-Granger two-step cointegration test:
      (a) OLS-regress log(price_A) on log(price_B) to get a hedge ratio.
      (b) Run the ADF test (script 02) on the regression residual
          ("the spread"). If the spread rejects the random-walk null,
          A and B are "cointegated" -- individually non-stationary, but a
          fixed linear combination of them is stationary.
    Then cross-check with the Hurst Exponent (script 03) on the same
    spread: two independent statistical routes should agree before you'd
    trust the result enough to trade it.

  JOB 2 -- "Given that a signal exists, how much should I bet, and when?"
    This is NOT what ADF/Hurst answer -- they say IF something reverts,
    not how violently it will move while doing so. That's a volatility
    question, so it's GARCH's (script 04) job: fit GARCH(1,1) to the
    spread's daily changes, and use the fitted conditional volatility to
    scale position size down in turbulent periods and up in calm ones --
    the same "long-run variance / half-life" machinery from script 04,
    now driving risk management instead of just being reported.

IMPORTANT CAVEAT: the backtest at the end is deliberately simple and is
here to illustrate the MECHANISM of combining these tools, not to claim a
profitable strategy. It: (a) fits GARCH in-sample (look-ahead bias -- a
real system would refit on a rolling/expanding window), (b) ignores
transaction costs and financing, (c) uses raw log-spread PnL rather than
proper dollar-neutral sizing. Treat the numbers as directional, not
investment advice.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

TICKER_A = "V"    # Visa
TICKER_B = "MA"   # Mastercard -- similar payments-network economics,
                   # a standard textbook example of a plausibly
                   # cointegrated pair (see Ernie Chan's pairs-trading books)


def download_prices(ticker_a, ticker_b, period="5y"):
    import yfinance as yf
    data = yf.download([ticker_a, ticker_b], period=period, progress=False, auto_adjust=True)["Close"].dropna()
    return data[ticker_a].to_numpy(), data[ticker_b].to_numpy(), data.index


def hurst_on_levels(x, lags=None):
    """
    Same variance-scaling calculation as 03_hurst_exponent.py's
    hurst_exponent(), but WITHOUT the internal np.log() step. A spread
    from an Engle-Granger regression is already a difference of logs
    (and can be negative, centred near 0), so it should be treated
    directly as "the level" being tested -- taking log() of it again
    would be nonsensical (and would crash on negative values).
    """
    x = np.asarray(x, dtype=float)
    if lags is None:
        lags = range(2, max(20, len(x) // 10))
    tau_values = np.array(list(lags))
    var_values = np.array([np.var(x[tau:] - x[:-tau]) for tau in tau_values])
    poly = np.polyfit(np.log(tau_values), np.log(var_values), 1)
    return poly[0] / 2.0


def job1_cointegration_check(price_a, price_b, name_a, name_b):
    print("=" * 70)
    print(f"JOB 1: is {name_a}/{name_b} a tradeable mean-reverting pair?")
    print("=" * 70)

    log_a, log_b = np.log(price_a), np.log(price_b)

    # Step 1 (Engle-Granger): OLS hedge ratio from log_a = alpha + beta*log_b
    X = np.column_stack([np.ones_like(log_b), log_b])
    (intercept, hedge_ratio), *_ = np.linalg.lstsq(X, log_a, rcond=None)
    spread = log_a - hedge_ratio * log_b - intercept

    print(f"hedge ratio (beta): {name_a} = {intercept:.3f} + {hedge_ratio:.3f} * {name_b}")
    print(f"spread = log({name_a}) - {hedge_ratio:.3f}*log({name_b}) - {intercept:.3f}")
    print(f"spread stats: mean={spread.mean():.4f}, std={spread.std():.4f}, n={len(spread)}\n")

    # Step 2: ADF test on the spread (p=1, matching script 02's setup)
    adf_stat, pvalue, _, _, crit = adfuller(spread, maxlag=1, regression="c", autolag=None)
    print(f"ADF on spread: stat={adf_stat:.4f}  p-value={pvalue:.4f}  "
          f"5% critical={crit['5%']:.3f}")
    adf_rejects = adf_stat < crit["5%"]
    print(f"  -> {'REJECTS' if adf_rejects else 'does NOT reject'} the random-walk null at 5%\n")

    # Half-life: reuse the ADF regression's own gamma_hat as theta, exactly
    # like script 01's OU half-life = ln(2)/theta.
    dspread = np.diff(spread)
    X2 = np.column_stack([np.ones(len(spread) - 1), spread[:-1]])
    (const, gamma_hat), *_ = np.linalg.lstsq(X2, dspread, rcond=None)
    theta = -gamma_hat
    half_life = np.log(2) / theta if theta > 0 else np.inf
    print(f"implied mean-reversion speed theta = {theta:.4f}  ->  half-life = {half_life:.1f} trading days\n")

    # Hurst Exponent on the same spread, as an independent check
    H = hurst_on_levels(spread)
    hurst_says_reverting = H < 0.45
    print(f"Hurst Exponent on spread: H={H:.3f}  "
          f"({'mean-reverting' if hurst_says_reverting else 'not clearly mean-reverting'})\n")

    tradeable = adf_rejects and hurst_says_reverting
    print(f"VERDICT: ADF and Hurst {'AGREE' if adf_rejects == hurst_says_reverting else 'DISAGREE'} -- "
          f"{'this spread looks tradeable.' if tradeable else 'skip this pair, or investigate further.'}\n")

    return spread, hedge_ratio, tradeable, half_life


def build_zscore_signal(spread, window=60, entry_z=1.5):
    """
    Classic mean-reversion signal: standardize the spread against its own
    rolling mean/std, go short the spread when z is high (expect it to
    fall back), long when z is low, and flatten once it crosses back
    through zero. This is the JOB 1 output turned into trade decisions.
    """
    s = pd.Series(spread)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
    z = (s - roll_mean) / roll_std

    position = np.zeros(len(s))
    pos = 0
    for i in range(len(s)):
        zi = z.iloc[i]
        if np.isnan(zi):
            position[i] = 0
            continue
        if pos == 0:
            if zi > entry_z:
                pos = -1
            elif zi < -entry_z:
                pos = 1
        elif pos == 1 and zi >= 0:
            pos = 0
        elif pos == -1 and zi <= 0:
            pos = 0
        position[i] = pos
    return position, z.to_numpy()


def job2_garch_position_sizing(spread, position):
    """
    JOB 2: fit GARCH(1,1) to the spread's daily changes and use the
    fitted conditional volatility to scale position size -- smaller bets
    when the spread is turbulent, larger when it's calm, targeting
    roughly constant risk per trade instead of a fixed unit size.
    """
    print("=" * 70)
    print("JOB 2: GARCH-based position sizing on the spread's daily changes")
    print("=" * 70)

    dspread = np.diff(spread, prepend=spread[0])

    # arch_model wants a non-trivial scale for numerical stability.
    am = arch_model(pd.Series(dspread[1:]) * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    sigma = np.concatenate([[np.nan], np.asarray(res.conditional_volatility) / 100])

    persistence = res.params["alpha[1]"] + res.params["beta[1]"]
    print(f"fitted GARCH(1,1) on spread changes: alpha+beta (persistence) = {persistence:.3f}")
    print(f"(same interpretation as script 04: how long a turbulent period in the")
    print(f"spread itself, not just the underlying prices, is expected to last)\n")

    # Inverse-volatility sizing, normalized around the typical volatility
    # level so the average bet size stays close to 1 (comparable to the
    # static unit-size strategy), then softly clipped to avoid extreme
    # leverage on the rare very-calm day.
    target_vol = np.nanmedian(sigma)
    scale = np.clip(target_vol / sigma, 0.2, 3.0)
    scale = np.nan_to_num(scale, nan=1.0)

    pnl_static = position * dspread
    pnl_dynamic = position * dspread * scale

    return pnl_static, pnl_dynamic, sigma, scale


def summarize_backtest(pnl_static, pnl_dynamic, burn_in=60):
    def stats(pnl):
        pnl = pnl[burn_in:]
        ann_ret = np.nanmean(pnl) * 252
        ann_vol = np.nanstd(pnl) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        return ann_ret, ann_vol, sharpe

    static_stats = stats(pnl_static)
    dynamic_stats = stats(pnl_dynamic)

    table = pd.DataFrame(
        [static_stats, dynamic_stats],
        columns=["annualized return", "annualized vol", "sharpe (naive)"],
        index=["static unit size", "GARCH-vol-scaled"],
    )
    print("=" * 70)
    print("ILLUSTRATIVE BACKTEST: static sizing vs. GARCH-scaled sizing")
    print("=" * 70)
    print(table.to_string())
    print(
        "\nThe point to take from this table is the VOLATILITY column, not\n"
        "the Sharpe ratio: GARCH-scaled sizing should show lower annualized\n"
        "volatility than static sizing (smaller bets in turbulent regimes),\n"
        "even though it uses the exact same entry/exit signal. Whether that\n"
        "translates into a better Sharpe ratio here depends on whether this\n"
        "pair's actual reversion tends to happen during high- or low-\n"
        "volatility periods -- which is precisely the kind of question a\n"
        "real (walk-forward, cost-aware) backtest would need to answer\n"
        "before trading real money on it. This demo is about the mechanism,\n"
        "not a claim that this specific setup is profitable.\n"
    )
    return table


def main():
    try:
        price_a, price_b, dates = download_prices(TICKER_A, TICKER_B)
    except Exception as e:
        print(f"Could not download real data ({e}); this script needs network access.")
        return

    spread, hedge_ratio, tradeable, half_life = job1_cointegration_check(
        price_a, price_b, TICKER_A, TICKER_B
    )

    position, zscore = build_zscore_signal(spread, window=60, entry_z=1.5)
    pnl_static, pnl_dynamic, sigma, scale = job2_garch_position_sizing(spread, position)
    summarize_backtest(pnl_static, pnl_dynamic)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(spread, color="tab:purple", linewidth=0.8)
    axes[0].set_title(f"Spread = log({TICKER_A}) - {hedge_ratio:.3f}*log({TICKER_B})  "
                       f"(half-life ~{half_life:.0f} days)")

    axes[1].plot(zscore, color="tab:orange", linewidth=0.8)
    axes[1].axhline(1.5, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(-1.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Rolling z-score signal (entry at +/-1.5)")

    axes[2].plot(sigma, color="tab:red", linewidth=0.8)
    axes[2].set_title("GARCH(1,1) conditional volatility of spread changes")

    axes[3].plot(np.nancumsum(pnl_static), label="static unit size", color="tab:blue")
    axes[3].plot(np.nancumsum(pnl_dynamic), label="GARCH-vol-scaled", color="tab:green")
    axes[3].set_title("Cumulative illustrative PnL (log-spread units, no costs)")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig("examples/real_world_pairs_trading.png", dpi=120)
    print("Saved plot to examples/real_world_pairs_trading.png")


if __name__ == "__main__":
    main()
