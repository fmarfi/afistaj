"""
Augmented Dickey-Fuller (ADF) test: calculating the test statistic by hand
============================================================================

Reference note: "Statistical Mean Reversion Testing.md", section
"Augmented Dickey Fuller (ADF) Test".

The note's regression (with p = 1, the lag order the note says is
"usually sufficient for trading research", and beta = 0 since drift is
assumed negligible next to short-term noise) is:

    delta_y_t = alpha + gamma * y_{t-1} + epsilon_t

where delta_y_t = y_t - y_{t-1}. This is just an OLS regression of the
*change* in the series on its own *previous level*. The intuition:
  - If the series is mean-reverting, a high y_{t-1} should be followed by
    a *negative* delta_y_t (it gets pulled back down) -> gamma < 0.
  - If the series is a random walk, y_{t-1} carries no information about
    the next change -> gamma = 0.

The test statistic is:

    DF_tau = gamma_hat / SE(gamma_hat)

which looks exactly like a t-statistic -- but it ISN'T one. Dickey and
Fuller showed the usual Student-t distribution doesn't apply here (because
y_{t-1} is not a fixed regressor, it's a lagged version of a stochastic
series), so they derived their own critical-value distribution by
simulation. That's why the note stresses: the statistic must be MORE
NEGATIVE than the critical value to reject the random-walk null, and why
we can't just look up a stats textbook's t-table.

This script:
  1. Implements the DF_tau calculation from scratch with plain OLS, so
     every term in the formula is visible.
  2. Cross-checks the from-scratch number against statsmodels' `adfuller`
     (which also supplies the correct Dickey-Fuller critical values,
     since deriving those by simulation ourselves is out of scope here).
  3. Runs both on the three series saved by 01_ornstein_uhlenbeck_process.py.
  4. Adds the classic trading use case the note flags: testing whether a
     SPREAD between two prices is mean-reverting (the basis of pairs
     trading), rather than testing a single price series.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

RNG = np.random.default_rng(seed=7)


def adf_test_statistic_from_scratch(series, p=1):
    """
    Compute DF_tau by hand via OLS on:
        delta_y_t = alpha + gamma * y_{t-1} + delta_1*delta_y_{t-1} + ... + eps_t

    p is the note's "lag model of order p": it contributes (p-1) lagged
    difference terms alongside y_{t-1}. p=1 (the note's recommended
    default for trading research) means NO extra lagged-difference terms
    -- just the plain delta_y_t = alpha + gamma*y_{t-1} + eps_t regression.

    Returns (gamma_hat, se_gamma_hat, df_tau).
    """
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)  # delta_y_t = y_t - y_{t-1}, one shorter than y

    # Build the design matrix. Row t (indexing into dy) needs:
    # [1, y_t, dy_{t-1}, dy_{t-2}, ..., dy_{t-p+1}], which requires
    # t-p+1 >= 0, i.e. t >= p-1.
    n = len(dy)
    start = p - 1  # first usable row once (p-1) lagged differences exist
    rows = n - start

    X = np.ones((rows, 2 + (p - 1)))  # intercept + y_{t-1} + (p-1) lagged diffs
    target = np.empty(rows)

    for i, t in enumerate(range(start, n)):
        target[i] = dy[t]                 # delta_y_t
        X[i, 1] = y[t]                    # y_{t-1}  (note: dy index t == level index t+1-1 = t, see below)
        for lag in range(1, p):
            X[i, 1 + lag] = dy[t - lag]   # delta_y_{t-lag}

    # OLS: beta_hat = (X'X)^-1 X'target
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ target

    residuals = target - X @ beta_hat
    dof = rows - X.shape[1]
    sigma2 = (residuals @ residuals) / dof
    cov_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov_beta))

    gamma_hat = beta_hat[1]
    se_gamma_hat = se[1]
    df_tau = gamma_hat / se_gamma_hat
    return gamma_hat, se_gamma_hat, df_tau


def worked_example():
    """Show the OLS design matrix and DF_tau arithmetic on a tiny 8-point series."""
    print("=" * 70)
    print("WORKED EXAMPLE: DF_tau on a toy 8-point series, p=1")
    print("=" * 70)
    y = np.array([10.0, 10.8, 9.9, 10.5, 9.7, 10.3, 9.6, 10.4])
    print(f"y = {y}")

    gamma_hat, se_gamma_hat, df_tau = adf_test_statistic_from_scratch(y, p=1)
    print(f"gamma_hat (coefficient on y_t-1)       = {gamma_hat:.4f}")
    print(f"SE(gamma_hat)                          = {se_gamma_hat:.4f}")
    print(f"DF_tau = gamma_hat / SE(gamma_hat)      = {df_tau:.4f}")
    print(
        "\ngamma_hat < 0 here: each step's change tends to reverse the prior\n"
        "level's deviation, which is the fingerprint of mean reversion --\n"
        "but with only 8 points this is far too small a sample to draw any\n"
        "real conclusion; it's here to show the arithmetic, not a verdict.\n"
    )


DF_CRITICAL_VALUES_NOTE = (
    "Critical values below (1%/5%/10%) come from statsmodels, which uses\n"
    "MacKinnon's tabulated approximation to the Dickey-Fuller distribution\n"
    "the note mentions -- reproducing that simulation-derived table by hand\n"
    "is not practical, so this is the one piece we don't rebuild from scratch."
)


def run_and_report(name, series):
    """Run both the from-scratch calculation and statsmodels' adfuller, side by side."""
    gamma_hat, se_gamma_hat, df_tau_manual = adf_test_statistic_from_scratch(series, p=1)

    # maxlag=1, regression="c" (constant only, no trend) matches the note's
    # p=1, beta=0 simplification exactly, so the manual and library numbers
    # should line up closely.
    # NOTE: with autolag=None, adfuller returns 5 values (no icbest) --
    # icbest is only included when autolag picks the lag automatically.
    result = adfuller(series, maxlag=1, regression="c", autolag=None)
    adf_stat_lib, pvalue, usedlag, nobs, crit_values = result

    print(f"--- {name} ---")
    print(f"  from-scratch DF_tau : {df_tau_manual:.4f}   (gamma_hat={gamma_hat:.4f}, SE={se_gamma_hat:.4f})")
    print(f"  statsmodels ADF stat: {adf_stat_lib:.4f}   (p-value={pvalue:.4f})")
    print(f"  critical values      1%: {crit_values['1%']:.3f}  5%: {crit_values['5%']:.3f}  10%: {crit_values['10%']:.3f}")

    verdict_level = None
    for level in ("1%", "5%", "10%"):
        if adf_stat_lib < crit_values[level]:
            verdict_level = level
            break

    if verdict_level:
        print(f"  verdict: ADF stat is MORE NEGATIVE than the {verdict_level} critical value")
        print(f"           -> reject the random-walk null at {verdict_level} -> series looks mean-reverting.")
    else:
        print("  verdict: ADF stat is NOT more negative than even the 10% critical value")
        print("           -> cannot reject the random-walk null -> no evidence of mean reversion.")
    print()
    return {
        "series": name,
        "from-scratch DF_tau": round(df_tau_manual, 4),
        "statsmodels ADF stat": round(adf_stat_lib, 4),
        "p-value": round(pvalue, 4),
        "5% critical value": round(crit_values["5%"], 4),
        "reject random walk @5%": adf_stat_lib < crit_values["5%"],
    }


def pairs_trading_spread_example():
    """
    The note's key practical point: traders rarely run the ADF test on a
    single price series (real prices usually behave like random walks).
    Instead, they build a SPREAD between two related instruments and test
    THAT for mean reversion -- the basis of statistical arbitrage / pairs
    trading (e.g. two cointegrated stocks, or a stock vs. a synthetic hedge).

    Here we simulate that setup directly:
      - asset_a: a random walk (a stand-in for a real, non-stationary price).
      - asset_b: asset_a plus a mean-reverting OU noise term -- i.e. by
        construction, asset_a and asset_b are cointegrated with hedge
        ratio 1, so spread = asset_b - asset_a is stationary/mean-reverting
        even though neither asset_a nor asset_b is, individually.
    """
    print("=" * 70)
    print("TRADING APPLICATION: ADF test on a pairs-trading spread")
    print("=" * 70)

    n = 500
    innovations = RNG.normal(scale=1.0, size=n)
    asset_a = 100 + np.cumsum(innovations)  # random walk "price"

    # OU-distributed noise added on top of asset_a to build asset_b, so the
    # spread (asset_b - asset_a) is exactly that OU noise -> mean-reverting.
    ou_noise = np.empty(n)
    ou_noise[0] = 0.0
    theta, mu, sigma = 0.1, 0.0, 0.5
    for t in range(1, n):
        ou_noise[t] = ou_noise[t - 1] + theta * (mu - ou_noise[t - 1]) + sigma * RNG.normal()
    asset_b = asset_a + 5 + ou_noise

    spread = asset_b - asset_a

    print("asset_a: random walk price series (non-stationary by construction)")
    print("asset_b: asset_a + constant + OU noise (so it tracks asset_a closely)")
    print("spread = asset_b - asset_a  ->  should isolate the OU noise, mean-reverting\n")

    rows = [
        run_and_report("asset_a (random walk)", asset_a),
        run_and_report("asset_b (random walk + OU-linked)", asset_b),
        run_and_report("spread = asset_b - asset_a", spread),
    ]
    return pd.DataFrame(rows).set_index("series")


def main():
    worked_example()
    print(DF_CRITICAL_VALUES_NOTE + "\n")

    print("=" * 70)
    print("ADF TEST on the three series from 01_ornstein_uhlenbeck_process.py")
    print("=" * 70)
    data = np.load("examples/ou_example_series.npz")
    rows = [
        run_and_report("mean-reverting (theta=0.05)", data["mean_reverting"]),
        run_and_report("random walk (theta=0)", data["random_walk"]),
        run_and_report("trending (momentum, rho=0.8)", data["trending"]),
    ]
    summary1 = pd.DataFrame(rows).set_index("series")

    summary2 = pairs_trading_spread_example()

    print("=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)
    print(pd.concat([summary1, summary2]).to_string())
    print(
        "\nExpected pattern: the 'mean-reverting' series and the pairs-trading\n"
        "'spread' should show large-negative ADF stats (reject random\n"
        "walk); 'random walk' and 'trending' should not -- and individually,\n"
        "asset_a/asset_b (both random walks) should NOT reject either, even\n"
        "though their spread does. That gap between the individual series\n"
        "and their spread is precisely why pairs trading works: you don't\n"
        "need either leg to be mean-reverting, only the combination.\n"
    )


if __name__ == "__main__":
    main()
