"""
Hurst Exponent: calculating it from the variance-scaling relationship
============================================================================

Reference note: "Statistical Mean Reversion Testing.md", section
"Hurst Exponent".

The note's starting point: for a Geometric Brownian Motion, the variance
of the log-price difference over a lag tau scales LINEARLY with tau:

    <|log(t+tau) - log(t)|^2> ~ tau

If autocorrelations exist (successive moves aren't independent), that
linear relationship breaks and gets an exponent:

    <|log(t+tau) - log(t)|^2> ~ tau^(2H)

Take logs of both sides and this becomes a straight line:

    log(Var(tau)) = 2H * log(tau) + constant

So H is HALF the slope of a log-log plot of variance-at-lag-tau against
tau. That's the actual calculation this script performs: for a range of
tau values, compute Var(tau) empirically from the series, fit a straight
line to (log(tau), log(Var(tau))) with linear regression, and H = slope/2.

Interpretation from the note:
    H < 0.5  -> mean reverting  (H near 0   = strongly mean reverting)
    H = 0.5  -> geometric Brownian motion / random walk
    H > 0.5  -> trending        (H near 1.0 = strongly trending)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=99)


def hurst_exponent(series, lags=None):
    """
    Estimate H via the variance-scaling method described above.

    For each lag tau in `lags`:
      1. Build the lagged difference series: log(x[tau:]) - log(x[:-tau])
         (the note works with log price, which is why we take np.log first
         -- this also matches the standard reference implementation of
         this calculation).
      2. Var(tau) = variance of that lagged-difference series.

    Then fit log(Var(tau)) = 2H*log(tau) + c via least-squares (np.polyfit
    degree 1), and H = slope / 2.

    IMPORTANT CAVEAT (see lag_sensitivity_demo() below for a live example):
    this two-point variance estimator has known finite-sample bias that
    grows with max(tau) relative to len(series) -- large tau leaves few
    effectively-independent windows to estimate Var(tau) from, which
    biases the fitted slope (and hence H) downward, making even a true
    random walk look mean-reverting. The rule of thumb used here is to
    cap max(tau) at roughly len(series)/10.

    Returns (H, tau_values, var_values, poly_coeffs) so the caller can
    both report H and plot the log-log fit that produced it.
    """
    x = np.asarray(series, dtype=float)
    log_x = np.log(x)

    if lags is None:
        max_lag = max(20, len(x) // 10)
        lags = range(2, max_lag)

    tau_values = np.array(list(lags))
    var_values = np.array([
        np.var(log_x[tau:] - log_x[:-tau])
        for tau in tau_values
    ])

    # log(Var(tau)) = 2H * log(tau) + c  ->  linear fit, slope = 2H
    poly = np.polyfit(np.log(tau_values), np.log(var_values), 1)
    slope = poly[0]
    H = slope / 2.0
    return H, tau_values, var_values, poly


def worked_example():
    """
    Show the calculation for a single lag (tau=5) on a small synthetic
    series, spelling out each step: log prices, lagged differences,
    variance -- the one number that feeds the log-log regression.
    """
    print("=" * 70)
    print("WORKED EXAMPLE: Var(tau) for tau=5 on a 20-point series")
    print("=" * 70)

    x = 100 * np.exp(np.cumsum(RNG.normal(scale=0.01, size=20)))
    log_x = np.log(x)
    tau = 5

    diffs = log_x[tau:] - log_x[:-tau]
    print(f"price series (first 6): {np.round(x[:6], 3)}")
    print(f"log(price) (first 6):   {np.round(log_x[:6], 4)}")
    print(f"\nlog(x[tau:]) - log(x[:-tau]) for tau={tau}:")
    print(f"  = {np.round(diffs, 4)}")
    print(f"\nVar(tau={tau}) = variance of those {len(diffs)} differences = {np.var(diffs):.6f}")
    print(
        "\nRepeating this for many tau values, then fitting a line to\n"
        "log(tau) vs log(Var(tau)), gives slope = 2H -- that's the full\n"
        "Hurst Exponent calculation.\n"
    )


def lag_sensitivity_demo(series, name="random walk"):
    """
    Demonstrate the finite-sample bias flagged in hurst_exponent()'s
    docstring: estimate H on the SAME series using progressively larger
    max-tau cutoffs (as a fraction of series length) and show H drifting
    away from the "true" value as max(tau)/len(series) grows.

    This matters practically: if you pick max_lag carelessly (e.g. "use
    lags up to 100" regardless of how much history you have), you can
    manufacture a spurious mean-reverting signal out of a plain random
    walk. Always report the (tau range, N) alongside any H estimate.
    """
    print("=" * 70)
    print(f"CAVEAT DEMO: H estimate vs. max-lag choice, on '{name}'")
    print("=" * 70)
    n = len(series)
    fractions = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
    rows = []
    for frac in fractions:
        max_lag = max(3, int(n * frac))
        H, _, _, _ = hurst_exponent(series, lags=range(2, max_lag))
        rows.append({
            "max(tau) as % of N": f"{frac*100:.0f}%",
            "max(tau)": max_lag,
            "H estimate": round(H, 3),
        })
    table = pd.DataFrame(rows).set_index("max(tau) as % of N")
    print(table.to_string())
    print(
        "\nFor a true random walk, H should stay near 0.5 regardless of\n"
        "max(tau) -- the fact that it visibly drifts downward as max(tau)\n"
        "grows is exactly the estimator bias, not a change in the series.\n"
        "This is why hurst_exponent() defaults max(tau) to len(series)/10.\n"
    )


def summarize_and_plot(series_dict):
    """Compute H for each named series, print a summary table, and plot the log-log fits."""
    rows = []
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, series in series_dict.items():
        # lags=None -> hurst_exponent() picks max(tau) = len(series)/10,
        # the bias-aware default demonstrated in lag_sensitivity_demo().
        H, tau_values, var_values, poly = hurst_exponent(series, lags=None)

        if H < 0.4:
            regime = "mean reverting"
        elif H > 0.6:
            regime = "trending"
        else:
            regime = "~random walk (GBM-like)"

        rows.append({"series": name, "H": round(H, 3), "regime": regime})

        # log-log scatter + fitted line, one per series
        log_tau = np.log(tau_values)
        log_var = np.log(var_values)
        ax.scatter(log_tau, log_var, s=8, label=f"{name} (H={H:.2f})")
        fitted = np.polyval(poly, log_tau)
        ax.plot(log_tau, fitted, linewidth=1)

    ax.set_xlabel("log(tau)")
    ax.set_ylabel("log(Var(tau))")
    ax.set_title("Hurst Exponent: log-log variance-scaling fits (slope = 2H)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("examples/hurst_exponent_fit.png", dpi=120)
    print("Saved plot to examples/hurst_exponent_fit.png\n")

    return pd.DataFrame(rows).set_index("series")


def main():
    worked_example()

    # Reuse the mean-reverting / random-walk / trending series from
    # 01_ornstein_uhlenbeck_process.py so H can be checked against the
    # ADF test's verdict on the exact same data.
    data = np.load("examples/ou_example_series.npz")

    # hurst_exponent() takes np.log(series) internally, so series values
    # must be strictly positive -- shift the OU series (centred near 0-ish
    # fluctuations around 100) up if needed. They're centred at 100 with
    # small sigma, so they're already comfortably positive.
    ou_series = {
        "mean-reverting (theta=0.05)": data["mean_reverting"],
        "random walk (theta=0)": data["random_walk"],
        "trending (momentum, rho=0.8)": data["trending"],
    }

    # Add a "pure" Geometric Brownian Motion as a clean H=0.5 reference,
    # since the note explicitly derives the tau-scaling law from GBM.
    n = 500
    gbm_returns = RNG.normal(loc=0.0002, scale=0.01, size=n)
    gbm = 100 * np.exp(np.cumsum(gbm_returns))
    ou_series["pure GBM (reference, H should be ~0.5)"] = gbm

    lag_sensitivity_demo(data["random_walk"], name="random walk (theta=0)")

    print("=" * 70)
    print("HURST EXPONENT across example series")
    print("=" * 70)
    table = summarize_and_plot(ou_series)
    print(table.to_string())
    print(
        "\nCross-check against 02_adf_test.py:\n"
        "  - 'mean-reverting' should have H well below 0.5, AND the ADF\n"
        "    test should have rejected its random-walk null -- two\n"
        "    different statistical routes agreeing on the same conclusion.\n"
        "  - 'random walk' and 'pure GBM' should sit close to H=0.5.\n"
        "  - 'trending' should have H above 0.5, despite the ADF test also\n"
        "    failing to reject its null -- ADF tests specifically for a\n"
        "    unit root/mean reversion, while H also distinguishes trending\n"
        "    behaviour from a flat random walk; the two tools answer\n"
        "    related but not identical questions.\n"
    )


if __name__ == "__main__":
    main()
