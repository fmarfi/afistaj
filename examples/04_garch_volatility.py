"""
GARCH(1,1): calculating volatility clustering, by hand and by MLE
============================================================================

This one isn't from your note directly -- your note's ADF test and Hurst
Exponent both ask "does the PRICE LEVEL revert to a mean?" GARCH answers a
different, complementary question: "does the VOLATILITY of returns cluster
over time?" Real markets do both -- prices sometimes mean-revert (or
trend), and *volatility itself* goes through calm and turbulent regimes,
independent of the direction of price moves. Script 05 ties GARCH together
with the ADF/Hurst tools from scripts 01-03 into one realistic workflow.

The GARCH(1,1) model (Bollerslev, 1986) for a return series r_t:

    r_t     = mu + eps_t
    eps_t   = sigma_t * z_t,                      z_t ~ iid N(0, 1)
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

Reading it left to right, term by term:
  - eps_t is the "surprise" return (actual minus mean) -- what OU's dW_t
    was to price, eps_t is to return, except now we track its SIZE too.
  - sigma_t^2 (the conditional variance) is a weighted average of:
      omega        a baseline variance level,
      eps_{t-1}^2  yesterday's squared surprise (alpha = how much a big
                   move yesterday raises today's expected variance),
      sigma_{t-1}^2 yesterday's variance itself (beta = how persistent
                   volatility is, independent of any specific shock).
  - alpha + beta close to 1 means volatility shocks decay slowly (a
    turbulent period stays turbulent for a long time); alpha+beta < 1 is
    required for the process to have a finite, stable long-run variance
    at all (stationarity condition, directly analogous to the note's OU
    process needing theta > 0 for a stationary distribution to exist).

Exactly like 01_ornstein_uhlenbeck_process.py's
    stationary_var = sigma^2 / (2*theta)
GARCH(1,1) has a closed-form long-run (unconditional) variance:
    long_run_var = omega / (1 - alpha - beta)
and a half-life for a volatility shock to decay by half:
    half_life = ln(0.5) / ln(alpha + beta)
-- the same "how many days until this reverts to normal" question script 1
asked about price, now asked about variance instead.

This script:
  1. Simulates a GARCH(1,1) return series (Euler-style recursion, same
     spirit as the OU simulator).
  2. Implements the GARCH log-likelihood from scratch and fits
     omega/alpha/beta via scipy's MLE optimizer.
  3. Cross-checks the from-scratch fit against the `arch` package's
     GARCH(1,1) fit on the same data (same role statsmodels played for
     the ADF test in script 02).
  4. Fits GARCH to REAL daily returns (fetched live) to show volatility
     clustering isn't just a simulation artefact.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from arch import arch_model

RNG = np.random.default_rng(seed=17)


def simulate_garch11(omega, alpha, beta, n, mu=0.0, rng=RNG):
    """
    Simulate a GARCH(1,1) return series via direct recursion:
        sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
        eps_t     = sigma_t * z_t,   z_t ~ N(0,1)
        r_t       = mu + eps_t

    Start sigma_1^2 at the long-run variance so the series doesn't need a
    long burn-in period to reach its stationary volatility regime.
    """
    long_run_var = omega / (1 - alpha - beta)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    r = np.empty(n)

    sigma2[0] = long_run_var
    eps[0] = np.sqrt(sigma2[0]) * rng.normal()
    r[0] = mu + eps[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.normal()
        r[t] = mu + eps[t]

    return r, sigma2


def worked_example():
    """Spell out the sigma_t^2 recursion arithmetic for 5 steps by hand."""
    print("=" * 70)
    print("WORKED EXAMPLE: GARCH(1,1) variance recursion, 5 steps")
    print("=" * 70)
    omega, alpha, beta = 0.02, 0.10, 0.85
    long_run_var = omega / (1 - alpha - beta)
    print(f"omega={omega}, alpha={alpha}, beta={beta}  ->  long-run var = omega/(1-alpha-beta) = {long_run_var:.4f}\n")

    rng = np.random.default_rng(seed=3)
    sigma2_prev = long_run_var
    eps_prev = np.sqrt(sigma2_prev) * rng.normal()
    print(f"t=1: sigma2 = long-run var = {sigma2_prev:.4f}, eps_1 = sigma_1*z_1 = {eps_prev:+.4f}")

    for t in range(2, 6):
        sigma2_t = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
        eps_t = np.sqrt(sigma2_t) * rng.normal()
        print(
            f"t={t}: sigma2 = {omega} + {alpha}*({eps_prev:.4f})^2 + {beta}*{sigma2_prev:.4f} "
            f"= {omega:.4f} + {alpha*eps_prev**2:.4f} + {beta*sigma2_prev:.4f} = {sigma2_t:.4f}  |  "
            f"eps_{t} = sqrt({sigma2_t:.4f})*z = {eps_t:+.4f}"
        )
        sigma2_prev, eps_prev = sigma2_t, eps_t
    print(
        "\nNotice sigma2 reacts to the SIZE of the previous surprise\n"
        "(eps_prev^2), not its sign -- a big move either way raises the\n"
        "next period's expected variance. That's volatility clustering.\n"
    )


def garch11_neg_loglik(params, returns):
    """
    Negative log-likelihood of a GARCH(1,1) model with Normal innovations,
    to be MINIMIZED (equivalently: maximizes the likelihood of observing
    this exact return series under the given omega/alpha/beta/mu).

    For each t:  eps_t ~ N(0, sigma_t^2)
    log density: -0.5*(log(2*pi) + log(sigma_t^2) + eps_t^2/sigma_t^2)
    Total log-likelihood = sum over t; we return its negative.
    """
    omega, alpha, beta, mu = params
    eps = returns - mu
    n = len(eps)

    sigma2 = np.empty(n)
    sigma2[0] = np.var(eps)  # reasonable starting variance
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

    # Guard against a bad optimizer step producing non-positive variance.
    if np.any(sigma2 <= 0):
        return np.inf

    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    return -ll


def fit_garch11_from_scratch(returns):
    """
    Maximum-likelihood fit of omega/alpha/beta/mu via scipy's constrained
    optimizer, minimizing garch11_neg_loglik. Bounds enforce omega>0,
    0<=alpha, 0<=beta (stationarity, alpha+beta<1, is checked afterwards
    rather than enforced as a hard constraint, to keep the optimizer simple).
    """
    x0 = [np.var(returns) * 0.05, 0.05, 0.90, np.mean(returns)]
    bounds = [(1e-8, None), (0.0, 1.0), (0.0, 1.0), (None, None)]

    result = minimize(
        garch11_neg_loglik, x0, args=(returns,),
        method="L-BFGS-B", bounds=bounds,
    )
    omega, alpha, beta, mu = result.x
    return omega, alpha, beta, mu, result


def cross_check_with_arch_package(returns):
    """
    Fit the SAME return series with the `arch` package's battle-tested
    GARCH(1,1) implementation, so the from-scratch MLE numbers above have
    an independent reference point -- the same role statsmodels' adfuller
    played for the from-scratch ADF calculation in 02_adf_test.py.
    """
    # arch_model expects returns in percent-ish scale for numerical
    # stability; rescale by 100 and undo it afterwards.
    am = arch_model(returns * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    omega = res.params["omega"] / 100 ** 2
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    mu = res.params["mu"] / 100
    return omega, alpha, beta, mu, res


def report_fit(name, omega, alpha, beta, mu):
    persistence = alpha + beta
    long_run_var = omega / (1 - persistence) if persistence < 1 else np.nan
    long_run_vol = np.sqrt(long_run_var) if not np.isnan(long_run_var) else np.nan
    half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan
    return {
        "fit": name,
        "omega": round(omega, 6),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "alpha+beta (persistence)": round(persistence, 4),
        "long-run daily vol": round(long_run_vol, 5) if not np.isnan(long_run_vol) else None,
        "half-life (days)": round(half_life, 1) if not np.isnan(half_life) else None,
    }


def simulated_data_demo():
    print("=" * 70)
    print("FIT CHECK: from-scratch MLE vs. the `arch` package, on SIMULATED data")
    print("=" * 70)
    true_omega, true_alpha, true_beta = 0.00003, 0.08, 0.90
    returns, sigma2_true = simulate_garch11(true_omega, true_alpha, true_beta, n=2000, mu=0.0003)

    scratch_omega, scratch_alpha, scratch_beta, scratch_mu, opt_result = fit_garch11_from_scratch(returns)
    lib_omega, lib_alpha, lib_beta, lib_mu, arch_result = cross_check_with_arch_package(returns)

    rows = [
        {"fit": "TRUE (used to simulate)", "omega": true_omega, "alpha": true_alpha,
         "beta": true_beta, "alpha+beta (persistence)": round(true_alpha + true_beta, 4),
         "long-run daily vol": round(np.sqrt(true_omega / (1 - true_alpha - true_beta)), 5),
         "half-life (days)": round(np.log(0.5) / np.log(true_alpha + true_beta), 1)},
        report_fit("from-scratch MLE", scratch_omega, scratch_alpha, scratch_beta, scratch_mu),
        report_fit("arch package", lib_omega, lib_alpha, lib_beta, lib_mu),
    ]
    table = pd.DataFrame(rows).set_index("fit")
    print(table.to_string())
    print(
        "\nBoth fitting methods should land close to the TRUE parameters\n"
        "used to simulate the data, and close to each other -- confirming\n"
        "the from-scratch negative log-likelihood is implemented correctly.\n"
        f"(scipy optimizer converged: {opt_result.success})\n"
    )
    return returns, sigma2_true


def real_data_demo():
    """
    Fit GARCH(1,1) to real daily returns to show volatility clustering is
    an empirical market feature, not just something that falls out of a
    simulation. Downloads live data -- if that fails (no network), this
    step is skipped gracefully rather than crashing the whole script.
    """
    print("=" * 70)
    print("REAL DATA: fitting GARCH(1,1) to daily log returns")
    print("=" * 70)
    try:
        import yfinance as yf
        prices = yf.download("SPY", period="3y", progress=False, auto_adjust=True)["Close"].dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna().to_numpy().flatten()
    except Exception as e:
        print(f"Could not download real data ({e}); skipping real-data demo.\n")
        return None, None

    omega, alpha, beta, mu, res = cross_check_with_arch_package(log_returns)
    row = report_fit("SPY daily returns (arch package)", omega, alpha, beta, mu)
    print(pd.DataFrame([row]).set_index("fit").to_string())
    print(
        f"\nalpha+beta = {alpha+beta:.3f} means real market volatility shocks\n"
        f"decay with a half-life of roughly {row['half-life (days)']} trading days --\n"
        "compare that to the OU half-life in script 01: exactly the same\n"
        "concept (ln(2)/rate), just applied to variance instead of price.\n"
    )
    # res.conditional_volatility is on the *100-rescaled fitting scale
    # (see cross_check_with_arch_package's comment) -- divide back down
    # before squaring, or this plots ~100x too large.
    conditional_vol = np.asarray(res.conditional_volatility) / 100
    return log_returns, conditional_vol ** 2


def main():
    worked_example()
    sim_returns, sim_sigma2 = simulated_data_demo()
    real_returns, real_sigma2 = real_data_demo()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(sim_returns, linewidth=0.6, color="tab:blue")
    axes[0, 0].set_title("Simulated GARCH(1,1) returns")
    axes[0, 1].plot(np.sqrt(sim_sigma2), color="tab:red")
    axes[0, 1].set_title("Simulated conditional volatility (sigma_t)")

    if real_returns is not None:
        axes[1, 0].plot(real_returns, linewidth=0.6, color="tab:blue")
        axes[1, 0].set_title("SPY daily log returns")
        axes[1, 1].plot(np.sqrt(real_sigma2), color="tab:red")
        axes[1, 1].set_title("SPY fitted conditional volatility (arch package)")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("examples/garch_volatility_clustering.png", dpi=120)
    print("Saved plot to examples/garch_volatility_clustering.png")


if __name__ == "__main__":
    main()
