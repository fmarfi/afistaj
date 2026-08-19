"""
Ornstein-Uhlenbeck process: simulating mean reversion from first principles
============================================================================

Reference note: "Statistical Mean Reversion Testing.md", section
"Testing for Mean Reversion".

The note gives the continuous-time SDE for a mean-reverting series:

    dx_t = theta * (mu - x_t) * dt + sigma * dW_t

Reading it left to right:
  - theta (speed of reversion): how hard the process gets pulled back
    towards mu. theta = 0 means "no pull at all" -> the process becomes
    a pure random walk (Geometric/Arithmetic Brownian Motion).
  - mu (long-run mean): the level the process is attracted to.
  - sigma (volatility): the size of the random shocks.
  - dW_t (Wiener increment): the note's callout explains this is the
    piece with mean 0 and std-dev sqrt(dt) -- NOT dt. That sqrt(dt)
    scaling is exactly what we implement below with sigma * sqrt(dt) * Z.

There's no closed-form path for a stochastic process (only a closed-form
*distribution*), so "calculating" this SDE in practice means discretising
it and stepping forward -- the Euler-Maruyama scheme:

    x_{t+dt} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * Z,   Z ~ N(0, 1)

This script builds that simulator, then uses it to generate three example
series (mean-reverting, random walk, trending) that the next two scripts
(02_adf_test.py, 03_hurst_exponent.py) will test statistically.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: we save PNGs instead of popping a window
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=42)  # fixed seed -> reproducible example


def simulate_momentum_process(x0, rho, sigma, n_steps, rng=RNG):
    """
    A genuinely "trending" series, built the way 03_hurst_exponent.py's
    theory actually requires: POSITIVE AUTOCORRELATION between successive
    increments (momentum), not just a constant deterministic drift.

    The note is explicit about this: the tau^(2H) scaling law only departs
    from tau^1 "if any autocorrelations exist (i.e. any sequential price
    movements possess non-zero correlation)". A random walk plus a fixed
    per-step drift does NOT introduce that -- its increments are still
    i.i.d., just with a non-zero mean, so it still measures H ~ 0.5. What
    actually produces H > 0.5 is increments that depend on the previous
    increment, e.g. an AR(1) on the increments themselves:

        increment_t = rho * increment_{t-1} + sigma * Z_t,   Z_t ~ N(0,1)
        x_t = x_{t-1} + increment_t

    rho > 0 means "a step in one direction makes the next step in the
    same direction more likely" -- literal price momentum -- and the
    cumulative sum of that persistent process is what shows up as
    H > 0.5 under the variance-scaling calculation.
    """
    increments = np.empty(n_steps)
    increments[0] = rng.normal(scale=sigma)
    for t in range(1, n_steps):
        increments[t] = rho * increments[t - 1] + rng.normal(scale=sigma)

    x = np.empty(n_steps + 1)
    x[0] = x0
    x[1:] = x0 + np.cumsum(increments)
    return x


def simulate_ou_process(x0, theta, mu, sigma, dt, n_steps, rng=RNG):
    """
    Euler-Maruyama simulation of dx_t = theta*(mu - x_t)*dt + sigma*dW_t.

    Parameters
    ----------
    x0     : starting value of the series
    theta  : speed of mean reversion (0 = no reversion = random walk)
    mu     : long-run mean the process reverts to
    sigma  : volatility of the random shocks
    dt     : size of each time step (e.g. 1.0 = one day per step)
    n_steps: number of steps to simulate

    Returns
    -------
    numpy array of length n_steps + 1 (including x0)
    """
    x = np.empty(n_steps + 1)
    x[0] = x0

    for t in range(n_steps):
        # dW_t has std-dev sqrt(dt), NOT dt -- this is the note's key point.
        dW = rng.normal(loc=0.0, scale=np.sqrt(dt))
        drift = theta * (mu - x[t]) * dt
        diffusion = sigma * dW
        x[t + 1] = x[t] + drift + diffusion

    return x


def worked_example():
    """
    Walk through the calculation step-by-step for the first 5 steps of a
    mean-reverting series, printing every term of the update equation so
    the arithmetic behind the simulation loop is visible.
    """
    print("=" * 70)
    print("WORKED EXAMPLE: first 5 steps of the Euler-Maruyama update")
    print("=" * 70)

    x0, theta, mu, sigma, dt = 100.0, 0.15, 100.0, 1.0, 1.0
    rng = np.random.default_rng(seed=1)
    x_prev = x0
    print(f"x0={x0}, theta={theta}, mu={mu}, sigma={sigma}, dt={dt}\n")

    for step in range(1, 6):
        dW = rng.normal(loc=0.0, scale=np.sqrt(dt))
        drift = theta * (mu - x_prev) * dt
        diffusion = sigma * dW
        x_next = x_prev + drift + diffusion
        print(
            f"step {step}: drift = theta*(mu-x)*dt = {theta}*({mu}-{x_prev:.4f})*{dt} = {drift:+.4f}  |  "
            f"diffusion = sigma*dW = {sigma}*{dW:+.4f} = {diffusion:+.4f}  |  "
            f"x_next = {x_prev:.4f} {drift:+.4f} {diffusion:+.4f} = {x_next:.4f}"
        )
        x_prev = x_next
    print()


def theoretical_stationary_stats(mu, theta, sigma, dt):
    """
    For theta > 0 the OU process has a *stationary* distribution it
    settles into regardless of x0:

        X_inf ~ Normal(mu, sigma^2 / (2*theta))

    That variance formula falls straight out of the SDE: at equilibrium
    the pull back towards mu (strength theta) exactly balances the
    variance being pumped in by sigma*dW every step. We report it here so
    the *simulated* mean/variance below can be checked against the
    *theoretical* prediction -- if they don't roughly agree, theta*dt is
    too large for the discretisation to be accurate.

    half_life = ln(2) / theta is the number of time-steps it takes an
    average deviation from mu to shrink by half -- the standard way
    traders translate theta into "how many days until this spread
    reverts", e.g. for pairs trading.
    """
    stationary_var = sigma ** 2 / (2 * theta)
    half_life = np.log(2) / theta if theta > 0 else np.inf
    return stationary_var, half_life


def summarize(name, series, mu, theta, sigma, dt):
    """Build one row of empirical-vs-theoretical statistics for a series."""
    empirical_mean = series.mean()
    empirical_var = series.var()

    if theta > 0:
        theo_var, half_life = theoretical_stationary_stats(mu, theta, sigma, dt)
        theo_mean = mu
    else:
        # theta = 0 -> no stationary distribution at all: variance grows
        # linearly with time forever (that's the definition of a random
        # walk), so "theoretical variance" isn't a fixed number.
        theo_var, half_life = np.nan, np.inf
        theo_mean = np.nan

    return {
        "series": name,
        "empirical mean": round(empirical_mean, 2),
        "theoretical mean": theo_mean,
        "empirical var": round(empirical_var, 2),
        "theoretical stationary var": None if np.isnan(theo_var) else round(theo_var, 2),
        "half-life (steps)": "inf (no reversion)" if half_life == np.inf else round(half_life, 1),
    }


def main():
    worked_example()

    n_steps = 500
    dt = 1.0

    # 1) Mean-reverting series: theta > 0 pulls the path back to mu = 100
    mean_reverting = simulate_ou_process(
        x0=100, theta=0.05, mu=100, sigma=1.0, dt=dt, n_steps=n_steps
    )

    # 2) Random walk: theta = 0 removes the pull entirely -> classic
    #    Brownian motion, the "null hypothesis" the ADF test checks against.
    random_walk = simulate_ou_process(
        x0=100, theta=0.0, mu=100, sigma=1.0, dt=dt, n_steps=n_steps
    )

    # 3) Trending series: positively autocorrelated increments (momentum),
    #    NOT a random walk plus deterministic drift -- see
    #    simulate_momentum_process()'s docstring for why that distinction
    #    matters (used later to get Hurst H > 0.5).
    trending = simulate_momentum_process(
        x0=300, rho=0.8, sigma=0.5, n_steps=n_steps
    )

    # Persist the three series so 02_adf_test.py / 03_hurst_exponent.py use
    # the exact same example data instead of re-simulating with new noise.
    np.savez(
        "examples/ou_example_series.npz",
        mean_reverting=mean_reverting,
        random_walk=random_walk,
        trending=trending,
    )
    print("Saved series to examples/ou_example_series.npz "
          "(used by the ADF and Hurst scripts).\n")

    # Statistical summary: empirical mean/variance from the simulated path
    # vs. the theoretical stationary values the OU math predicts, plus the
    # half-life -- the number that actually matters for trading, since it
    # tells you how many bars to expect before a deviation from mu decays.
    print("=" * 70)
    print("STATISTICAL SUMMARY: empirical vs. theoretical")
    print("=" * 70)
    rows = [
        summarize("mean-reverting (theta=0.05)", mean_reverting, 100, 0.05, 1.0, dt),
        summarize("random walk (theta=0)", random_walk, 100, 0.0, 1.0, dt),
        summarize("trending (momentum, rho=0.8)", trending, 100, 0.0, 1.0, dt),
    ]
    table = pd.DataFrame(rows).set_index("series")
    print(table.to_string())
    print(
        "\nReading this table:\n"
        "  - 'empirical mean' close to 'theoretical mean' (100) confirms the\n"
        "    simulation is correctly pulling the series back to mu.\n"
        "  - 'empirical var' close to 'theoretical stationary var' confirms\n"
        "    the noise/reversion balance matches the closed-form prediction\n"
        "    sigma^2 / (2*theta) -- this is the actual 'calculation' behind\n"
        "    the OU SDE, not just a plot that looks mean-reverting.\n"
        "  - random walk / trending both have NaN theoretical var and\n"
        "    infinite half-life because neither has an equilibrium to\n"
        "    return to -- variance grows without bound in both cases (that's\n"
        "    the null hypothesis the ADF test checks for), even though only\n"
        "    'trending' has the positive-autocorrelation structure that the\n"
        "    Hurst Exponent (03_hurst_exponent.py) will pick up as H > 0.5.\n"
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(mean_reverting, color="tab:green")
    axes[0].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Mean-reverting (theta=0.05): keeps returning to mu=100")

    axes[1].plot(random_walk, color="tab:blue")
    axes[1].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Random walk (theta=0): no pull back to mu, drifts freely")

    axes[2].plot(trending, color="tab:red")
    axes[2].set_title("Trending (momentum: rho=0.8 autocorrelated increments)")

    plt.tight_layout()
    plt.savefig("examples/ou_process_simulation.png", dpi=120)
    print("Saved plot to examples/ou_process_simulation.png")


if __name__ == "__main__":
    main()
