"""
Simulation-based (Monte Carlo) calibration of entry/stop z-score thresholds
for the mean-reversion engine, per pair, instead of one fixed entry_z=1.5/
stop_z=3.5 applied to every pair regardless of its own OU parameters or
trading costs.

Why simulation, not a textbook closed-form: the classic no-cost OU result
(Bertram 2010; Zhang & Zhang 2008, "Trading a mean-reverting asset") is that
a pure mean-reversion strategy with NO stop-loss wants to capture every
fluctuation (entry threshold -> 0), since each is positive expected value on
its own. But this engine always pairs entry_z with a proportionally-scaled
stop_z (same 1.5:3.5 ratio as the fixed-threshold default) -- and a very
tight entry_z also means a very tight stop, which triggers on ordinary OU
noise before a position has any real chance to revert. Simulation directly
confirms this: even at ZERO transaction cost, there is a genuine INTERIOR
optimum (empirically around entry_z~1.0 for typical kappa/sigma, not the
grid floor) -- see test_ou_calibration.py's
test_calibrated_entry_z_has_interior_optimum_even_at_zero_cost. Cost then
pushes that optimum wider still by additionally penalizing trade frequency
(test_calibrate_entry_z_widens_as_cost_increases). Simulating the pair's
OWN fitted (kappa, sigma) this way -- same Euler-Maruyama method as
examples/01_ornstein_uhlenbeck_process.py -- avoids needing to solve (or
misremember) the exact transcendental optimal-threshold equation from the
literature, at the cost of being an approximation bounded by the grid
resolution and simulation noise rather than an exact closed form.
"""

import numpy as np


def simulate_ou_paths(kappa, sigma, n_bars, n_sims, rng, x0=0.0):
    """
    Vectorized Euler-Maruyama simulation of n_sims independent OU paths,
    dt=1 bar: x_t = x_(t-1) + kappa*(0 - x_(t-1)) + sigma*eps_t. Returns an
    (n_sims, n_bars) array. Same discretization examples/01 uses, just
    vectorized across simulations instead of looped one at a time.
    """
    paths = np.empty((n_sims, n_bars))
    paths[:, 0] = x0
    noise = rng.normal(0, sigma, size=(n_sims, n_bars - 1))
    for t in range(1, n_bars):
        paths[:, t] = paths[:, t - 1] + kappa * (0 - paths[:, t - 1]) + noise[:, t - 1]
    return paths


def simulate_trading_pnl(paths, entry_z, stop_z, stationary_std, cost_per_round_trip):
    """
    Vectorized (across simulations) trading loop over each path. Thresholds
    are expressed in units of the OU process's own THEORETICAL stationary
    std (sigma/sqrt(2*kappa)), not a rolling estimate -- this calibration
    asks "what threshold is right for a process with these true parameters,"
    independent of any particular rolling-window estimation noise (that's a
    separate, already-solved problem, see _engine_meanrev.derive_entry_exit_z
    / _common.derive_entry_exit_z, which derives the WINDOW, not the level).
    Returns (mean net PnL per bar across all simulated paths, mean trades per path).
    """
    entry_level = entry_z * stationary_std
    stop_level = stop_z * stationary_std
    n_sims, n_bars = paths.shape
    pos = np.zeros(n_sims)
    total_pnl = np.zeros(n_sims)
    total_trades = np.zeros(n_sims)

    for t in range(1, n_bars):
        x = paths[:, t]
        x_prev = paths[:, t - 1]
        total_pnl += pos * (x - x_prev)

        flat = pos == 0.0
        enter_short = flat & (x > entry_level)
        enter_long = flat & (x < -entry_level)
        total_trades += (enter_short | enter_long).astype(float)
        pos = np.where(enter_short, -1.0, pos)
        pos = np.where(enter_long, 1.0, pos)

        exit_long = (pos == 1.0) & (x >= 0)
        exit_short = (pos == -1.0) & (x <= 0)
        hard_stop = (pos != 0.0) & (np.abs(x) > stop_level)
        pos = np.where(exit_long | exit_short | hard_stop, 0.0, pos)

    total_pnl -= total_trades * cost_per_round_trip
    return float(np.mean(total_pnl)) / n_bars, float(np.mean(total_trades))


DEFAULT_ENTRY_Z_GRID = tuple(np.round(np.arange(0.25, 3.01, 0.25), 2))


def calibrate_entry_z(kappa, sigma, cost_per_round_trip, entry_z_grid=DEFAULT_ENTRY_Z_GRID,
                       stop_z_ratio=3.5 / 1.5, n_bars=500, n_sims=300, rng=None):
    """
    Grid search over candidate entry_z values (stop_z = entry_z *
    stop_z_ratio, keeping the same 1.5:3.5 ratio as the fixed-threshold
    default), simulating the pair's OWN fitted (kappa, sigma), and returns
    whichever entry_z maximizes expected net-of-cost PnL per bar.

    Returns None if kappa/sigma aren't valid (not mean-reverting) -- caller
    should fall back to the fixed DEFAULTS entry_z/stop_z in that case.
    """
    if kappa is None or sigma is None or kappa <= 0 or sigma <= 0 or not np.isfinite(kappa):
        return None
    if rng is None:
        rng = np.random.default_rng(0)

    stationary_std = sigma / np.sqrt(2 * kappa)
    paths = simulate_ou_paths(kappa, sigma, n_bars, n_sims, rng)

    best_entry_z, best_score = entry_z_grid[0], -np.inf
    grid_scores = []
    for entry_z in entry_z_grid:
        stop_z = entry_z * stop_z_ratio
        score, _ = simulate_trading_pnl(paths, entry_z, stop_z, stationary_std, cost_per_round_trip)
        grid_scores.append((float(entry_z), score))
        if score > best_score:
            best_score, best_entry_z = score, entry_z

    return {
        "entry_z": float(best_entry_z), "stop_z": float(best_entry_z * stop_z_ratio),
        "expected_pnl_per_bar": float(best_score), "stationary_std": float(stationary_std),
        "grid": grid_scores,
    }
