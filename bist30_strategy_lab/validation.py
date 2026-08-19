"""
Validation harness: turns three separately-built strategy families' results
into one honest, bias-corrected comparison.

Answers the exact question this mission exists to answer without repeating
the overfitting mistake already made once this session (tuning one
strategy's parameters on one ticker looked great in-sample, didn't
generalize): is a strategy's best-looking result real, or does it just look
that way because several strategies (and their parameter variants) were
tried and the best of the bunch was reported?

Two independent pieces, used together:
  1. MOVING BLOCK BOOTSTRAP (reused directly from examples/07_bootstrap_
     significance.py's mechanics) -- is a single strategy's Sharpe ratio
     distinguishable from noise, given the one history it was actually
     tested on?
  2. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado, 2014) -- discounts the
     observed Sharpe for the number of trials it took to find it (N
     strategies/parameter sets tried), the sample size, and the return
     distribution's skew/kurtosis, answering "how much of this Sharpe is
     just what you'd expect from trying N things by chance."
Plus cross-universe consistency (% of tickers/pairs a family was actually
profitable on) and a parameter-perturbation sensitivity check -- a result
that collapses under a small parameter nudge is a red flag for overfitting,
mirroring exactly what happened with the single-ticker tuning earlier this
session.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import _common as c
import universe as uni

RNG = np.random.default_rng(123)
EULER_MASCHERONI = 0.5772156649


# ---------------------------------------------------------------------------
# 1. Moving block bootstrap (mechanics reused from examples/07_bootstrap_
#    significance.py -- generalized to take bars_per_year as a parameter
#    instead of a hardcoded 252, since this mission trades hourly bars).
# ---------------------------------------------------------------------------

def moving_block_bootstrap_indices(n, block_length, rng):
    """Identical mechanics to examples/07's function of the same name."""
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n]


def bootstrap_stats(pnl, bars_per_year):
    """
    nan-safe (nanmean/nanstd): a resampled block can include bars from
    before an engine's warm-up finished setting a signal (e.g. mean-
    reversion's walk_forward_meanrev leaves spread/pnl as NaN until it has
    enough trailing sessions to fit a hedge ratio at all) -- those bars
    carry no information and should be skipped, not propagate NaN through
    the whole resample the way plain np.mean/np.std would.
    """
    ann_ret = np.nanmean(pnl) * bars_per_year
    ann_vol = np.nanstd(pnl) * np.sqrt(bars_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe


def bootstrap_significance(pnl_oos, bars_per_year, block_length, n_boot=2000, rng=RNG):
    """
    Draws n_boot moving-block-resampled versions of the OOS pnl series and
    reports what fraction had Sharpe > 0 -- P(Sharpe > 0), plus a 90% CI on
    the Sharpe ratio itself. block_length should be tied to the strategy's
    own typical holding period (see each engine's rebalance/holding
    constants), same reasoning as examples/07's BLOCK_LENGTH=21 choice.
    """
    n = len(pnl_oos)
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        idx = moving_block_bootstrap_indices(n, block_length, rng)
        _, _, sharpes[i] = bootstrap_stats(pnl_oos[idx], bars_per_year)
    # A handful of draws can land on a degenerate (zero-variance -- e.g. all
    # non-trading, flat-pnl bars) block, which bootstrap_stats correctly
    # reports as NaN Sharpe rather than a fabricated number. Use nanpercentile
    # so those few degenerate draws are excluded from the summary instead of
    # propagating NaN through the whole 90% CI.
    p5, p50, p95 = np.nanpercentile(sharpes, [5, 50, 95])
    return {
        "sharpe_median": float(p50), "sharpe_ci_low": float(p5), "sharpe_ci_high": float(p95),
        "p_sharpe_gt_0": float(np.mean(sharpes > 0)),
        "n_degenerate_draws": int(np.isnan(sharpes).sum()),
    }


# ---------------------------------------------------------------------------
# 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) -- from scratch.
# ---------------------------------------------------------------------------

def _sharpe_std_error(sr_per_period, n_obs, skew=0.0, kurtosis=3.0):
    """
    Standard error of an estimated per-period Sharpe ratio, correcting for
    the return series' own skew/(raw, not excess) kurtosis -- Mertens
    (2002)/Christie (2005)'s formula, the one PSR/DSR are built on.
    kurtosis=3.0 (a normal distribution's raw kurtosis) recovers the
    textbook SE = sqrt(1/(n_obs-1)) special case.
    """
    var_sr = (1 - skew * sr_per_period + (kurtosis - 1) / 4 * sr_per_period ** 2) / max(n_obs - 1, 1)
    return float(np.sqrt(max(var_sr, 1e-12)))


def expected_max_sharpe(n_trials, sr_std_per_period):
    """
    Expected value of the MAXIMUM per-period Sharpe ratio you'd observe out
    of n_trials independent strategies/configurations, if every one of them
    truly had zero skill (an extreme-value-theory approximation, per Bailey
    & Lopez de Prado 2014). This -- not zero -- is the fair benchmark an
    observed "best of N" Sharpe should be judged against.
    """
    if n_trials <= 1:
        return 0.0
    gamma = EULER_MASCHERONI
    return sr_std_per_period * (
        (1 - gamma) * norm.ppf(1 - 1.0 / n_trials) + gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )


def deflated_sharpe_ratio(observed_sharpe_annualized, n_trials, n_obs, bars_per_year,
                           skew=0.0, kurtosis=3.0):
    """
    DSR = P(true Sharpe > 0), after deflating the observed (annualized)
    Sharpe for having been the best result out of n_trials strategies/
    parameter variants tried. Decreases monotonically as n_trials grows,
    for a fixed observed Sharpe -- trying more things and reporting the
    best one should require stronger evidence, not the same evidence.
    """
    sr = observed_sharpe_annualized / np.sqrt(bars_per_year)  # per-period Sharpe
    sr_std = _sharpe_std_error(sr, n_obs, skew, kurtosis)
    sr_benchmark = expected_max_sharpe(n_trials, sr_std)
    dsr = float(norm.cdf((sr - sr_benchmark) / sr_std)) if sr_std > 0 else float("nan")
    return {
        "dsr": dsr,
        "sr_benchmark_annualized": float(sr_benchmark * np.sqrt(bars_per_year)),
        "sr_per_period": float(sr),
        "sr_std_per_period": float(sr_std),
    }


# ---------------------------------------------------------------------------
# 3. Full-universe consistency drivers -- one per engine "shape". Pairs
#    trading picks ONE pair per run, so consistency is measured by re-
#    running across the top-K qualified pairs; the basket engines
#    (momentum, PCA) already trade the whole universe in a single run, so
#    consistency is measured from each traded ticker's own summed pnl in
#    the trade log.
# ---------------------------------------------------------------------------

def evaluate_meanrev_universe(prices, params=None, bars_per_year=None, top_k_pairs=15,
                               family="mean_reversion_ou"):
    """
    family: the label attached to this evaluation's leaderboard row. Pass a
    distinct value (e.g. "mean_reversion_ou_cost_calibrated") when calling
    this twice with different params on the same prices -- e.g. to compare
    fixed vs. cost-calibrated entry_z as two separate leaderboard entries --
    otherwise both calls collide under the same family name and the
    leaderboard/bootstrap-check logic in 03_run_validation_leaderboard.py
    can silently pick the wrong one of the two.
    """
    import _engine_meanrev as em
    params = {**em.DEFAULTS, **(params or {})}
    if bars_per_year is None:
        bars_per_year = uni.bars_per_year(prices.index)
    bars_per_day = bars_per_year / 252

    screening, split_idx, sess, log_prices = em.screen_only(prices, params, bars_per_day)
    qualified = screening[screening["qualifies"]].head(top_k_pairs)

    rows, results_by_pair = [], {}
    for _, row in qualified.iterrows():
        pair = (row["ticker_a"], row["ticker_b"])
        try:
            result = em.run_full_backtest_meanrev(
                prices, params=params, bars_per_year=bars_per_year,
                screening_result=(screening, split_idx, sess, log_prices), selected_pair=pair,
            )
        except Exception:
            continue
        results_by_pair[row["pair"]] = result
        rows.append({"unit": row["pair"], "sharpe": result["stats"]["sharpe (naive)"],
                      "n_trades": result["n_trades"], "max_drawdown": result["stats"]["max drawdown"]})

    per_unit = pd.DataFrame(rows)
    pct_profitable = float((per_unit["sharpe"] > 0).mean() * 100) if len(per_unit) else float("nan")
    best_result = None
    if len(per_unit):
        best_unit = per_unit.sort_values("sharpe", ascending=False).iloc[0]["unit"]
        best_result = results_by_pair[best_unit]

    return {
        "family": family, "n_units": len(per_unit), "pct_units_profitable": pct_profitable,
        "per_unit": per_unit, "best_result": best_result, "bars_per_year": bars_per_year,
    }


def evaluate_basket_universe(family_name, run_fn, prices, params=None, bars_per_year=None):
    if bars_per_year is None:
        bars_per_year = uni.bars_per_year(prices.index)
    result = run_fn(prices, params=params, bars_per_year=bars_per_year)

    trades_df = pd.DataFrame(result["trades"])
    if len(trades_df) and "ticker" in trades_df.columns:
        per_unit = trades_df.groupby("ticker")["pnl"].sum().reset_index()
        per_unit.columns = ["unit", "total_pnl"]
        pct_profitable = float((per_unit["total_pnl"] > 0).mean() * 100) if len(per_unit) else float("nan")
    else:
        per_unit = pd.DataFrame(columns=["unit", "total_pnl"])
        pct_profitable = float("nan")

    return {
        "family": family_name, "n_units": len(per_unit), "pct_units_profitable": pct_profitable,
        "per_unit": per_unit, "best_result": result, "bars_per_year": bars_per_year,
    }


# ---------------------------------------------------------------------------
# 4. Parameter-perturbation sensitivity -- a strategy whose Sharpe collapses
#    under a small nudge is flagged as likely overfit, the same lesson
#    learned earlier this session tuning a single-ticker pairs strategy.
# ---------------------------------------------------------------------------

def perturbation_sensitivity(run_fn, prices, base_params, perturb_keys, bars_per_year, pct=0.1):
    """
    Reruns run_fn with each of perturb_keys nudged +-pct, reports the
    Sharpe delta from the base run for each nudge. Large deltas relative to
    the base Sharpe are a red flag, not proof of a bad strategy by
    themselves -- read alongside DSR and cross-universe consistency.
    """
    base_result = run_fn(prices, params=base_params, bars_per_year=bars_per_year)
    base_sharpe = base_result["stats"]["sharpe (naive)"]

    rows = []
    for key in perturb_keys:
        if key not in base_params or not isinstance(base_params[key], (int, float)):
            continue
        for direction in (1 + pct, 1 - pct):
            nudged = dict(base_params)
            nudged[key] = type(base_params[key])(base_params[key] * direction)
            try:
                result = run_fn(prices, params=nudged, bars_per_year=bars_per_year)
                sharpe = result["stats"]["sharpe (naive)"]
            except Exception:
                sharpe = float("nan")
            rows.append({"param": key, "multiplier": round(direction, 2), "sharpe": sharpe,
                         "sharpe_delta": sharpe - base_sharpe if np.isfinite(sharpe) else float("nan")})
    return {"base_sharpe": base_sharpe, "perturbations": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# 5. Leaderboard -- ranks by DSR-adjusted Sharpe + cross-universe
#    consistency, NOT peak single-backtest Sharpe.
# ---------------------------------------------------------------------------

def leaderboard(evaluations, n_trials, starting_capital=100_000):
    """
    evaluations: list of dicts from evaluate_meanrev_universe /
    evaluate_basket_universe. n_trials should reflect the TOTAL number of
    strategy/parameter combinations actually tried across this whole
    comparison exercise (not just len(evaluations)) -- if you tuned each
    family's parameters before settling on the ones passed in here, that
    tuning counts too. Ranks by Deflated Sharpe Ratio first (the bias-
    corrected "is this real" answer), cross-universe consistency second.
    """
    rows = []
    for ev in evaluations:
        result = ev["best_result"]
        if result is None:
            continue
        stats = result["stats"]
        capital_out = c.add_capital_pnl(dict(result), starting_capital)
        dsr_info = deflated_sharpe_ratio(
            stats["sharpe (naive)"], n_trials, len(result["results"]), ev["bars_per_year"],
        )
        rows.append({
            "family": ev["family"], "sharpe": stats["sharpe (naive)"], "dsr": dsr_info["dsr"],
            "dsr_benchmark_sharpe": dsr_info["sr_benchmark_annualized"],
            "total_return_pct": round(capital_out["total_return_pct"], 2),
            "max_drawdown": stats["max drawdown"], "n_units_tested": ev["n_units"],
            "pct_units_profitable": ev["pct_units_profitable"],
        })
    if not rows:
        return pd.DataFrame(columns=["family", "sharpe", "dsr", "total_return_pct",
                                      "max_drawdown", "n_units_tested", "pct_units_profitable"])
    return pd.DataFrame(rows).sort_values(
        ["dsr", "pct_units_profitable"], ascending=False
    ).reset_index(drop=True)
