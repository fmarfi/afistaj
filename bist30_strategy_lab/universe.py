"""
The real BIST30 universe, and the hourly-bar data layer everything else in
this mission builds on.

Both prior missions (examples/06_walk_forward_bist_pairs.py,
intraday_pairs_trading/_engine.py) traded an ad hoc 16-ticker BIST_UNIVERSE
picked by hand for "a reasonable chance of finding genuine within-sector
relationships" -- explicitly NOT the real index. This mission trades the
actual BIST30 (XU030) constituents instead, so any strategy that looks good
has to prove itself across the real, official large-cap universe rather
than a hand-picked subset.

Cross-validated against three independent sources (Aug 2026); stable through
at least Q3 2026 per Borsa Istanbul's quarterly index-revision announcement.
This list is the canonical one for this mission -- it is NOT shared with or
imported by dashboard/bist30_dashboard.py or dashboard/bist30_webapp.py,
which have their own (older, slightly stale) copies; reconciling those is
out of scope here.
"""

import numpy as np
import pandas as pd

BIST30_INDEX = "XU030.IS"

BIST30_TICKERS = [
    "AEFES.IS", "AKBNK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS",
    "DSTKF.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS",
    "GARAN.IS", "GUBRF.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS",
    "MGROS.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS",
    "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
    "TRALT.IS", "TTKOM.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS",
]

# Tickers worth knowing about before trusting a lookback-heavy fit on them.
TICKER_NOTES = {
    "TRALT.IS": "Renamed from KOZAL.IS (Koza Altin Isletmeleri -> Turk Altin "
                "Isletmeleri A.S.). VERIFIED via a real download (2026-08): "
                "yfinance only carries TRALT.IS history from 2025-11-24 "
                "onward (~9 months as of 2026-08) -- older bars under the "
                "KOZAL.IS name are NOT automatically included, so any "
                "lookback longer than that window is trading on a much "
                "shorter effective history than the other 29 tickers.",
    "DSTKF.IS": "IPO'd Jan 2025 -- only ~1.5 years of listing history as of "
                "2026-08, and much higher volatility than the rest of the "
                "universe (reportedly BIST30's best YTD-2026 performer). "
                "Not excluded outright; every walk-forward window naturally "
                "skips it wherever it lacks enough trailing history (see "
                "_common.min_history_guard).",
}

# Tickers flagged with a short-history caveat above -- kept as a separate,
# explicit set so an engine can check `ticker in SHORT_HISTORY_TICKERS`
# without parsing TICKER_NOTES's prose. Both verified by direct download:
# DSTKF.IS ~3255 hourly bars, TRALT.IS ~1540 hourly bars, vs. ~6100+ for a
# normal full-history constituent over the same 730d-requested pull.
SHORT_HISTORY_TICKERS = {"DSTKF.IS", "TRALT.IS"}

# yfinance caps the 60-minute interval at ~730 days of history (vs. 60 days
# for 5m/15m bars, and effectively unlimited for daily bars). This is a
# deliberate middle ground for this mission: finer granularity than a daily
# study, far more usable history than 5-minute intraday. The tradeoff --
# ~2 years is less calendar depth than a multi-decade daily study would
# give PCA factor stability or the Deflated Sharpe Ratio to work with -- is
# real and is documented in WORKFLOW.md, not silently assumed away.
BIST30_INTERVAL = "60m"
BIST30_PERIOD = "730d"


def download_universe(tickers=None, period=BIST30_PERIOD, interval=BIST30_INTERVAL):
    """
    Hourly OHLC-adjusted close panel for the given tickers (default: the
    full BIST30 universe). Same yf.download(..., auto_adjust=True) shape as
    examples/06_walk_forward_bist_pairs.py's download_universe, just with
    hourly params instead of daily.
    """
    import yfinance as yf
    if tickers is None:
        tickers = BIST30_TICKERS
    prices = yf.download(tickers, period=period, interval=interval,
                          progress=False, auto_adjust=True)["Close"]
    # Only drop rows where EVERY ticker is missing (e.g. a holiday nothing
    # traded on). Do NOT forward-fill or drop rows for a single short-
    # history ticker's sake (e.g. DSTKF.IS, listed ~2025-01) -- that would
    # silently truncate the ENTIRE 30-ticker panel down to whichever
    # ticker IPO'd most recently. Leave each column's pre-listing cells as
    # NaN; callers slice to a specific ticker's/pair's own valid range
    # (see _common.min_history_guard) instead of forcing one shared range.
    prices = prices.dropna(how="all")
    # yfinance returns UTC timestamps; convert once here so every downstream
    # consumer (screening, walk-forward, the dashboard) shows the time a
    # BIST trader would actually recognize -- same reasoning as
    # intraday_pairs_trading/_engine.py's download_intraday_universe.
    prices.index = prices.index.tz_convert("Europe/Istanbul")
    return prices


def bars_per_year(index):
    """
    Measures the real average bar density from an actual downloaded index,
    rather than assuming a bar count per day the way the daily (252) and
    5-minute-intraday (95*250) missions do -- BIST's hourly bar count per
    session isn't something to guess at when it's cheap to measure directly.
    """
    if len(index) < 2:
        return np.nan
    n_days = pd.Series(index).dt.date.nunique()
    if n_days == 0:
        return np.nan
    bars_per_day = len(index) / n_days
    return bars_per_day * 252
