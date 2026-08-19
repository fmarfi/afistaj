"""
pytest suite for universe.py. No network access required -- these check the
static ticker list itself, not a live download (matching this repo's
convention, see intraday_pairs_trading/test_engine.py, of keeping the
regression suite synthetic-data-only).

Run: python -m pytest bist30_strategy_lab/test_universe.py -v
(from the afistaj/ project root, with the venv active)
"""

import universe


def test_exactly_thirty_unique_tickers():
    assert len(universe.BIST30_TICKERS) == 30
    assert len(set(universe.BIST30_TICKERS)) == 30


def test_every_ticker_has_is_suffix():
    assert all(t.endswith(".IS") for t in universe.BIST30_TICKERS)


def test_index_ticker_not_in_constituent_list():
    assert universe.BIST30_INDEX not in universe.BIST30_TICKERS


def test_renamed_ticker_present_old_name_absent():
    assert "TRALT.IS" in universe.BIST30_TICKERS
    assert "KOZAL.IS" not in universe.BIST30_TICKERS
    assert "KOZAA.IS" not in universe.BIST30_TICKERS


def test_short_history_tickers_flagged():
    assert "DSTKF.IS" in universe.BIST30_TICKERS
    assert "DSTKF.IS" in universe.SHORT_HISTORY_TICKERS
    assert "TRALT.IS" in universe.SHORT_HISTORY_TICKERS


def test_short_history_tickers_are_a_subset_of_the_universe():
    assert universe.SHORT_HISTORY_TICKERS.issubset(set(universe.BIST30_TICKERS))


def test_bars_per_year_scales_with_bar_density():
    import pandas as pd
    # 3 sessions, 5 bars each = 5 bars/day -> 5*252 bars/year
    idx = pd.to_datetime([
        "2026-01-05 10:00", "2026-01-05 11:00", "2026-01-05 12:00", "2026-01-05 13:00", "2026-01-05 14:00",
        "2026-01-06 10:00", "2026-01-06 11:00", "2026-01-06 12:00", "2026-01-06 13:00", "2026-01-06 14:00",
        "2026-01-07 10:00", "2026-01-07 11:00", "2026-01-07 12:00", "2026-01-07 13:00", "2026-01-07 14:00",
    ])
    result = universe.bars_per_year(idx)
    assert result == 5 * 252


def test_bars_per_year_empty_index_is_nan():
    import pandas as pd
    import numpy as np
    result = universe.bars_per_year(pd.DatetimeIndex([]))
    assert np.isnan(result)
