"""
Interactive dashboard for this mission: a LEADERBOARD tab (reads 02/03's
cached, DSR-corrected full-universe comparison -- fast, no recompute) and
an EXPLORE tab (live single-backtest parameter tweaking per family, same
engines 02/03 use, same shape as intraday_pairs_trading/04's screen-then-
select UX).

The Explore tab shows the individual positions, not just the aggregate:
every trade is marked on the equity curve (and, for mean reversion, on the
z-score chart), each traded stock gets its OWN price chart with that stock's
own buys and sells on it, a per-trade PnL chart shows whether the total is
spread across trades or carried by one, and the trade log is a collapsible
card per trade whose expanded detail carries each leg's real quoted
entry/exit price -- the same trade-visibility treatment
intraday_pairs_trading/04 has.

Every chart is a real Plotly figure (zoom, pan, hover, box-select a date
range, toggle a series from the legend, save a PNG), not a rendered PNG:
"why did it trade there" is a question you answer by zooming into the bar,
which a static image can't support. All the bar-timeline charts on a page
share one x range -- zoom the equity curve and both stocks' price panels
follow -- so a drawdown can be lined up against the trades that caused it.

Run 02_run_full_universe_backtests.py and 03_run_validation_leaderboard.py
FIRST to populate the leaderboard tab's cache.

Run: python bist30_strategy_lab/04_interactive_dashboard.py
Then open http://127.0.0.1:5100 in a browser.
"""

import pickle
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from flask import Flask, request, Response

import universe as uni
import _common as c
import _engine_meanrev as em
import _engine_momentum as mo
import _engine_momentum_variants as ev
import _engine_pca as pca

app = Flask(__name__)
_price_cache = {}
LEADERBOARD_PKL = pathlib.Path(__file__).parent / "_leaderboard_detail.pkl"

BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8a86"

FAMILIES = {
    "mean_reversion_ou": {"label": "Mean reversion (OU half-life pairs)", "module": em, "run": em.run_full_backtest_meanrev},
    "momentum": {"label": "Momentum (cross-sectional rotation)", "module": mo, "run": mo.run_full_backtest_momentum},
    "pca_stat_arb": {"label": "PCA basket stat-arb", "module": pca, "run": pca.run_full_backtest_pca},
    # The upgrade-bench rules (DMI / ADX / Hurst / vol-adjusted), runnable
    # here as full backtests rather than only as bench statistics. Needs
    # OHLC: DMI is built from highs and lows, which the close-only panel
    # the other families use does not carry -- see get_ohlc.
    "momentum_variant": {"label": "Momentum upgrades (DMI / ADX / Hurst)", "module": ev,
                         "run": ev.run_full_backtest_variant, "needs_ohlc": True},
}
# Explore-tab-selectable families only (the 3 above). The cost-calibrated
# mean-reversion variant (see _ou_calibration.py) is a 4th evaluation in the
# cached leaderboard but isn't its own explore-tab entry yet -- toggle
# "derive_entry_z_from_cost" under Mean reversion's parameters instead.
LEADERBOARD_FAMILY_LABELS = {
    **{k: v["label"] for k, v in FAMILIES.items()},
    "mean_reversion_ou_cost_calibrated": "Mean reversion, cost-calibrated entry_z",
}

PERIOD_OPTIONS = ["180d", "365d", "730d"]

MEANREV_FIELDS = [
    ("session_lookback", "Session lookback (sessions)", 5, 60, int),
    ("rebalance_sessions", "Rebalance every (sessions)", 1, 20, int),
    ("entry_z", "Entry z-score", 0.5, 4.0, float),
    ("stop_z", "Stop-loss z-score", 1.0, 8.0, float),
]
MOMENTUM_FIELDS = [
    ("lookback_mom", "ROC lookback (bars)", 20, 500, int),
    ("ma_fast", "Fast MA (bars)", 5, 200, int),
    ("ma_slow", "Slow MA (bars)", 20, 500, int),
    ("rebalance", "Rebalance every (bars)", 5, 200, int),
    ("top_n", "Basket size (top N)", 1, 15, int),
]
PCA_FIELDS = [
    ("fit_window", "PCA fit window (bars)", 100, 1500, int),
    ("rebalance", "Refit every (bars)", 10, 300, int),
    ("n_factors", "Number of factors", 1, 8, int),
    ("z_window", "Residual z-score window (bars)", 20, 300, int),
    ("entry_z", "Entry z-score", 0.5, 4.0, float),
    ("stop_z", "Stop-loss z-score", 1.0, 8.0, float),
]
# The variant family carries momentum's own knobs plus the ones its extra
# signals need. Which of the last three actually bite depends on the rule
# selected -- the form says so rather than hiding them.
MOMENTUM_VARIANT_FIELDS = MOMENTUM_FIELDS + [
    ("di_window", "DMI / ADX window (bars)", 5, 400, int),
    ("adx_min", "ADX threshold (ADX rules only)", 0.0, 60.0, float),
    ("hurst_window", "Hurst window (bars, Hurst rules only)", 60, 800, int),
    ("hurst_min", "Hurst threshold (Hurst rules only)", 0.30, 0.80, float),
]
FIELDS_BY_FAMILY = {"mean_reversion_ou": MEANREV_FIELDS, "momentum": MOMENTUM_FIELDS,
                    "pca_stat_arb": PCA_FIELDS, "momentum_variant": MOMENTUM_VARIANT_FIELDS}


_ohlc_cache = {}


def get_ohlc(period):
    """
    Open/High/Low/Close for the variant family. Cached separately from the
    close-only panel because it is a different request, and kept per period
    so switching history length does not silently reuse the wrong window.
    """
    if _ohlc_cache.get("period") != period:
        import momentum_upgrades as up
        _ohlc_cache["panel"] = up.download_ohlc(period=period)
        _ohlc_cache["period"] = period
    return _ohlc_cache["panel"]


def get_prices(period):
    if _price_cache.get("period") != period:
        _price_cache["prices"] = uni.download_universe(period=period)
        _price_cache["period"] = period
    return _price_cache["prices"]


PLOTLY_CONFIG = {
    "scrollZoom": True, "displaylogo": False, "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

# BIST's hourly bars only exist 09:30-17:30 on weekdays. Without these
# breaks a datetime axis spends most of its width on closed hours and
# weekends the panel has no bars for, which makes a multi-day hold look
# like a flat line rather than a position.
SESSION_RANGEBREAKS = [dict(bounds=["sat", "mon"]), dict(bounds=[18, 9], pattern="hour")]

# Light-theme axis colors, matching PAGE_CSS's --border/--text-2. The page's
# own script re-colors every figure for dark mode (see PAGE_JS);
# Plotly can't read CSS custom properties itself.
LIGHT_GRID, LIGHT_TEXT = "#e4e2dd", "#52514e"


def chart_div(fig, sync_x=True):
    """
    One figure as an embeddable div. plotly.js is served once from
    /plotly.js and browser-cached rather than re-inlined into every figure
    (the bundle is ~5 MB). data-xsync marks the charts whose x axis is the
    bar timeline, which the page keeps zoomed and panned together.
    """
    html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)
    return f'<div class="chart"{" data-xsync=\"1\"" if sync_x else ""}>{html}</div>'


def style_figure(fig, height, title=None, time_axis=True):
    fig.update_layout(
        height=height, hovermode="x unified", dragmode="zoom",
        margin=dict(l=64, r=24, t=44 if title else 16, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, sans-serif", size=11, color=LIGHT_TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)),
        title=dict(text=title, x=0, font=dict(size=12)) if title else None,
    )
    fig.update_xaxes(gridcolor=LIGHT_GRID, linecolor=LIGHT_GRID, zeroline=False,
                     showspikes=True, spikemode="across", spikethickness=1,
                     spikedash="dot", spikecolor=GRAY)
    fig.update_yaxes(gridcolor=LIGHT_GRID, linecolor=LIGHT_GRID, zeroline=False,
                     title_font_size=10)
    if time_axis:
        fig.update_xaxes(rangebreaks=SESSION_RANGEBREAKS)
    return fig


def bar_x(result):
    """
    The backtest's own bar timestamps, as tz-naive Istanbul wall time.
    plotly.js re-interprets an offset-carrying timestamp in the VIEWER's
    timezone, which would slide every BIST bar off its session and out of
    SESSION_RANGEBREAKS's 09:00-18:00 window; dropping the offset after the
    conversion pins each bar to the hour a BIST trader saw.
    """
    ts = result["results"]["timestamp"]
    try:
        return ts.dt.tz_localize(None).to_numpy()
    except TypeError:
        return ts.to_numpy()


def add_insample_shading(fig, x, split_idx, rows=1):
    """
    Everything left of the split is the in-sample fit window -- no trades
    are taken there and the equity curve is flat by construction. Shaded
    rather than clipped off so the charts still show the price history the
    hedge ratio / factor loadings were actually fitted on.
    """
    if split_idx <= 0 or split_idx >= len(x):
        return
    fig.add_vrect(
        x0=pd.Timestamp(x[0]).to_pydatetime(), x1=pd.Timestamp(x[split_idx]).to_pydatetime(),
        fillcolor=GRAY, opacity=0.10, line_width=0, layer="below",
        annotation_text="in-sample (not traded)", annotation_position="top left",
        annotation_font_size=9, annotation_font_color=GRAY,
        row="all" if rows > 1 else 1, col=1,
    )


def trade_pnl(trade):
    """A trade's PnL in currency if add_capital_pnl has run, else its raw unit PnL."""
    return trade.get("pnl_currency", trade["pnl"])


def timestamp_tz(result):
    """Timezone of the engine's own bar index (Europe/Istanbul -- see universe.download_universe)."""
    try:
        return result["results"]["timestamp"].dt.tz
    except (AttributeError, TypeError):
        return None


def fmt_time(ts, tz=None):
    """
    Trade timestamps arrive in two shapes across the three engines: the
    mean-reversion engine builds its trade log from `index.to_numpy()`
    (tz-dropped datetime64 in UTC), momentum/PCA from `index[t]` directly
    (tz-aware Timestamps). Normalize both back to the bar index's own
    timezone so a trade time reads as the hour a BIST trader saw.
    """
    t = pd.Timestamp(ts)
    if tz is not None:
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        t = t.tz_convert(tz)
    return t.strftime("%Y-%m-%d %H:%M")


def trade_bar_positions(result):
    """
    Each trade paired with its (entry, exit) positional index in the
    backtest's own bar index, and its chronological trade number. Matching
    is done on integer nanoseconds (Timestamp.value) rather than on the
    objects themselves because the engines' trade times and their `results`
    timestamps differ in tz awareness -- see fmt_time.
    """
    timestamps = result["results"]["timestamp"].to_numpy()
    ts_to_pos = {pd.Timestamp(ts).value: i for i, ts in enumerate(timestamps)}
    positioned = []
    for n, t in enumerate(result["trades"], start=1):
        entry_pos = ts_to_pos.get(pd.Timestamp(t["entry_time"]).value)
        exit_pos = ts_to_pos.get(pd.Timestamp(t["exit_time"]).value)
        if entry_pos is not None and exit_pos is not None:
            positioned.append((n, t, entry_pos, exit_pos))
    return positioned


def trade_pnl_text(trade):
    value = trade_pnl(trade)
    return f"{value:+,.0f} TRY" if "pnl_currency" in trade else f"{value:+.5f} units"


def trade_marker_traces(x, y_values, positioned, tz, row_label=""):
    """
    Outlined triangle where a trade opened, filled dot (green profit / red
    loss) where it closed -- the same convention the static charts used,
    now as two hoverable traces per panel instead of hundreds of scatter
    calls, so a marker can tell you WHICH trade it is on hover.
    """
    entry = dict(x=[], y=[], text=[])
    exits = dict(x=[], y=[], text=[], color=[])
    for n, t, entry_pos, exit_pos in positioned:
        y_entry, y_exit = y_values[entry_pos], y_values[exit_pos]
        if not (np.isfinite(y_entry) and np.isfinite(y_exit)):
            continue
        pnl_text = trade_pnl_text(t)
        head = f"trade #{n}"
        entry["x"].append(x[entry_pos]); entry["y"].append(y_entry)
        entry["text"].append(f"<b>{head} opened</b><br>{fmt_time(t['entry_time'], tz)}"
                             f"<br>{row_label}{y_entry:,.2f}<br>PnL when closed: {pnl_text}")
        exits["x"].append(x[exit_pos]); exits["y"].append(y_exit)
        exits["text"].append(f"<b>{head} closed</b> ({t['exit_reason']})<br>{fmt_time(t['exit_time'], tz)}"
                             f"<br>{row_label}{y_exit:,.2f}<br>PnL: {pnl_text}<br>held {t['hold_bars']} bars")
        exits["color"].append(AQUA if trade_pnl(t) > 0 else RED)

    return [
        go.Scatter(x=entry["x"], y=entry["y"], mode="markers", name="trade opened",
                   marker=dict(symbol="triangle-up", size=9, color="rgba(0,0,0,0)",
                               line=dict(color=GRAY, width=1.2)),
                   text=entry["text"], hovertemplate="%{text}<extra></extra>"),
        go.Scatter(x=exits["x"], y=exits["y"], mode="markers", name="trade closed (green win / red loss)",
                   marker=dict(symbol="circle", size=8, color=exits["color"] or RED),
                   text=exits["text"], hovertemplate="%{text}<extra></extra>"),
    ]


def strategy_chart(result, family, family_label):
    """
    Equity curve and -- for mean reversion -- the z-score signal it traded
    on, stacked on ONE shared x axis so a drawdown lines up bar-for-bar
    with the signal that caused it. Plotted over the full history with the
    in-sample stretch shaded, rather than starting at the split, so the
    charts share an x range with the per-stock price panels below.
    """
    x = bar_x(result)
    tz = timestamp_tz(result)
    positioned = trade_bar_positions(result)
    show_z = family == "mean_reversion_ou" and "z" in result["results"]
    rows = 2 if show_z else 1

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.4] if show_z else [1.0])

    equity = np.asarray(result["equity_curve"], dtype=float)
    fig.add_trace(go.Scatter(x=x, y=equity, mode="lines", name="equity", line=dict(color=BLUE, width=1.4),
                             hovertemplate="%{y:,.0f} TRY<extra>equity</extra>"), row=1, col=1)
    fig.add_hline(y=result["starting_capital"], line=dict(color=GRAY, width=0.8, dash="dash"), row=1, col=1)
    for trace in trade_marker_traces(x, equity, positioned, tz, "equity "):
        fig.add_trace(trace, row=1, col=1)
    fig.update_yaxes(title_text="equity (TRY)", row=1, col=1)

    if show_z:
        res = result["results"]
        z = res["z"].to_numpy(dtype=float)
        entry_z, stop_z = res["entry_z"].to_numpy(dtype=float), res["stop_z"].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name="z-score", line=dict(color=BLUE, width=0.8),
                                 hovertemplate="z %{y:.2f}<extra></extra>"), row=2, col=1)
        for band, sign, color, dash, name in ((entry_z, 1, GRAY, "dash", "entry band"),
                                              (entry_z, -1, GRAY, "dash", None),
                                              (stop_z, 1, RED, "dot", "stop band"),
                                              (stop_z, -1, RED, "dot", None)):
            fig.add_trace(go.Scatter(x=x, y=sign * band, mode="lines", name=name or "",
                                     line=dict(color=color, width=0.9, dash=dash),
                                     showlegend=name is not None, hoverinfo="skip"), row=2, col=1)
        # Disqualified regime as a masked fill rather than per-stretch
        # shapes: hundreds of shapes would slow every zoom down.
        disqualified = ~res["qualified"].to_numpy(dtype=bool)
        z_extent = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
        band_top = np.where(disqualified, z_extent, np.nan)
        fig.add_trace(go.Scatter(x=x, y=band_top, mode="lines", name="disqualified regime",
                                 line=dict(width=0), fill="tozeroy", fillcolor="rgba(227,73,72,0.10)",
                                 hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=-band_top, mode="lines", showlegend=False,
                                 line=dict(width=0), fill="tozeroy", fillcolor="rgba(227,73,72,0.10)",
                                 hoverinfo="skip"), row=2, col=1)
        for trace in trade_marker_traces(x, z, positioned, tz, "z "):
            trace.showlegend = False
            fig.add_trace(trace, row=2, col=1)
        fig.update_yaxes(title_text="z-score", row=2, col=1)

    add_insample_shading(fig, x, result["split_idx"], rows)
    return chart_div(style_figure(
        fig, 460 if show_z else 320,
        f"{family_label} -- equity{' and signal' if show_z else ''}, out-of-sample (no trading costs)"))


def trade_side(trade, family):
    """
    Which way a single-leg trade was positioned. Momentum is long-only by
    construction; the PCA engine records `direction` on newly-run trades
    but older cached results predate that field, so fall back to the sign
    of the entry z-score -- the exact rule the engine itself uses (enter
    short above +entry_z, long below -entry_z).
    """
    if family in ("momentum", "momentum_variant"):
        return "long"      # both rotate long-only; nothing is ever shorted
    if trade.get("direction") in ("long", "short"):
        return trade["direction"]
    entry_z = trade.get("entry_z")
    if entry_z is None or not np.isfinite(entry_z):
        return "long"
    return "short" if entry_z > 0 else "long"


def trade_legs(trade, result, family):
    """
    (ticker, side, entry_price, exit_price) per stock actually traded. A
    mean-reversion trade touches BOTH stocks in opposite directions --
    "long spread" is buy A / sell B -- so it yields two legs and shows up
    on each ticker's own chart with that leg's own side and quoted prices.
    """
    if family == "mean_reversion_ou":
        long_a = trade["direction"] == "long spread"
        return [(result["ticker_a"], "long" if long_a else "short",
                 trade.get("entry_price_a"), trade.get("exit_price_a")),
                (result["ticker_b"], "short" if long_a else "long",
                 trade.get("entry_price_b"), trade.get("exit_price_b"))]
    ticker = trade.get("ticker")
    if ticker is None:
        return []
    return [(ticker, trade_side(trade, family), trade.get("entry_price"), trade.get("exit_price"))]


def leg_amount(result, family, ticker, multiplier, entry_i):
    """
    The real traded amount of ONE leg, in TRY: the base unit (starting
    capital / scale_max) times what the engine's own sizing put to work at
    that bar (_common.base_unit_multiplier), times this leg's own weight.

    For a pair, the second leg is the hedge ratio's worth of the first --
    the spread is log(a) - beta*log(b), so holding one unit of spread means
    beta units of B against one unit of A. For the single-leg families the
    weight is 1: the multiplier already carries the 1/n basket split.
    """
    dpu = result.get("dollar_per_unit")
    if dpu is None:
        return None
    weight = 1.0
    if family == "mean_reversion_ou" and ticker == result.get("ticker_b"):
        beta = result["results"]["beta"].iat[entry_i]
        weight = abs(float(beta)) if np.isfinite(beta) else 1.0
    return dpu * multiplier * weight


def enriched_legs(result, family):
    """
    Every traded stock mapped to its own legs (chronological), and every
    trade number mapped to its legs, each carrying the amount actually put
    to work and the share count that amount buys at the quoted entry price.
    """
    per_ticker, per_trade = {}, {}
    for n, trade, entry_i, exit_i in trade_bar_positions(result):
        multiplier = c.base_unit_multiplier(result["results"], entry_i)
        for ticker, side, entry_price, exit_price in trade_legs(trade, result, family):
            amount = leg_amount(result, family, ticker, multiplier, entry_i)
            usable_price = entry_price if entry_price and np.isfinite(entry_price) else None
            leg = dict(n=n, trade=trade, ticker=ticker, entry_i=entry_i, exit_i=exit_i, side=side,
                       entry_price=entry_price, exit_price=exit_price, amount=amount,
                       # Whole shares, floored: BIST trades whole shares, even
                       # though the backtest itself sizes in continuous units
                       # and never rounds -- see the caveat under the charts.
                       shares=int(amount // usable_price) if amount and usable_price else None)
            per_ticker.setdefault(ticker, []).append(leg)
            per_trade.setdefault(n, []).append(leg)
    return per_ticker, per_trade


def fmt_amount(amount):
    return f"{amount:,.0f} TRY" if amount and np.isfinite(amount) else "&mdash;"


def leg_move_pct(leg):
    """The leg's own price move, signed by which way this stock was held."""
    entry_price, exit_price = leg["entry_price"], leg["exit_price"]
    if not entry_price or exit_price is None or not np.isfinite(entry_price) or not np.isfinite(exit_price):
        return np.nan
    move = (exit_price / entry_price - 1) * 100
    return move if leg["side"] == "long" else -move


def ticker_traces(ticker, legs, price, x, tz, family, result, visible=True, showlegend=True):
    """
    One stock's own chart: its close, the stretches where a position in
    THIS stock was open (green held long / orange held short), and every
    buy and sell that happened in it. Marker prices are the engine's own
    recorded fills where it has them, so what's plotted is the same number
    the trade log shows -- not a re-read of the panel.

    Trace count per ticker is fixed at 5 so ticker_dropdown_chart can flip
    visibility in blocks without recomputing anything client-side.
    """
    traces = [go.Scatter(x=x, y=price, mode="lines", name="close", visible=visible, showlegend=False,
                         line=dict(color=GRAY, width=1),
                         hovertemplate="%{y:.2f}<extra>" + ticker + "</extra>")]

    for side, color, label in (("long", AQUA, "held long"), ("short", ORANGE, "held short")):
        seg_x, seg_y = [], []
        for leg in legs:
            if leg["side"] != side:
                continue
            span = slice(leg["entry_i"], leg["exit_i"] + 1)
            seg_x.extend(list(x[span]) + [None])
            seg_y.extend(list(price[span]) + [np.nan])
        # numpy rather than a Python list: plotly.py 6 base64-encodes numeric
        # arrays, and these hold-highlight series are the longest ones on the
        # page after the close itself.
        traces.append(go.Scatter(x=seg_x, y=np.array(seg_y, dtype=float), mode="lines",
                                 name=label, visible=visible,
                                 showlegend=showlegend and bool(seg_x), opacity=0.6,
                                 line=dict(color=color, width=3.5), hoverinfo="skip"))

    buys = dict(x=[], y=[], text=[])
    sells = dict(x=[], y=[], text=[])
    for leg in legs:
        trade = leg["trade"]
        pair_note = ""
        if family == "mean_reversion_ou":
            pair_note = ("<br>pair trade: "
                         + direction_label(trade["direction"], result["ticker_a"], result["ticker_b"]))
        for when in ("entry", "exit"):
            # Opening a long and closing a short are both buys; the marker
            # says what happened in THIS stock, not what the spread did.
            is_buy = (leg["side"] == "long") == (when == "entry")
            bucket = buys if is_buy else sells
            i = leg["entry_i"] if when == "entry" else leg["exit_i"]
            price_at = leg["entry_price"] if when == "entry" else leg["exit_price"]
            if price_at is None or not np.isfinite(price_at):
                price_at = price[i]
            if not np.isfinite(price_at):
                continue
            move = leg_move_pct(leg)
            bucket["x"].append(x[i])
            bucket["y"].append(price_at)
            size_note = ""
            if leg.get("amount"):
                shares = f" &asymp; {leg['shares']:,} shares" if leg.get("shares") else ""
                size_note = f"<br>amount: {fmt_amount(leg['amount'])}{shares}"
            bucket["text"].append(
                f"<b>{'BUY' if is_buy else 'SELL'} {ticker}</b> @ {price_at:,.2f}"
                f"<br>{fmt_time(trade['entry_time' if when == 'entry' else 'exit_time'], tz)}"
                f"<br>trade #{leg['n']} {when}, held {leg['side']}{pair_note}{size_note}"
                + (f"<br>this leg: {move:+.2f}% (side-adjusted)" if np.isfinite(move) else "")
                + f"<br>trade PnL: {trade_pnl_text(trade)}"
                + (f"<br>exit: {trade['exit_reason']}" if when == "exit" else ""))

    for bucket, symbol, color, name in ((buys, "triangle-up", AQUA, "buy"),
                                        (sells, "triangle-down", RED, "sell")):
        traces.append(go.Scatter(x=bucket["x"], y=bucket["y"], mode="markers", name=name,
                                 visible=visible, showlegend=showlegend,
                                 marker=dict(symbol=symbol, size=10, color=color,
                                             line=dict(color="rgba(0,0,0,0.35)", width=0.8)),
                                 text=bucket["text"], hovertemplate="%{text}<extra></extra>"))
    return traces


TRACES_PER_TICKER = 5

# How many stocks the basket families' picker offers, most-traded first.
# Every ticker embedded in the page carries its own full close series, so an
# uncapped picker on a 30-name universe puts tens of MB of JSON in one page
# and makes every zoom sluggish -- the cap is about page weight, not about
# hiding trades (the trade log below still lists all of them).
MAX_TICKER_PANELS = 8


def ticker_price_frame(result, prices):
    """Each stock's close on the backtest's OWN bar index (the mean-reversion engine drops bars either leg is missing)."""
    return prices.reindex(pd.DatetimeIndex(result["results"]["timestamp"]))


def ticker_stacked_chart(result, family, prices, per_ticker, tickers):
    """Both legs of a pair, stacked on the equity chart's shared x axis."""
    x = bar_x(result)
    tz = timestamp_tz(result)
    frame = ticker_price_frame(result, prices)
    fig = make_subplots(rows=len(tickers), cols=1, shared_xaxes=True, vertical_spacing=0.06)

    for row, ticker in enumerate(tickers, start=1):
        price = frame[ticker].to_numpy(dtype=float) if ticker in frame else np.full(len(x), np.nan)
        for trace in ticker_traces(ticker, per_ticker.get(ticker, []), price, x, tz, family, result,
                                   showlegend=(row == 1)):
            fig.add_trace(trace, row=row, col=1)
        fig.update_yaxes(title_text=f"{ticker} (TRY)", row=row, col=1)

    add_insample_shading(fig, x, result["split_idx"], len(tickers))
    return chart_div(style_figure(fig, 250 * len(tickers),
                                  "Each stock's own price, with its own buys and sells"))


def ticker_dropdown_chart(result, family, prices, per_ticker, tickers):
    """
    Basket families can touch most of the universe, so one panel with a
    ticker picker instead of 20 stacked rows. The dropdown only flips trace
    visibility -- switching stocks doesn't re-run the backtest.
    """
    x = bar_x(result)
    tz = timestamp_tz(result)
    frame = ticker_price_frame(result, prices)
    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        price = frame[ticker].to_numpy(dtype=float) if ticker in frame else np.full(len(x), np.nan)
        for trace in ticker_traces(ticker, per_ticker[ticker], price, x, tz, family, result,
                                   visible=(i == 0), showlegend=(i == 0)):
            fig.add_trace(trace)

    buttons = []
    for i, ticker in enumerate(tickers):
        visible = [False] * (len(tickers) * TRACES_PER_TICKER)
        for j in range(TRACES_PER_TICKER):
            visible[i * TRACES_PER_TICKER + j] = True
        buttons.append(dict(label=f"{ticker} ({len(per_ticker[ticker])})", method="update",
                            args=[{"visible": visible},
                                  {"yaxis.title.text": f"{ticker} (TRY)"}]))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True, x=0, xanchor="left", y=1.16, yanchor="top",
        bgcolor="rgba(0,0,0,0)", bordercolor=LIGHT_GRID, font=dict(size=11),
    )])
    fig.update_yaxes(title_text=f"{tickers[0]} (TRY)")
    add_insample_shading(fig, x, result["split_idx"])
    return chart_div(style_figure(fig, 380,
                                  "Each stock's own price, with its own buys and sells "
                                  "(pick a stock -- ordered by trade count)"))


def ticker_trade_table(per_ticker, tickers, family):
    """Which stocks the strategy actually traded, and how each leg did."""
    single_leg = family != "mean_reversion_ou"
    rows = ""
    for ticker in tickers:
        legs = per_ticker[ticker]
        moves = np.array([leg_move_pct(leg) for leg in legs], dtype=float)
        finite = moves[np.isfinite(moves)]
        n_long = sum(1 for leg in legs if leg["side"] == "long")
        avg_move = f"{finite.mean():+.2f}%" if len(finite) else "&mdash;"
        move_class = "pos" if len(finite) and finite.mean() >= 0 else "neg"
        amounts = np.array([leg["amount"] for leg in legs if leg.get("amount")], dtype=float)
        amount_cell = (f"<td>{fmt_amount(amounts.mean())}</td>" if len(amounts)
                       else "<td>&mdash;</td>")
        pnl_cell = ""
        if single_leg:
            total = sum(trade_pnl(leg["trade"]) for leg in legs)
            pnl_cell = (f"<td class='{'pos' if total > 0 else 'neg'}'>{total:+,.0f} TRY</td>"
                        if "pnl_currency" in legs[0]["trade"] else f"<td>{total:+.5f}</td>")
        rows += (f"<tr><td>{ticker}</td><td>{len(legs)}</td><td>{n_long}</td>"
                 f"<td>{len(legs) - n_long}</td>{amount_cell}"
                 f"<td class='{move_class}'>{avg_move}</td>{pnl_cell}</tr>")

    pnl_header = "<th>Attributed PnL</th>" if single_leg else ""
    note = ("" if single_leg else
            "<p class='hint'>A pair trade's PnL belongs to the spread, not to one leg, so it isn't "
            "split per stock here -- the side-adjusted move shows which leg carried it.</p>")
    return (f"<table><tr><th>Ticker</th><th>Legs traded</th><th>Long</th><th>Short</th>"
            f"<th>Avg amount traded</th><th>Avg move (side-adjusted)</th>{pnl_header}</tr>"
            f"{rows}</table>" + note)


def ticker_charts_section(result, family, prices, per_ticker):
    """
    The per-stock section: a price chart per traded stock carrying that
    stock's own trades, plus a per-stock summary. Answers "what did this
    strategy actually do in AKBNK" -- which neither the equity curve nor
    the spread's z-score can show, because both aggregate the legs away.
    """
    if prices is None or not per_ticker:
        return ""

    if family == "mean_reversion_ou":
        tickers = [t for t in (result["ticker_a"], result["ticker_b"]) if t in per_ticker]
        chart = ticker_stacked_chart(result, family, prices, per_ticker, tickers)
    else:
        ranked = sorted(per_ticker, key=lambda t: (-len(per_ticker[t]), t))
        tickers = ranked[:MAX_TICKER_PANELS]
        if len(ranked) > len(tickers):
            omitted = len(ranked) - len(tickers)
            note = (f"<p class='hint'>Showing the {len(tickers)} most-traded of {len(ranked)} stocks this "
                    f"run touched; {omitted} with fewer trades are in the trade log below but not the "
                    f"picker.</p>")
        else:
            note = ""
        chart = ticker_dropdown_chart(result, family, prices, per_ticker, tickers) + note

    return (f"<h2>Trades on each stock's own chart</h2>{chart}"
            f"<p class='hint'>Green stretches are bars this stock was held long, orange bars it was held "
            f"short; a triangle is a real buy or sell in this stock at the price the engine recorded. "
            f"Hover any marker for its trade number, side, traded amount and PnL.</p>"
            + ticker_trade_table(per_ticker, tickers, family))


def trade_contribution_chart(result):
    """
    Per-trade PnL as bars plus the running total: answers "is this edge
    spread across many trades, or is one outlier carrying the whole
    equity curve" -- which the headline return alone can't show.
    """
    trades = result["trades"]
    pnls = np.array([trade_pnl(t) for t in trades], dtype=float)
    tz = timestamp_tz(result)
    labels = [f"trade #{n}<br>{fmt_time(t['entry_time'], tz)} &rarr; {fmt_time(t['exit_time'], tz)}"
              f"<br>{t['exit_reason']}" for n, t in enumerate(trades, start=1)]
    n_axis = np.arange(1, len(pnls) + 1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=n_axis, y=pnls, name="per-trade PnL",
                         marker_color=[AQUA if p > 0 else RED for p in pnls],
                         text=labels, hovertemplate="%{text}<br>PnL %{y:+,.0f} TRY<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=n_axis, y=np.cumsum(pnls), mode="lines", name="cumulative",
                             line=dict(color=BLUE, width=1.6),
                             hovertemplate="cumulative %{y:+,.0f} TRY<extra></extra>"),
                  secondary_y=True)
    fig.update_xaxes(title_text="trade # (chronological)")
    fig.update_yaxes(title_text="PnL per trade (TRY)", secondary_y=False)
    fig.update_yaxes(title_text="cumulative (TRY)", secondary_y=True, showgrid=False)
    return chart_div(style_figure(fig, 300, time_axis=False), sync_x=False)


def leaderboard_chart(board):
    colors = [AQUA if d > 0.5 else (ORANGE if d > 0.2 else RED) for d in board["dsr"]]
    fig = go.Figure(go.Bar(
        x=board["dsr"], y=board["family"], orientation="h", marker_color=colors,
        text=[f"{d:.3f}" for d in board["dsr"]], textposition="outside",
        customdata=np.stack([board["sharpe"], board["total_return_pct"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>DSR %{x:.3f}<br>raw Sharpe %{customdata[0]:.3f}"
                      "<br>total return %{customdata[1]:+.2f}%<extra></extra>",
    ))
    fig.update_xaxes(title_text="Deflated Sharpe Ratio (P[true Sharpe > 0], trial-corrected)", range=[0, 1.08])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    return chart_div(style_figure(fig, 300, time_axis=False), sync_x=False)


def direction_label(direction, ticker_a, ticker_b):
    """
    "long/short spread" names neither of the two stocks actually traded.
    spread = log(a) - beta*log(b), so a short spread is sell a / buy b --
    same wording as intraday_pairs_trading/_engine.py's direction_label.
    """
    if direction == "short spread":
        return f"sell {ticker_a} / buy {ticker_b}"
    if direction == "long spread":
        return f"buy {ticker_a} / sell {ticker_b}"
    return direction


# grid template + header labels per family: the three engines log different
# fields (momentum has no z-score, mean reversion trades a pair rather than
# one ticker), so the card layout is per-family rather than one shape
# padded with blanks.
TRADE_COLUMNS = {
    "mean_reversion_ou": ("1.8fr 1.4fr 0.6fr 0.6fr 0.5fr 0.9fr 1fr 1.1fr",
                          ["Entry &rarr; Exit", "Direction", "Entry z", "Exit z", "Hold", "Reason",
                           "Amount", "PnL"]),
    "momentum": ("1.9fr 1fr 0.5fr 1fr 1fr 1.1fr",
                 ["Entry &rarr; Exit", "Ticker", "Hold", "Reason", "Amount", "PnL"]),
    # The variant family logs exactly what momentum logs -- one long leg per
    # trade, no z-score -- so it shares the layout rather than duplicating it.
    "momentum_variant": ("1.9fr 1fr 0.5fr 1fr 1fr 1.1fr",
                         ["Entry &rarr; Exit", "Ticker", "Hold", "Reason", "Amount", "PnL"]),
    "pca_stat_arb": ("1.8fr 1fr 0.6fr 0.6fr 0.5fr 0.9fr 1fr 1.1fr",
                     ["Entry &rarr; Exit", "Ticker", "Entry z", "Exit z", "Hold", "Reason",
                      "Amount", "PnL"]),
}


def attach_single_leg_prices(trades, prices):
    """
    Momentum and PCA log a ticker per trade but no quoted prices (unlike
    the mean-reversion engine, which attaches both legs' prices itself).
    Fill them in here from the same price panel the backtest ran on, so
    every family's trade card can be checked against real market data.
    Kept in the dashboard rather than in the engines: it's presentation
    only, and the engines' cached batch output (02/03) doesn't need it.
    """
    for t in trades:
        ticker = t.get("ticker")
        if ticker is None or ticker not in prices.columns:
            continue
        try:
            t["entry_price"] = float(prices.at[t["entry_time"], ticker])
            t["exit_price"] = float(prices.at[t["exit_time"], ticker])
        except (KeyError, ValueError, TypeError):
            continue
    return trades


def fmt_z(value):
    return f"{value:.2f}" if value is not None and pd.notna(value) else "&mdash;"


def trade_price_rows(legs):
    """
    The rows of a card's expanded detail table: how much each leg put to
    work, roughly how many shares that buys, and what it was actually
    quoted at on entry and exit. This is the part that makes a trade
    verifiable against real market data, and the reason trades are cards
    rather than one wide table -- carrying these columns on every row
    would push the table well past a readable width.
    """
    rows = ""
    for leg in legs:
        entry_price, exit_price = leg["entry_price"], leg["exit_price"]
        if leg["ticker"] is None or entry_price is None or exit_price is None:
            continue
        pct = (exit_price / entry_price - 1) * 100
        shares = f"{leg['shares']:,}" if leg.get("shares") else "&mdash;"
        rows += (f"<tr><td>{leg['ticker']}</td><td>{'buy' if leg['side'] == 'long' else 'sell'}</td>"
                 f"<td>{fmt_amount(leg.get('amount'))}</td><td>{shares}</td>"
                 f"<td>{entry_price:.2f}</td><td>{exit_price:.2f}</td>"
                 f"<td class='{'pos' if pct >= 0 else 'neg'}'>{pct:+.2f}%</td></tr>")
    return rows


def render_trade_cards(result, family, per_trade, limit=400):
    trades = result["trades"]
    if not trades:
        return ("<p>No trades were entered with these parameters -- try loosening entry_z "
                "or the trend/regime filter.</p>")

    tz = timestamp_tz(result)
    cols, headers = TRADE_COLUMNS[family]
    pnls = np.array([trade_pnl(t) for t in trades], dtype=float)
    hold = np.array([t["hold_bars"] for t in trades], dtype=float)
    reasons = pd.Series([t["exit_reason"] for t in trades]).value_counts()

    summary = f"""<p class="hint">{len(trades)} trades &middot; win rate {(pnls > 0).mean() * 100:.0f}%
    &middot; avg hold {hold.mean():.0f} bars &middot; best {pnls.max():+,.0f} TRY / worst {pnls.min():+,.0f} TRY
    &middot; exit reasons: {", ".join(f"{k} {int(v)}" for k, v in reasons.items())}
    &middot; click a row to see each leg's traded amount, share count and actual entry/exit price.</p>"""

    header = "".join(f"<span>{h}</span>" for h in headers)
    cards = []
    for n, t in enumerate(trades[:limit], start=1):
        legs = per_trade.get(n, [])
        # Gross across both sides for a pair: a market-neutral spread
        # commits capital on the long AND the short leg, so one number for
        # "the amount traded" has to say which it is.
        amounts = [leg["amount"] for leg in legs if leg.get("amount")]
        amount_text = fmt_amount(sum(amounts)) if amounts else "&mdash;"
        pnl_value = trade_pnl(t)
        pnl_text = f"{pnl_value:+,.0f} TRY" if "pnl_currency" in t else f"{pnl_value:+.5f} (raw units)"
        times = f"{fmt_time(t['entry_time'], tz)} &rarr; {fmt_time(t['exit_time'], tz)}"

        if family == "mean_reversion_ou":
            cells = [times, direction_label(t["direction"], result["ticker_a"], result["ticker_b"]),
                     fmt_z(t.get("entry_z")), fmt_z(t.get("exit_z")), t["hold_bars"], t["exit_reason"],
                     amount_text]
        elif family in ("momentum", "momentum_variant"):
            cells = [times, t["ticker"], t["hold_bars"], t["exit_reason"], amount_text]
        else:
            cells = [times, t["ticker"], fmt_z(t.get("entry_z")), fmt_z(t.get("exit_z")),
                     t["hold_bars"], t["exit_reason"], amount_text]

        price_rows = trade_price_rows(legs)
        detail = (f"""<table><tr><th></th><th>Side</th><th>Amount</th><th>Shares</th>
                  <th>Entry price</th><th>Exit price</th><th>Change</th></tr>
                  {price_rows}</table>""" if price_rows else
                  "<p class='hint'>No quoted prices recorded for this trade.</p>")
        if "dollar_per_unit" in t:
            multi_leg = len(legs) > 1
            detail += f"""<p class="hint">Amount above is what this trade put to work at entry:
            the base unit of <strong>{t['dollar_per_unit']:,.0f} TRY</strong> (out of
            {t['starting_capital']:,.0f} TRY starting capital) times the engine's own sizing at that bar
            {'-- the vol-target scale, and the hedge ratio on the second leg' if multi_leg
             else '-- an equal share of the book across the names held at the time'}.
            {'The two legs are gross exposure on both sides of a market-neutral spread, not two independent bets. '
             if multi_leg else ''}Sizing is off the STARTING capital, not the equity at the time, and it is
            not held fixed through the trade: this {pnl_text} reflects the whole path, not a single stake.
            Share counts are the amount divided by the quoted price -- the backtest itself sizes in
            continuous units and never rounds to a lot.</p>"""

        cards.append(f"""<details class="trade-card"><summary>
          {"".join(f"<span>{v}</span>" for v in cells)}
          <span class="{'pos' if pnl_value > 0 else 'neg'}">{pnl_text}</span>
        </summary><div class="trade-detail">{detail}</div></details>""")

    note = (f"<p class='hint'>Showing the first {limit} of {len(trades)} trades.</p>"
            if len(trades) > limit else "")
    return (summary + f'<div class="trade-cards-wrap" style="--cols: {cols}">'
            f'<div class="trade-cards-header">{header}</div>'
            f'<div class="trade-cards">{"".join(cards)}</div></div>' + note)


UPGRADE_BENCH_PKL = pathlib.Path(__file__).parent / "_upgrade_bench.pkl"


def phase_distribution_chart(variants, benchmark_sharpe):
    """
    Every variant's whole rebalance-phase distribution, one row each.

    The point of showing all 45 phases rather than a bar per variant: a
    variant whose points span 0.1 to 1.8 has not got a Sharpe of 1.2, it has
    a coin flip whose average happens to be 1.2. The benchmark is a single
    line because buy-and-hold has no rebalance phase to be lucky about.
    """
    fig = go.Figure()
    for record in variants:
        sharpes = np.asarray(record["phase_sharpes"], dtype=float)
        fig.add_trace(go.Box(
            x=sharpes, name=record["name"], orientation="h",
            boxpoints="all", jitter=0.5, pointpos=0, marker=dict(size=4, color=BLUE),
            line=dict(color=GRAY, width=1), fillcolor="rgba(42,120,214,0.10)",
            hovertemplate="%{x:.2f}<extra>" + record["name"] + "</extra>",
        ))
    fig.add_vline(x=benchmark_sharpe, line=dict(color=ORANGE, width=2, dash="dash"),
                  annotation_text="buy and hold", annotation_position="top",
                  annotation_font_size=10, annotation_font_color=ORANGE)
    fig.update_xaxes(title_text="Sharpe, out-of-sample (one point per rebalance phase)")
    fig.update_layout(showlegend=False)
    return chart_div(style_figure(fig, 60 + 42 * len(variants),
                                  "Phase distribution: how much of each result is the calendar",
                                  time_axis=False), sync_x=False)


def robustness_heatmap(record, benchmark_sharpe):
    """
    One variant across its OWN parameters. This is the chart that decides
    whether a rule is real: a plateau of similar colours means the rule works
    over a range of settings, while one bright cell surrounded by dark ones
    means the backtest found a coincidence. Cells are centred on the
    benchmark Sharpe, so anything below buy-and-hold reads cold.
    """
    grid = record.get("grid")
    if not grid:
        return ""
    sharpe = np.asarray(grid["sharpe"], dtype=float)
    columns = ["" if c is None else str(c) for c in grid["cols"]]
    rows = [str(r) for r in grid["rows"]]

    fig = go.Figure(go.Heatmap(
        z=sharpe, x=columns, y=rows, zmid=benchmark_sharpe,
        colorscale=[[0.0, RED], [0.5, "#f2efe6"], [1.0, AQUA]],
        colorbar=dict(title=dict(text="Sharpe", side="right"), thickness=12),
        text=[["%.2f" % value if np.isfinite(value) else "" for value in row] for row in sharpe],
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate=(grid["row_label"] + " %{y}<br>" + grid["col_label"] +
                       " %{x}<br>phase-neutral Sharpe %{z:.2f}<extra></extra>"),
    ))
    fig.update_xaxes(title_text=grid["col_label"], type="category")
    fig.update_yaxes(title_text=grid["row_label"], type="category")
    finite = sharpe[np.isfinite(sharpe)]
    verdict = ""
    if len(finite):
        if finite.min() < 0:
            verdict = (" &mdash; <span class='neg'>turns negative inside its own settings: "
                       "a fitted cell, not an edge</span>")
        elif finite.min() > benchmark_sharpe:
            verdict = " &mdash; <span class='pos'>beats the benchmark everywhere tested</span>"
    return (f"<h3>{record['name']} &middot; {record['label']}{verdict}</h3>"
            + chart_div(style_figure(fig, 300, None, time_axis=False), sync_x=False))


def equity_comparison_chart(cache, selected):
    """Phase-neutral equity of the selected variants against buy-and-hold, after 5 bps."""
    timestamps = pd.DatetimeIndex(cache["timestamps"])
    try:
        x = timestamps.tz_localize(None)
    except TypeError:
        x = timestamps
    split = cache["split_idx"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[split:], y=cache["benchmark"]["equity"][split:], mode="lines",
                             name="buy and hold", line=dict(color=ORANGE, width=2, dash="dash"),
                             hovertemplate="%{y:,.0f} TRY<extra>buy and hold</extra>"))
    for record in selected:
        fig.add_trace(go.Scatter(x=x[split:], y=record["equity_after_cost"][split:], mode="lines",
                                 name=record["name"], line=dict(width=1.4),
                                 hovertemplate="%{y:,.0f} TRY<extra>" + record["name"] + "</extra>"))
    fig.update_yaxes(title_text="equity (TRY), 5 bps costs")
    return chart_div(style_figure(fig, 360,
                                  "Phase-neutral equity vs buy-and-hold (out-of-sample, after costs)"))


def render_robustness_tab(selected_name=None):
    if not UPGRADE_BENCH_PKL.exists():
        return ("<p>No upgrade bench found. Run <code>python 05_run_upgrade_bench.py</code> first "
                "(a few minutes: every variant is 45 phase runs plus a parameter grid).</p>")

    with open(UPGRADE_BENCH_PKL, "rb") as handle:
        cache = pickle.load(handle)

    variants = cache["variants"]
    benchmark = cache["benchmark"]["stats"]
    benchmark_sharpe = benchmark["sharpe (naive)"]
    span_start, span_end = cache["panel_span"]

    body = f"""
    <p class="hint">Every figure is out-of-sample and <strong>phase-neutral</strong>: the book is
    staggered across all {cache['params']['rebalance']} rebalance offsets, which is the number that
    does not change when you re-run it. A single backtest of this strategy swings from 13% to 57%
    annualized depending only on which bar the rebalance grid lands on, so single backtests are not
    shown here at all. Panel: {cache['panel_shape'][0]:,} bars &times; {cache['panel_shape'][1]} tickers,
    {span_start:%Y-%m-%d} to {span_end:%Y-%m-%d}.</p>"""

    rows = ""
    for record in sorted(variants, key=lambda r: -r["tranched"]["sharpe (naive)"]):
        grid = record.get("grid")
        finite = (np.asarray(grid["sharpe"], dtype=float)[np.isfinite(grid["sharpe"])]
                  if grid else np.array([]))
        nbhd_median = np.median(finite) if len(finite) else float("nan")
        nbhd_worst = finite.min() if len(finite) else float("nan")
        beats = record["tranched"]["sharpe (naive)"] > benchmark_sharpe
        fragile = len(finite) and nbhd_worst < 0
        rows += (
            f"<tr class=\"{'top-family' if beats and not fragile else ''}\">"
            f"<td><a href='?tab=robustness&variant={record['name']}'>{record['name']}</a></td>"
            f"<td>{record['label']}</td>"
            f"<td class='{'pos' if beats else ''}'>{record['tranched']['annualized return'] * 100:+.1f}%</td>"
            f"<td class='{'pos' if beats else ''}'>{record['tranched']['sharpe (naive)']:.2f}</td>"
            f"<td>{record['costed']['sharpe (naive)']:.2f}</td>"
            f"<td>{'&mdash;' if not len(finite) else f'{nbhd_median:.2f}'}</td>"
            f"<td class='{'neg' if fragile else ''}'>"
            f"{'&mdash;' if not len(finite) else f'{nbhd_worst:.2f}'}</td>"
            f"<td>{record['avg_held']:.1f}</td><td>{record['turnover_per_year']:.0f}x</td></tr>")

    rows += (f"<tr><td><strong>benchmark</strong></td><td>equal-weight all "
             f"{cache['panel_shape'][1]}, no trading</td>"
             f"<td>{benchmark['annualized return'] * 100:+.1f}%</td>"
             f"<td>{benchmark_sharpe:.2f}</td><td>{benchmark_sharpe:.2f}</td>"
             f"<td>{benchmark_sharpe:.2f}</td><td>{benchmark_sharpe:.2f}</td>"
             f"<td>{cache['panel_shape'][1]}</td><td>0x</td></tr>")

    body += f"""<h2>Candidates</h2>
    <table><tr><th>Variant</th><th>Rule</th><th>Return</th><th>Sharpe</th><th>Sharpe @5bps</th>
    <th>Nbhd median</th><th>Nbhd worst</th><th>Avg held</th><th>Turnover/yr</th></tr>{rows}</table>
    <p class="hint"><strong>Nbhd median / worst</strong> are the same rule re-measured across its own
    parameter grid. They matter more than the headline: a variant that beats the benchmark at one
    threshold and goes negative at the next one has found a cell in a table, not a market property.
    Click a variant name for its parameter grid.</p>"""

    body += "<h2>Phase distributions</h2>" + phase_distribution_chart(variants, benchmark_sharpe)

    shortlist = [r for r in variants
                 if r["tranched"]["sharpe (naive)"] > benchmark_sharpe * 0.85][:4]
    if shortlist:
        body += "<h2>Equity vs benchmark</h2>" + equity_comparison_chart(cache, shortlist)

    chosen = [r for r in variants if r["name"] == selected_name] if selected_name else []
    grid_records = chosen if chosen else [r for r in variants if r.get("grid")]
    body += ("<h2>Parameter grids &mdash; plateau or single cell?</h2>"
             "<p class='hint'>Each cell is the phase-neutral Sharpe at that setting. Green is above "
             "buy-and-hold, red below. Look for a block of colour, not a bright square.</p>")
    for record in grid_records:
        body += robustness_heatmap(record, benchmark_sharpe)

    body += f"""<p class="hint">Bench built {pd.Timestamp(cache['generated_at'], unit='s'):%Y-%m-%d %H:%M}
    by <code>05_run_upgrade_bench.py</code>. Re-run it after changing the universe, the parameters or
    the data window.</p>"""
    return body


def render_leaderboard_tab():
    if not LEADERBOARD_PKL.exists():
        return ("<p>No cached leaderboard found. Run <code>02_run_full_universe_backtests.py</code> "
                "then <code>03_run_validation_leaderboard.py</code> from the project root first.</p>")

    with open(LEADERBOARD_PKL, "rb") as f:
        cache = pickle.load(f)
    board, evaluations = cache["board"], cache["evaluations"]
    bootstrap, sensitivity = cache.get("bootstrap"), cache.get("sensitivity")

    rows = "".join(
        f"""<tr class="{'top-family' if i == 0 else ''}">
        <td>{r['family']}</td><td>{r['sharpe']:.3f}</td>
        <td><strong>{r['dsr']:.3f}</strong></td>
        <td>{r['total_return_pct']:+.2f}%</td><td>{r['max_drawdown']*100:.1f}%</td>
        <td>{r['n_units_tested']}</td><td>{r['pct_units_profitable']:.0f}%</td>
        </tr>"""
        for i, r in board.iterrows()
    )
    body = f"""
    <p class="hint">n_trials = {cache['n_trials']} (see WORKFLOW.md's "Validation methodology" for the accounting).
    Ranked by Deflated Sharpe Ratio (DSR) -- the probability the true Sharpe is &gt; 0 AFTER discounting for
    having tried {cache['n_trials']} strategy/parameter combinations -- not by raw Sharpe or total return.</p>
    {leaderboard_chart(board)}
    <table><tr><th>Family</th><th>Sharpe (raw)</th><th>DSR</th><th>Total return</th>
    <th>Max drawdown</th><th>Units tested</th><th>% units profitable</th></tr>{rows}</table>
    """

    if bootstrap:
        body += f"""<h2>Bootstrap significance -- top-ranked family ({cache['top_family']})</h2>
        <p>Moving block bootstrap (2000 resamples): median Sharpe {bootstrap['sharpe_median']:.3f}
        (90% CI {bootstrap['sharpe_ci_low']:.3f} to {bootstrap['sharpe_ci_high']:.3f}),
        P(Sharpe &gt; 0) = <strong>{bootstrap['p_sharpe_gt_0']:.3f}</strong>.</p>"""

    if sensitivity is not None and len(sensitivity.get("perturbations", [])):
        pert = sensitivity["perturbations"]
        pert_rows = "".join(
            f"<tr><td>{r['param']}</td><td>{r['multiplier']}x</td>"
            f"<td>{r['sharpe']:.3f}</td><td>{r['sharpe_delta']:+.3f}</td></tr>"
            for _, r in pert.iterrows()
        )
        body += f"""<h2>Parameter-perturbation sensitivity -- {cache['top_family']}</h2>
        <p class="hint">Base Sharpe: {sensitivity['base_sharpe']:.3f}. Large deltas from a small nudge are a red
        flag for overfitting -- read alongside DSR and cross-universe consistency, not alone.</p>
        <table><tr><th>Param</th><th>Multiplier</th><th>Sharpe</th><th>Delta</th></tr>{pert_rows}</table>"""

    for family, ev in evaluations.items():
        per_unit = ev["per_unit"]
        if len(per_unit):
            body += f"<h2>{LEADERBOARD_FAMILY_LABELS.get(family, family)} -- per-unit detail</h2>"
            body += per_unit.round(4).to_html(index=False, classes="", border=0)

    return body


def render_explore_tab(params):
    family = params["family"]
    conf = FAMILIES[family]
    fields = FIELDS_BY_FAMILY[family]

    field_html = "".join(
        f"""<label>{label}<input type="number" name="{key}" value="{params['fparams'][key]}"
        min="{lo}" max="{hi}" step="any"></label>"""
        for key, label, lo, hi, _type in fields
    )

    family_options = "".join(
        f'<option value="{key}" {"selected" if key == family else ""}>{c2["label"]}</option>'
        for key, c2 in FAMILIES.items()
    )
    period_options = "".join(
        f'<option value="{p}" {"selected" if p == params["period"] else ""}>{p}</option>'
        for p in PERIOD_OPTIONS
    )

    pair_field_html = ""
    if family == "mean_reversion_ou" and params.get("screening") is not None:
        screening = params["screening"]
        qualifying = screening[screening["qualifies"]]
        selected_pair = params.get("pair", "auto")

        def option(row):
            value = f"{row['ticker_a']}|{row['ticker_b']}"
            label = f"{row['pair']} (ADF={row['adf_stat']:.2f}, H={row['H']:.2f}, half-life={row['half_life_days']:.1f}d)"
            sel = "selected" if value == selected_pair else ""
            return f'<option value="{value}" {sel}>{label}</option>'

        qualifying_options = "".join(option(r) for _, r in qualifying.iterrows())
        auto_sel = "selected" if selected_pair == "auto" else ""
        pair_field_html = f"""<label style="grid-column: 1 / -1">
        Pair ({len(qualifying)}/{len(screening)} confirmed at current thresholds)
        <select name="pair">
          <option value="auto" {auto_sel}>Auto (best confirmed pair by ADF stat)</option>
          {qualifying_options or '<option disabled>none qualify at these thresholds</option>'}
        </select></label>"""

    # The variant family's rule is itself a choice, so it gets a picker
    # rather than being buried in the numeric fields. Changing it re-runs
    # immediately: with eleven rules, submitting by hand each time is the
    # kind of friction that stops people from actually comparing them.
    variant_field_html = ""
    if family == "momentum_variant":
        selected_variant = params.get("variant", ev.DEFAULTS["variant"])
        options = "".join(
            f'<option value="{name}" {"selected" if name == selected_variant else ""}>'
            f'{name} &mdash; {label}</option>'
            for name, label in ev.VARIANT_LABELS.items())
        variant_field_html = f"""<label style="grid-column: 1 / -1">
        Selection rule ({len(ev.VARIANT_LABELS)} candidates from the upgrade bench)
        <select name="variant" onchange="this.form.submit()">{options}</select></label>
        <p class="hint" style="grid-column: 1 / -1">Every rule keeps the same universe, basket size
        and rebalance clock &mdash; only the way names are chosen changes, so a difference in the
        result is attributable to the rule. See the <a href="?tab=robustness">Robustness</a> tab for
        how each one behaves across its own parameter grid before reading much into one backtest.</p>"""

    cost_calib_html = ""
    if family == "mean_reversion_ou":
        checked = "checked" if params.get("derive_entry_z_from_cost") else ""
        cost_bps = params.get("cost_per_round_trip_bps", 5)
        cost_calib_html = f"""
        <label><input type="checkbox" name="derive_entry_z_from_cost" value="1" {checked}
          onchange="this.form.submit()"> Cost-calibrate entry_z per pair (Monte Carlo, see WORKFLOW.md)</label>
        <label>Assumed round-trip cost (bps)<input type="number" name="cost_per_round_trip_bps"
          value="{cost_bps:g}" min="0" max="500" step="any"></label>"""

    body = f"""
    <form method="get">
      <input type="hidden" name="tab" value="explore">
      <fieldset>
        <legend>Strategy family &amp; data</legend>
        <div class="field-grid">
          <label>Family<select name="family" onchange="this.form.submit()">{family_options}</select></label>
          <label>History length<select name="period">{period_options}</select></label>
          <label>Starting capital (TRY)<input type="number" name="starting_capital"
            value="{params['starting_capital']:g}" min="1000" max="10000000" step="any"></label>
        </div>
      </fieldset>
      <fieldset><legend>{"Selection rule" if family == "momentum_variant" else "Pair selection"}</legend>
        <div class="field-grid">{variant_field_html}{pair_field_html}</div></fieldset>
      <fieldset><legend>Parameters</legend><div class="field-grid">{field_html}{cost_calib_html}</div></fieldset>
      <button type="submit">Re-run backtest</button>
    </form>
    """

    if params.get("error"):
        return body + f'<div class="error"><strong>Error:</strong> {params["error"]}</div>'

    result = params.get("result")
    if result is None:
        return body + "<p>Set parameters above and click Re-run to see results.</p>"

    stats = result["stats"]
    tiles = [
        ("Starting capital", f"{result['starting_capital']:,.0f} TRY", ""),
        ("Final capital (no cost)", f"{result['final_capital']:,.0f} TRY",
         "pos" if result["final_capital"] >= result["starting_capital"] else "neg"),
        ("Total return", f"{result['total_return_pct']:+.1f}%",
         "pos" if result["total_return_pct"] >= 0 else "neg"),
        ("Sharpe (no cost)", f"{stats['sharpe (naive)']:.2f}", ""),
        ("Max drawdown", f"{stats['max drawdown']*100:.1f}%", ""),
        ("Trades", result["n_trades"], ""),
    ]
    if "ticker_a" in result:
        tiles.insert(0, ("Pair", f"{result['ticker_a']}/{result['ticker_b']}", ""))
        tiles.append(("% disqualified", f"{result.get('pct_disqualified', 0):.0f}%", ""))
        entry_z_used = result["results"]["entry_z"].dropna() if "entry_z" in result["results"] else None
        if entry_z_used is not None and len(entry_z_used):
            tiles.append(("entry_z used (mean)", f"{entry_z_used.mean():.2f}", ""))
    if "variant" in result:
        tiles.insert(0, ("Rule", result["variant"], ""))
    if "avg_n_held" in result:
        tiles.append(("Avg names held", result["avg_n_held"], ""))
    if "pct_idle" in result:
        tiles.append(("Bars holding nothing", f"{result['pct_idle']:.0f}%", ""))
    if "avg_n_active" in result:
        tiles.append(("Avg names active", result["avg_n_active"], ""))

    body += '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value {cls}">{value}</div></div>'
        for label, value, cls in tiles
    ) + "</div>"

    body += f"""<h2>Equity curve{' and signal' if family == 'mean_reversion_ou' else ''}</h2>
    {strategy_chart(result, family, conf['label'])}
    <p class="hint">Drag to zoom, double-click to reset, scroll to zoom the x axis; every chart below that runs
    on the bar timeline follows the same range. Base unit: {result['dollar_per_unit']:,.2f} TRY per abstract
    position unit (starting capital &divide; scale_max={result.get('scale_max', 1.0):g}); each trade's own
    traded amount is that unit times the engine's sizing at entry -- see the Amount column in the trade log.
    Sizing is off starting capital, so it never compounds, and share counts are indicative: the backtest
    sizes in continuous units and never rounds to a lot.</p>"""

    # Computed once and shared: the per-stock charts, the per-stock table and
    # the trade cards all need the same legs and the same traded amounts.
    per_ticker, per_trade = enriched_legs(result, family) if result["trades"] else ({}, {})

    body += ticker_charts_section(result, family, params.get("prices"), per_ticker)

    if result["trades"]:
        body += f"""<h2>Trade contribution</h2>{trade_contribution_chart(result)}"""

    cost_rows = "".join(
        f"<tr class='{'neg' if row['sharpe (naive)'] <= 0 else ''}'><td>{bps}</td>"
        f"<td>{row['annualized return']*100:.1f}%</td><td>{row['sharpe (naive)']:.2f}</td></tr>"
        for bps, row in result["cost_table"].iterrows()
    )
    body += f"""<h2>Cost sensitivity</h2>
    <table><tr><th>bps/switch</th><th>Ann. return</th><th>Sharpe</th></tr>{cost_rows}</table>"""

    body += "<h2>Positions opened (trade log)</h2>" + render_trade_cards(result, family, per_trade)

    if "screening" in result:
        body += f"""<h2>Pair screening (in-sample, top 10)</h2>
        <table><tr><th>Pair</th><th>Beta</th><th>ADF stat</th><th>H</th><th>Half-life (d)</th><th>Qualifies</th></tr>{"".join(
            f"<tr><td>{r['pair']}</td><td>{r['beta']:.3f}</td><td>{r['adf_stat']:.3f}</td>"
            f"<td>{r['H']:.3f}</td><td>{r['half_life_days']:.1f}</td><td>{'checkmark' if r['qualifies'] else ''}</td></tr>"
            for _, r in result['screening'].head(10).iterrows()
        )}</table>"""

    return body


# --cols is set per family on the wrapper (see TRADE_COLUMNS) so one
# stylesheet serves all three card layouts. summary::before is positioned
# absolutely on purpose: as a normal first child it would become a grid
# item and shift every real column one slot over.
TRADE_CARD_CSS = """
  .trade-cards { display: flex; flex-direction: column; gap: 4px; }
  .trade-card { border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); }
  .trade-card summary {
    cursor: pointer; padding: 7px 10px 7px 22px; font-size: 0.78rem; list-style: none;
    display: grid; grid-template-columns: var(--cols); gap: 8px; align-items: center; position: relative;
  }
  .trade-card summary::-webkit-details-marker { display: none; }
  .trade-card summary::before {
    content: '\\25B8'; color: #8a8a86; position: absolute; left: 8px; top: 9px;
  }
  .trade-card[open] summary::before { content: '\\25BE'; }
  .trade-cards-header {
    display: grid; grid-template-columns: var(--cols); gap: 8px;
    padding: 4px 10px 4px 26px; font-size: 0.72rem; color: #8a8a86;
  }
  .trade-detail { padding: 6px 14px 10px 26px; border-top: 1px solid var(--border); font-size: 0.78rem; }
  .trade-detail table { width: auto; min-width: 320px; margin-bottom: 4px; }
"""


PAGE_CSS = f"""
  :root {{ --surface:#fcfcfb; --surface-2:#f3f2ef; --border:#e4e2dd; --text:#0b0b0b; --text-2:#52514e; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface:#1a1a19; --surface-2:#232320; --border:#34332f; --text:#ffffff; --text-2:#c3c2b7; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--surface); color: var(--text); font-family: -apple-system, sans-serif; margin: 0; padding: 28px; }}
  h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
  h2 {{ font-size: 0.95rem; margin: 24px 0 10px; border-top: 1px solid var(--border); padding-top: 20px; }}
  .subtitle {{ color: var(--text-2); font-size: 0.85rem; margin-bottom: 16px; }}
  .hint {{ color: var(--text-2); font-size: 0.78rem; margin: 4px 0 0; }}
  .tabs {{ display: flex; gap: 6px; margin-bottom: 18px; }}
  .tabs a {{ padding: 7px 14px; border-radius: 6px; text-decoration: none; font-size: 0.85rem;
    color: var(--text-2); border: 1px solid var(--border); }}
  .tabs a.active {{ background: {BLUE}; color: white; border-color: {BLUE}; }}
  form {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
  fieldset {{ border: none; border-top: 1px solid var(--border); margin: 0 0 14px; padding: 12px 0 0; }}
  fieldset:first-of-type {{ border-top: none; padding-top: 0; }}
  legend {{ font-size: 0.78rem; font-weight: 600; color: var(--text); padding: 0; margin-bottom: 8px; }}
  .field-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
  label {{ display: flex; flex-direction: column; font-size: 0.78rem; color: var(--text-2); gap: 4px; }}
  input, select {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 6px; font-family: inherit; }}
  button {{ width: 100%; background: {BLUE}; color: white; border: none; border-radius: 6px; padding: 10px; font-size: 0.9rem; cursor: pointer; margin-top: 14px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-bottom: 20px; }}
  .tile {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .tile-label {{ font-size: 0.72rem; color: var(--text-2); }}
  .tile-value {{ font-size: 1.15rem; font-weight: 600; }}
  .chart {{ border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2);
    padding: 4px 2px; margin-bottom: 6px; overflow: hidden; }}
  .modebar {{ background: transparent !important; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-bottom: 12px; }}
  th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-2); font-weight: 500; }}
  tr.top-family {{ background: color-mix(in srgb, {AQUA} 10%, var(--surface)); }}
  .neg {{ color: {RED}; }}
  .pos {{ color: {AQUA}; }}
  .error {{ background: color-mix(in srgb, {RED} 15%, var(--surface)); border: 1px solid {RED}; border-radius: 8px; padding: 12px 16px; }}
  {TRADE_CARD_CSS}
"""


# Plain (non-f) string: the braces below are JavaScript's, not format slots.
#
# Two jobs Plotly can't do from the server side. (1) Theme: a figure's
# colors are baked into its JSON, so the page re-colors every chart to
# match prefers-color-scheme, and again whenever the OS theme flips.
# (2) X-axis sync: plotly.js shares an axis only WITHIN one figure, but the
# equity curve, the z-score and each stock's price panel are separate
# figures over the same bar timeline -- zooming one and having the others
# stay put would make them impossible to read against each other. Charts
# whose x axis is NOT the timeline (trade #, DSR) are left out via the
# data-xsync attribute, so they can't push a nonsense range onto the rest.
PAGE_JS = """
(function () {
  const timelineCharts = () => Array.from(
    document.querySelectorAll('.chart[data-xsync] .plotly-graph-div'));
  const allCharts = () => Array.from(document.querySelectorAll('.chart .plotly-graph-div'));

  function palette() {
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return dark ? { text: '#c3c2b7', grid: '#34332f' } : { text: '#52514e', grid: '#e4e2dd' };
  }

  function applyTheme() {
    const p = palette();
    allCharts().forEach(function (el) {
      const update = { 'font.color': p.text };
      Object.keys(el.layout || {}).forEach(function (key) {
        if (/^[xy]axis/.test(key)) {
          update[key + '.gridcolor'] = p.grid;
          update[key + '.linecolor'] = p.grid;
        }
      });
      Plotly.relayout(el, update);
    });
  }

  let syncing = false;
  function syncFrom(source, ev) {
    if (syncing) return;
    let update = null;
    if ('xaxis.range[0]' in ev) update = { 'xaxis.range': [ev['xaxis.range[0]'], ev['xaxis.range[1]']] };
    else if (ev['xaxis.autorange'] === true) update = { 'xaxis.autorange': true };
    else if (ev['xaxis.range']) update = { 'xaxis.range': ev['xaxis.range'] };
    if (!update) return;
    const others = timelineCharts().filter(function (el) { return el !== source; });
    if (!others.length) return;
    syncing = true;
    Promise.all(others.map(function (el) { return Plotly.relayout(el, update); }))
      .catch(function () { })
      .then(function () { syncing = false; });
  }

  function init() {
    if (typeof Plotly === 'undefined') return;
    timelineCharts().forEach(function (el) {
      if (el.on) el.on('plotly_relayout', function (ev) { syncFrom(el, ev); });
    });
    applyTheme();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
  }

  if (document.readyState === 'complete') init();
  else window.addEventListener('load', init);
})();
"""

_plotlyjs_cache = {}


@app.route("/plotly.js")
def plotly_js():
    """
    plotly.js served from the installed package, once, with a cache header
    -- not re-inlined into every figure. The bundle is ~5 MB and the
    Explore tab renders several figures; inlining it per figure would put
    tens of MB on the wire per page load. Served locally rather than from a
    CDN so the Leaderboard tab still works with no network (it reads only
    03's cache).
    """
    if "js" not in _plotlyjs_cache:
        _plotlyjs_cache["js"] = get_plotlyjs()
    return Response(_plotlyjs_cache["js"], mimetype="application/javascript",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route("/", methods=["GET"])
def index():
    tab = request.args.get("tab", default="leaderboard")

    if tab == "leaderboard":
        content = render_leaderboard_tab()
    elif tab == "robustness":
        content = render_robustness_tab(request.args.get("variant"))
    else:
        family = request.args.get("family", default="mean_reversion_ou")
        if family not in FAMILIES:
            family = "mean_reversion_ou"
        period = request.args.get("period", default="730d")
        if period not in PERIOD_OPTIONS:
            period = "730d"
        starting_capital = request.args.get("starting_capital", type=float, default=100_000.0)

        conf = FAMILIES[family]
        fields = FIELDS_BY_FAMILY[family]
        default_params = dict(conf["module"].DEFAULTS)
        fparams = {
            key: request.args.get(key, type=_type, default=default_params[key])
            for key, label, lo, hi, _type in fields
        }
        pair_param = request.args.get("pair", default="auto")

        params = {"family": family, "period": period, "starting_capital": starting_capital, "fparams": fparams}

        if family == "momentum_variant":
            variant = request.args.get("variant", default=ev.DEFAULTS["variant"])
            if variant not in ev.VARIANT_KINDS:
                variant = ev.DEFAULTS["variant"]
            params["variant"] = variant

        if family == "mean_reversion_ou":
            params["derive_entry_z_from_cost"] = request.args.get("derive_entry_z_from_cost", default="") == "1"
            params["cost_per_round_trip_bps"] = request.args.get(
                "cost_per_round_trip_bps", type=float, default=default_params["cost_per_round_trip_bps"]
            )

        try:
            prices = get_prices(period)
            bpy = uni.bars_per_year(prices.index)
            merged_params = {**default_params, **fparams}
            if family == "mean_reversion_ou":
                merged_params["derive_entry_z_from_cost"] = params["derive_entry_z_from_cost"]
                merged_params["cost_per_round_trip_bps"] = params["cost_per_round_trip_bps"]

            if family == "mean_reversion_ou":
                screening, split_idx, sess, log_prices = em.screen_only(prices, merged_params, bpy / 252)
                params["screening"] = screening
                selected_pair = None
                if pair_param != "auto":
                    ta, tb = pair_param.split("|")
                    selected_pair = (ta, tb)
                result = em.run_full_backtest_meanrev(
                    prices, params=merged_params, bars_per_year=bpy,
                    screening_result=(screening, split_idx, sess, log_prices), selected_pair=selected_pair,
                )
                params["pair"] = pair_param
            elif family == "momentum_variant":
                # This family reads highs and lows, so it gets the OHLC panel
                # rather than the close-only one the others share.
                merged_params["variant"] = params["variant"]
                panel = get_ohlc(period)
                prices = panel["Close"]
                # Measured from THIS panel: the OHLC request can return a
                # slightly different bar count than the close-only one, and
                # annualization must follow the data actually traded.
                bpy = uni.bars_per_year(prices.index)
                result = ev.run_full_backtest_variant(panel, params=merged_params, bars_per_year=bpy)
            else:
                result = conf["run"](prices, params=merged_params, bars_per_year=bpy)

            result = c.add_capital_pnl(result, starting_capital)
            attach_single_leg_prices(result["trades"], prices)
            params["result"] = result
            # Kept for the per-stock price panels: they plot each traded
            # ticker's own close on the backtest's own bar index.
            params["prices"] = prices
        except Exception as e:
            params["error"] = str(e)

        content = render_explore_tab(params)

    # Returned as-is rather than through render_template_string: the page
    # now embeds Plotly's own generated JSON/JS, and there is no template
    # variable left to substitute -- running a Jinja pass over machine-
    # generated braces would only risk mangling a figure.
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BIST30 Strategy Lab</title>
<style>{PAGE_CSS}</style>
<script src="/plotly.js"></script>
</head><body>
<h1>BIST30 Strategy Lab</h1>
<div class="subtitle">Statistical models, simulation-based validation, and technical strategy comparison
across the real BIST30 universe -- see WORKFLOW.md.</div>
<div class="tabs">
  <a href="?tab=leaderboard" class="{'active' if tab == 'leaderboard' else ''}">Leaderboard</a>
  <a href="?tab=explore" class="{'active' if tab == 'explore' else ''}">Explore</a>
  <a href="?tab=robustness" class="{'active' if tab == 'robustness' else ''}">Robustness</a>
</div>
{content}
<script>{PAGE_JS}</script>
</body></html>
"""
    return page


if __name__ == "__main__":
    print("Starting BIST30 Strategy Lab at http://127.0.0.1:5100")
    print("(Ctrl+C to stop)")
    app.run(port=5100, debug=False)
