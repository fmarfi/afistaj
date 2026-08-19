#!/usr/bin/env python3
"""
BIST30 technical-indicator dashboard -- Streamlit + Plotly web app.

Run:
    streamlit run bist30_webapp.py

Opens a browser tab at http://localhost:8501 with a TradingView-style
candlestick chart for the BIST30 or BIST100 index, or any of their
constituents. Indicators work like TradingView's "+ Indicator" flow: add
as many as you want (multiple SMAs, EMAs, DMI, ...), each with its own
settings, and remove them individually. Zoom/pan/hover run client-side in
the browser (Plotly), so it stays smooth regardless of how many are on
the chart.
"""
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

BIST30_INDEX = "XU030.IS"
BIST100_INDEX = "XU100.IS"

# BIST30 constituents (Yahoo Finance tickers, ".IS" suffix), as of 2026-08.
# Note: Koza Group's tickers were renamed -- KOZAA -> TRMET, KOZAL -> TRALT.
# Index composition changes periodically -- verify against Borsa Istanbul
# if you need the exact current list.
BIST30_TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
    "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS",
    "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "TRMET.IS", "TRALT.IS",
    "KRDMD.IS", "MGROS.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS",
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "VAKBN.IS",
    "YKBNK.IS",
]

# BIST100 constituents (superset of BIST30), as of 2026-08. Includes several
# recent/small-cap additions (e.g. ALTNY, BALSU, CANTE, ODINE, PASEU, REEDR)
# whose tickers were cross-checked but are less liquid/well-documented --
# verify against Borsa Istanbul if one of those doesn't resolve.
BIST100_TICKERS = sorted(set(BIST30_TICKERS) | {
    "AEFES.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALTNY.IS", "ANSGR.IS",
    "ASTOR.IS", "BALSU.IS", "BERA.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS",
    "BTCIM.IS", "CANTE.IS", "CCOLA.IS", "CIMSA.IS", "CVKMD.IS", "CWENE.IS",
    "DAPGM.IS", "DOAS.IS", "DOHOL.IS", "DSTKF.IS", "ECILC.IS", "EFOR.IS",
    "ENERY.IS", "ENJSA.IS", "ESEN.IS", "EUPWR.IS", "EUREN.IS", "FENER.IS",
    "GENIL.IS", "GESAN.IS", "GLRMK.IS", "GRSEL.IS", "GRTHO.IS", "GSRAY.IS",
    "HEKTS.IS", "IEYHO.IS", "ISMEN.IS", "IZENR.IS", "KLRHO.IS", "KTLEV.IS",
    "KUYAS.IS", "MAGEN.IS", "MAVI.IS", "MIATK.IS", "MPARK.IS", "OBAMS.IS",
    "ODINE.IS", "OTKAR.IS", "PAHOL.IS", "PASEU.IS", "PATEK.IS", "PSGYO.IS",
    "QUAGR.IS", "RALYH.IS", "REEDR.IS", "SARKY.IS", "SKBNK.IS", "SOKM.IS",
    "TKFEN.IS", "TRENJ.IS", "TSKB.IS", "TTKOM.IS", "TUKAS.IS", "TURSG.IS",
    "ULKER.IS", "VESTL.IS", "ZOREN.IS",
})

UNIVERSES = {
    "BIST30": (BIST30_INDEX, BIST30_TICKERS),
    "BIST100": (BIST100_INDEX, BIST100_TICKERS),
}

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
INTERVALS = ["1d", "1wk"]

# TradingView-style dark palette
BG = "#131722"
PANEL_BG = "#131722"
GRID = "#2a2e39"
TEXT = "#d1d4dc"
MUTED = "#787b86"
UP = "#26a69a"
DOWN = "#ef5350"
ADX_COLOR = "#f7a600"
RSI_COLOR = "#b388ff"
MACD_COLOR = "#2962ff"
SIGNAL_COLOR = "#ff6d00"

OVERLAY_PALETTE = ["#f7a600", "#2962ff", "#26a69a", "#e91e63", "#4caf50", "#ff6d00", "#b388ff", "#00bcd4"]

# Indicator types: which panel they belong to and their default params.
OVERLAY_TYPES = {"SMA", "EMA", "Bollinger Bands"}
PANEL_OF_TYPE = {"RSI": "rsi", "MACD": "macd", "DMI (DI+/-, ADX)": "dmi", "Volume": "volume"}
INDICATOR_TYPES = ["SMA", "EMA", "Bollinger Bands", "RSI", "MACD", "DMI (DI+/-, ADX)", "Volume"]

DEFAULT_INDICATORS = [
    {"type": "SMA", "params": {"window": 20}},
    {"type": "EMA", "params": {"window": 12}},
    {"type": "RSI", "params": {"window": 14}},
    {"type": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
    {"type": "DMI (DI+/-, ADX)", "params": {"window": 14}},
    {"type": "Volume", "params": {}},
]


# --------------------------------------------------------------------------
# Indicator math
# --------------------------------------------------------------------------

def compute_sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def compute_ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid, mid + num_std * std, mid - num_std * std


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_dmi(df: pd.DataFrame, window: int = 14):
    """Wilder's Directional Movement Index: +DI, -DI, and ADX."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close, prev_high, prev_low = close.shift(1), high.shift(1), low.shift(1)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def padded_range(*series, pad_frac: float = 0.12, floor_at_zero: bool = False, skip: int = 0):
    """Y-axis range that hugs the actual data with breathing room, instead of Plotly's default fit.

    `skip` drops the leading N bars from the range calculation only (not from the plotted
    line) -- EWM-based indicators (RSI/MACD/ADX) have a noisy warm-up period right at the
    start of the series that would otherwise blow out the whole panel's scale.
    """
    values = np.concatenate([np.asarray(s, dtype=float)[skip:] for s in series])
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi > lo else (abs(hi) if hi else 1.0)
    pad = span * pad_frac
    lo, hi = lo - pad, hi + pad
    if floor_at_zero:
        lo = max(lo, 0.0)
    return [lo, hi]


def indicator_label(ind: dict) -> str:
    t, p = ind["type"], ind["params"]
    if t in ("SMA", "EMA"):
        return f"{t} {p['window']}"
    if t == "Bollinger Bands":
        return f"BB {p['window']} ({p['num_std']}σ)"
    if t == "RSI":
        return f"RSI {p['window']}"
    if t == "MACD":
        return f"MACD {p['fast']},{p['slow']},{p['signal']}"
    if t == "DMI (DI+/-, ADX)":
        return f"DMI {p['window']}"
    return t


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------

def build_figure(df: pd.DataFrame, ticker: str, indicators: list) -> go.Figure:
    close = df["Close"]
    x = df.index

    overlays = [i for i in indicators if i["type"] in OVERLAY_TYPES]
    rsi_list = [i for i in indicators if i["type"] == "RSI"]
    macd_list = [i for i in indicators if i["type"] == "MACD"]
    dmi_list = [i for i in indicators if i["type"] == "DMI (DI+/-, ADX)"]
    volume_list = [i for i in indicators if i["type"] == "Volume"]

    rows = ["price"]
    rows += ["rsi"] if rsi_list else []
    rows += ["macd"] if macd_list else []
    rows += ["dmi"] if dmi_list else []
    rows += ["volume"] if volume_list else []

    row_px = {"price": 480, "rsi": 220, "macd": 220, "dmi": 220, "volume": 190}
    heights = [row_px[r] for r in rows]
    fig_height = sum(heights) + 60
    row_of = {name: i + 1 for i, name in enumerate(rows)}

    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                         vertical_spacing=0.035, row_heights=heights)

    price_row = row_of["price"]
    fig.add_trace(
        go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=close,
            increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
            name=ticker, showlegend=False,
        ),
        row=price_row, col=1,
    )

    color_cycle = itertools.cycle(OVERLAY_PALETTE)
    for ind in overlays:
        color = next(color_cycle)
        p = ind["params"]
        if ind["type"] == "SMA" and len(df) > p["window"]:
            fig.add_trace(go.Scatter(x=x, y=compute_sma(close, p["window"]), mode="lines",
                                      name=indicator_label(ind), line=dict(color=color, width=1.3)),
                          row=price_row, col=1)
        elif ind["type"] == "EMA" and len(df) > p["window"]:
            fig.add_trace(go.Scatter(x=x, y=compute_ema(close, p["window"]), mode="lines",
                                      name=indicator_label(ind), line=dict(color=color, width=1.3, dash="dash")),
                          row=price_row, col=1)
        elif ind["type"] == "Bollinger Bands":
            mid, upper, lower = compute_bollinger(close, p["window"], p["num_std"])
            fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(color=color, width=0.8),
                                      showlegend=False), row=price_row, col=1)
            fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(color=color, width=0.8),
                                      fill="tonexty", fillcolor="rgba(120,123,134,0.08)",
                                      showlegend=False), row=price_row, col=1)
            fig.add_trace(go.Scatter(x=x, y=mid, mode="lines", name=indicator_label(ind),
                                      line=dict(color=color, width=1, dash="dot")), row=price_row, col=1)

    last_close = float(close.iloc[-1])
    fig.add_hline(y=last_close, line_dash="dash", line_color=MUTED, line_width=0.8, row=price_row, col=1)

    if rsi_list:
        r = row_of["rsi"]
        color_cycle = itertools.cycle(OVERLAY_PALETTE)
        rsi_series = []
        for ind in rsi_list:
            rsi = compute_rsi(close, ind["params"]["window"])
            rsi_series.append(rsi)
            fig.add_trace(go.Scatter(x=x, y=rsi, mode="lines", name=indicator_label(ind),
                                      line=dict(color=next(color_cycle) if len(rsi_list) > 1 else RSI_COLOR, width=1.5)),
                          row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=DOWN, line_width=0.8, opacity=0.6, row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=UP, line_width=0.8, opacity=0.6, row=r, col=1)
        rsi_skip = min(max(ind["params"]["window"] for ind in rsi_list) * 2, max(len(df) - 1, 0))
        rsi_range = padded_range(*rsi_series, pad_frac=0.15, skip=rsi_skip) or [0, 100]
        rsi_range = [min(rsi_range[0], -3), max(rsi_range[1], 103)]
        fig.update_yaxes(range=rsi_range, row=r, col=1, title_text="RSI", dtick=20)

    if macd_list:
        r = row_of["macd"]
        show_hist = len(macd_list) == 1
        color_cycle = itertools.cycle(OVERLAY_PALETTE)
        macd_series = []
        for ind in macd_list:
            p = ind["params"]
            macd_line, signal_line, hist = compute_macd(close, p["fast"], p["slow"], p["signal"])
            if show_hist:
                macd_series += [macd_line, signal_line, hist]
                hist_colors = np.where(hist >= 0, UP, DOWN)
                fig.add_trace(go.Bar(x=x, y=hist, name="Histogram", marker_color=hist_colors,
                                      opacity=0.5, showlegend=False), row=r, col=1)
                fig.add_trace(go.Scatter(x=x, y=macd_line, mode="lines", name="MACD",
                                          line=dict(color=MACD_COLOR, width=1.4)), row=r, col=1)
                fig.add_trace(go.Scatter(x=x, y=signal_line, mode="lines", name="Signal",
                                          line=dict(color=SIGNAL_COLOR, width=1.4)), row=r, col=1)
            else:
                macd_series += [macd_line, signal_line]
                c = next(color_cycle)
                fig.add_trace(go.Scatter(x=x, y=macd_line, mode="lines", name=f"{indicator_label(ind)} MACD",
                                          line=dict(color=c, width=1.4)), row=r, col=1)
                fig.add_trace(go.Scatter(x=x, y=signal_line, mode="lines", name=f"{indicator_label(ind)} Signal",
                                          line=dict(color=c, width=1.4, dash="dot")), row=r, col=1)
        fig.add_hline(y=0, line_color=MUTED, line_width=0.6, row=r, col=1)
        macd_skip = min(max(ind["params"]["slow"] for ind in macd_list) * 2, max(len(df) - 1, 0))
        fig.update_yaxes(range=padded_range(*macd_series, skip=macd_skip), row=r, col=1, title_text="MACD")

    if dmi_list:
        r = row_of["dmi"]
        dmi_series = []
        for ind in dmi_list:
            plus_di, minus_di, adx = compute_dmi(df, ind["params"]["window"])
            dmi_series += [plus_di, minus_di, adx]
            suffix = f" ({ind['params']['window']})" if len(dmi_list) > 1 else ""
            fig.add_trace(go.Scatter(x=x, y=plus_di, mode="lines", name=f"+DI{suffix}",
                                      line=dict(color=UP, width=1.4)), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=minus_di, mode="lines", name=f"-DI{suffix}",
                                      line=dict(color=DOWN, width=1.4)), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=adx, mode="lines", name=f"ADX{suffix}",
                                      line=dict(color=ADX_COLOR, width=1.6)), row=r, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color=MUTED, line_width=0.7, opacity=0.6, row=r, col=1)
        dmi_skip = min(max(ind["params"]["window"] for ind in dmi_list) * 2, max(len(df) - 1, 0))
        fig.update_yaxes(range=padded_range(*dmi_series, floor_at_zero=True, skip=dmi_skip), row=r, col=1, title_text="DMI")

    if volume_list:
        r = row_of["volume"]
        up_mask = close.diff() >= 0
        if len(up_mask):
            up_mask.iloc[0] = True
        vol_colors = np.where(up_mask, UP, DOWN)
        fig.add_trace(go.Bar(x=x, y=df["Volume"], name="Volume", marker_color=vol_colors,
                              opacity=0.7, showlegend=False), row=r, col=1)
        fig.update_yaxes(range=[0, float(df["Volume"].max()) * 1.15], row=r, col=1, title_text="Volume")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL_BG,
        font_color=TEXT,
        margin=dict(l=60, r=60, t=10, b=10),
        height=fig_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_xaxes(rangeslider_visible=False, gridcolor=GRID, showspikes=True,
                      spikemode="across", spikecolor=MUTED, spikethickness=1)
    fig.update_yaxes(gridcolor=GRID, tickfont=dict(size=11), title_font=dict(size=12),
                      title_standoff=8, automargin=True)
    fig.update_yaxes(title_text="Price", row=price_row, col=1)
    fig.update_xaxes(showticklabels=True, tickfont=dict(size=11), row=len(rows), col=1)

    return fig


# --------------------------------------------------------------------------
# Add-indicator sidebar widget
# --------------------------------------------------------------------------

def render_indicator_params(ind_type: str) -> dict:
    if ind_type in ("SMA", "EMA"):
        return {"window": st.number_input("Window", 2, 500, 20, key="new_window")}
    if ind_type == "Bollinger Bands":
        window = st.number_input("Window", 2, 500, 20, key="new_bb_window")
        num_std = st.number_input("Std Dev", 0.5, 5.0, 2.0, step=0.5, key="new_bb_std")
        return {"window": window, "num_std": num_std}
    if ind_type == "RSI":
        return {"window": st.number_input("Window", 2, 200, 14, key="new_rsi_window")}
    if ind_type == "MACD":
        fast = st.number_input("Fast", 1, 200, 12, key="new_macd_fast")
        slow = st.number_input("Slow", 1, 200, 26, key="new_macd_slow")
        signal = st.number_input("Signal", 1, 100, 9, key="new_macd_signal")
        return {"fast": fast, "slow": slow, "signal": signal}
    if ind_type == "DMI (DI+/-, ADX)":
        return {"window": st.number_input("Window", 2, 200, 14, key="new_dmi_window")}
    return {}


def indicator_sidebar():
    st.markdown("**Add indicator**")
    ind_type = st.selectbox("Type", INDICATOR_TYPES, key="new_ind_type", label_visibility="collapsed")
    params = render_indicator_params(ind_type)
    if st.button("+ Add indicator", use_container_width=True):
        st.session_state.next_id += 1
        st.session_state.indicators.append({"id": st.session_state.next_id, "type": ind_type, "params": params})
        st.rerun()

    st.markdown("**Active indicators**")
    if not st.session_state.indicators:
        st.caption("None added yet.")
    for ind in list(st.session_state.indicators):
        c1, c2 = st.columns([5, 1])
        c1.write(indicator_label(ind))
        if c2.button("✕", key=f"remove_{ind['id']}"):
            st.session_state.indicators = [i for i in st.session_state.indicators if i["id"] != ind["id"]]
            st.rerun()


# --------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="BIST Dashboard", layout="wide", page_icon="📈")

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; }}
        section[data-testid="stSidebar"] {{ background-color: #1e222d; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "indicators" not in st.session_state:
        st.session_state.indicators = [dict(ind, id=i) for i, ind in enumerate(DEFAULT_INDICATORS)]
        st.session_state.next_id = len(DEFAULT_INDICATORS) - 1

    with st.sidebar:
        st.title("BIST Dashboard")

        universe_name = st.radio("Universe", list(UNIVERSES.keys()), index=0, horizontal=True)
        index_ticker, universe_tickers = UNIVERSES[universe_name]
        symbol_options = [index_ticker] + universe_tickers

        def label(t: str) -> str:
            if t == BIST30_INDEX:
                return "BIST30 Index (XU030.IS)"
            if t == BIST100_INDEX:
                return "BIST100 Index (XU100.IS)"
            return t.replace(".IS", "")

        ticker = st.selectbox("Symbol", symbol_options, index=0, format_func=label)
        col_p, col_i = st.columns(2)
        period = col_p.selectbox("Period", PERIODS, index=2)
        interval = col_i.selectbox("Interval", INTERVALS, index=0)

        st.markdown("---")
        indicator_sidebar()

        st.markdown("---")
        if st.button("Refresh data"):
            st.cache_data.clear()

        st.caption(f"{universe_name} index + {len(universe_tickers)} constituents · data via Yahoo Finance")

    try:
        df = fetch_data(ticker, period, interval)
    except Exception as exc:
        st.error(f"Failed to load data for {ticker}: {exc}")
        return

    if df.empty:
        st.error(f"No data returned for {ticker}. It may be delisted or the symbol is wrong.")
        return

    close = df["Close"]
    last_close = float(close.iloc[-1])
    first_close = float(close.iloc[0])
    change_pct = (last_close / first_close - 1) * 100 if first_close else 0.0
    day_change = float(close.iloc[-1] - close.iloc[-2]) if len(close) > 1 else 0.0
    day_change_pct = (day_change / close.iloc[-2] * 100) if len(close) > 1 and close.iloc[-2] else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Symbol", label(ticker), f"{day_change:+.2f} ({day_change_pct:+.2f}%) today")
    c2.metric("Last Close", f"{last_close:,.2f} TRY")
    c3.metric(f"Change over {period}", f"{change_pct:+.2f}%")

    fig = build_figure(df, ticker, st.session_state.indicators)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


if __name__ == "__main__":
    main()
