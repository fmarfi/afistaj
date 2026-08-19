#!/usr/bin/env python3
"""
Interactive BIST30 technical-indicator dashboard (matplotlib), TradingView-style.

Launches a window with:
  - a candlestick chart for any BIST30 constituent (or the BIST30 index itself)
  - Prev/Next buttons + a text box to jump straight to a ticker
  - a period selector (1mo / 3mo / 6mo / 1y / 2y)
  - checkboxes to turn indicators on/off: SMA, EMA, Bollinger Bands, RSI, MACD, Volume

Usage
-----
python bist30_dashboard.py                          # opens the interactive dashboard
python bist30_dashboard.py --ticker THYAO.IS         # start on a specific stock
python bist30_dashboard.py --list-tickers            # print BIST30 constituent tickers

# Headless one-off snapshot (no window, just a PNG):
python bist30_dashboard.py --ticker ASELS.IS --save asels.png --no-show
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

BIST30_INDEX = "XU030.IS"

# BIST30 constituents (Yahoo Finance tickers, ".IS" suffix), as of 2026-08.
# Index composition changes periodically -- verify against Borsa Istanbul
# if you need the exact current list.
BIST30_TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
    "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS",
    "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KOZAA.IS", "KOZAL.IS",
    "KRDMD.IS", "MGROS.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS",
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "VAKBN.IS",
    "YKBNK.IS",
]

ALL_TICKERS = [BIST30_INDEX] + BIST30_TICKERS

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"]

INDICATOR_LABELS = ["SMA", "EMA", "Bollinger Bands", "RSI", "MACD", "Volume"]
LABEL_TO_KEY = {
    "SMA": "sma", "EMA": "ema", "Bollinger Bands": "bb",
    "RSI": "rsi", "MACD": "macd", "Volume": "volume",
}

# --------------------------------------------------------------------------
# TradingView-style dark theme
# --------------------------------------------------------------------------

BG = "#131722"
PANEL_BG = "#131722"
GRID = "#2a2e39"
TEXT = "#d1d4dc"
MUTED = "#787b86"
UP = "#26a69a"
DOWN = "#ef5350"
SMA_COLOR = "#f7a600"
EMA_COLOR = "#2962ff"
BB_COLOR = "#787b86"
RSI_COLOR = "#b388ff"
MACD_COLOR = "#2962ff"
SIGNAL_COLOR = "#ff6d00"


# --------------------------------------------------------------------------
# Indicators
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


def fetch_data(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def plot_candles(ax, df: pd.DataFrame, x: np.ndarray):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    colors = np.where(c >= o, UP, DOWN)
    ax.vlines(x, l, h, color=colors, linewidth=1, zorder=2)
    ax.bar(x, np.abs(c - o), bottom=np.minimum(o, c), width=0.6, color=colors, linewidth=0, zorder=3)


def style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def empty_panel(ax, text: str):
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)


# --------------------------------------------------------------------------
# Shared render logic (used by both interactive dashboard and static --save)
# --------------------------------------------------------------------------

def render_chart(axes, df: pd.DataFrame, ticker: str, period: str, state: dict):
    ax_price, ax_rsi, ax_macd, ax_vol = axes
    for ax in axes:
        ax.clear()
        style_axes(ax)

    close = df["Close"]
    x = np.arange(len(df))

    # --- Price panel: candles + overlays ---
    plot_candles(ax_price, df, x)

    if state.get("sma"):
        for w, lw in ((20, 1.1), (50, 1.1)):
            if len(df) > w:
                ax_price.plot(x, compute_sma(close, w), color=SMA_COLOR, linewidth=lw,
                               label=f"SMA {w}", alpha=0.9 if w == 20 else 0.6)
    if state.get("ema"):
        for w in (12, 26):
            if len(df) > w:
                ax_price.plot(x, compute_ema(close, w), color=EMA_COLOR, linewidth=1.1,
                               linestyle="--", label=f"EMA {w}", alpha=0.9 if w == 12 else 0.6)
    if state.get("bb"):
        mid, upper, lower = compute_bollinger(close, 20)
        ax_price.plot(x, mid, color=BB_COLOR, linewidth=0.9, linestyle=":", label="BB mid (20)")
        ax_price.plot(x, upper, color=BB_COLOR, linewidth=0.7, alpha=0.7)
        ax_price.plot(x, lower, color=BB_COLOR, linewidth=0.7, alpha=0.7)
        ax_price.fill_between(x, lower, upper, color=BB_COLOR, alpha=0.06)

    last_close = close.iloc[-1]
    first_close = close.iloc[0]
    change_pct = (last_close / first_close - 1) * 100 if first_close else 0
    line_color = UP if change_pct >= 0 else DOWN
    ax_price.axhline(last_close, color=line_color, linewidth=0.7, linestyle="--", alpha=0.6)
    ax_price.text(1.002, last_close, f"{last_close:,.2f}", transform=ax_price.get_yaxis_transform(),
                  color=line_color, fontsize=8, va="center", ha="left")

    if ax_price.get_legend_handles_labels()[0]:
        ax_price.legend(loc="upper left", fontsize=7.5, ncol=3, facecolor=PANEL_BG,
                         edgecolor=GRID, labelcolor=TEXT, framealpha=0.7)
    ax_price.set_ylabel("Price", color=TEXT, fontsize=9)
    ax_price.set_title(
        f"{ticker}   {last_close:,.2f} TRY   ({change_pct:+.2f}% over period)",
        color=line_color, fontsize=11, loc="left", fontweight="bold",
    )

    # --- RSI panel ---
    if state.get("rsi"):
        rsi = compute_rsi(close, 14)
        ax_rsi.plot(x, rsi, color=RSI_COLOR, linewidth=1.1)
        ax_rsi.axhline(70, color=DOWN, linewidth=0.7, linestyle="--", alpha=0.5)
        ax_rsi.axhline(30, color=UP, linewidth=0.7, linestyle="--", alpha=0.5)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", color=TEXT, fontsize=9)
    else:
        empty_panel(ax_rsi, "RSI off")

    # --- MACD panel ---
    if state.get("macd"):
        macd_line, signal_line, hist = compute_macd(close)
        colors = np.where(hist >= 0, UP, DOWN)
        ax_macd.bar(x, hist, color=colors, width=0.7, alpha=0.5)
        ax_macd.plot(x, macd_line, color=MACD_COLOR, linewidth=1.0, label="MACD")
        ax_macd.plot(x, signal_line, color=SIGNAL_COLOR, linewidth=1.0, label="Signal")
        ax_macd.axhline(0, color=MUTED, linewidth=0.6)
        ax_macd.legend(loc="upper left", fontsize=7.5, ncol=2, facecolor=PANEL_BG,
                        edgecolor=GRID, labelcolor=TEXT, framealpha=0.7)
        ax_macd.set_ylabel("MACD", color=TEXT, fontsize=9)
    else:
        empty_panel(ax_macd, "MACD off")

    # --- Volume panel ---
    if state.get("volume"):
        up = close.diff() >= 0
        up.iloc[0] = True
        colors = np.where(up, UP, DOWN)
        ax_vol.bar(x, df["Volume"], color=colors, width=0.7, alpha=0.6)
        ax_vol.set_ylabel("Volume", color=TEXT, fontsize=9)
    else:
        empty_panel(ax_vol, "Volume off")

    # --- Shared x-axis: trading-day index -> date labels (no weekend gaps) ---
    def fmt(val, pos, _df=df):
        i = int(round(val))
        if 0 <= i < len(_df):
            return _df.index[i].strftime("%d %b")
        return ""

    for ax in axes:
        ax.set_xlim(-1, len(df))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.tick_params(labelbottom=False)
    ax_vol.tick_params(labelbottom=True)
    for label in ax_vol.get_xticklabels():
        label.set_rotation(0)


# --------------------------------------------------------------------------
# Interactive dashboard
# --------------------------------------------------------------------------

class Dashboard:
    def __init__(self, initial_ticker: str = BIST30_INDEX, initial_period: str = "6mo"):
        self.ticker = initial_ticker if initial_ticker in ALL_TICKERS else ALL_TICKERS[0]
        self.period = initial_period if initial_period in PERIODS else "6mo"
        self.state = {"sma": True, "ema": True, "bb": True, "rsi": True, "macd": True, "volume": True}
        self.cache: dict = {}
        self.df = None

        self._build_figure()
        self._build_widgets()
        self.load_and_draw()

    # -- layout -----------------------------------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(17, 10), facecolor=BG)
        self.fig.canvas.manager.set_window_title("BIST30 Dashboard")

        left, width = 0.24, 0.73
        self.ax_price = self.fig.add_axes([left, 0.55, width, 0.37], facecolor=PANEL_BG)
        self.ax_rsi = self.fig.add_axes([left, 0.40, width, 0.13], facecolor=PANEL_BG)
        self.ax_macd = self.fig.add_axes([left, 0.24, width, 0.14], facecolor=PANEL_BG)
        self.ax_vol = self.fig.add_axes([left, 0.08, width, 0.14], facecolor=PANEL_BG)
        for ax in (self.ax_rsi, self.ax_macd, self.ax_vol):
            ax.sharex(self.ax_price)

        self.status_ax = self.fig.add_axes([0.02, 0.02, 0.19, 0.04], facecolor=BG)
        self.status_ax.axis("off")
        self.status_text = self.status_ax.text(0, 0.5, "", color=MUTED, fontsize=8, va="center")

    def _build_widgets(self):
        def styled(ax):
            ax.set_facecolor(PANEL_BG)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            return ax

        # Ticker text box
        ax_box = styled(self.fig.add_axes([0.02, 0.90, 0.19, 0.045]))
        self.ticker_box = TextBox(ax_box, "", initial=self.ticker, color=PANEL_BG, hovercolor="#1e222d")
        self.ticker_box.label.set_color(TEXT)
        self.ticker_box.text_disp.set_color(TEXT)
        self.ticker_box.on_submit(self.on_ticker_submit)

        # Prev / Next buttons
        ax_prev = styled(self.fig.add_axes([0.02, 0.855, 0.09, 0.035]))
        ax_next = styled(self.fig.add_axes([0.12, 0.855, 0.09, 0.035]))
        self.btn_prev = Button(ax_prev, "< Prev", color=PANEL_BG, hovercolor="#1e222d")
        self.btn_next = Button(ax_next, "Next >", color=PANEL_BG, hovercolor="#1e222d")
        for b in (self.btn_prev, self.btn_next):
            b.label.set_color(TEXT)
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)

        # Period radio buttons
        self.fig.text(0.02, 0.815, "Period", color=MUTED, fontsize=9)
        ax_period = styled(self.fig.add_axes([0.02, 0.66, 0.19, 0.145]))
        ax_period.axis("off")
        self.period_radio = RadioButtons(ax_period, PERIODS, active=PERIODS.index(self.period),
                                          activecolor=EMA_COLOR)
        for label in self.period_radio.labels:
            label.set_color(TEXT)
            label.set_fontsize(9)
        self.period_radio.on_clicked(self.on_period)

        # Indicator checkboxes
        self.fig.text(0.02, 0.635, "Indicators", color=MUTED, fontsize=9)
        ax_ind = styled(self.fig.add_axes([0.02, 0.42, 0.19, 0.2]))
        ax_ind.axis("off")
        initial_states = [self.state[LABEL_TO_KEY[lbl]] for lbl in INDICATOR_LABELS]
        self.check = CheckButtons(ax_ind, INDICATOR_LABELS, initial_states)
        for label in self.check.labels:
            label.set_color(TEXT)
            label.set_fontsize(9)
        self.check.on_clicked(self.on_indicator_toggle)

        self.fig.text(0.02, 0.395, "BIST30 index + 30 constituents", color=MUTED, fontsize=7.5)

    # -- callbacks ----------------------------------------------------------
    def on_ticker_submit(self, text):
        text = text.strip().upper()
        if text and not text.endswith(".IS"):
            text += ".IS"
        if text:
            self.ticker = text
            self.load_and_draw()

    def on_prev(self, event):
        idx = ALL_TICKERS.index(self.ticker) if self.ticker in ALL_TICKERS else 0
        self.ticker = ALL_TICKERS[(idx - 1) % len(ALL_TICKERS)]
        self._sync_ticker_box()
        self.load_and_draw()

    def on_next(self, event):
        idx = ALL_TICKERS.index(self.ticker) if self.ticker in ALL_TICKERS else 0
        self.ticker = ALL_TICKERS[(idx + 1) % len(ALL_TICKERS)]
        self._sync_ticker_box()
        self.load_and_draw()

    def _sync_ticker_box(self):
        self.ticker_box.set_val(self.ticker)

    def on_period(self, label):
        self.period = label
        self.load_and_draw()

    def on_indicator_toggle(self, label):
        key = LABEL_TO_KEY[label]
        self.state[key] = not self.state[key]
        self.redraw()

    # -- data / drawing ------------------------------------------------------
    def load_and_draw(self):
        key = (self.ticker, self.period)
        if key in self.cache:
            self.df = self.cache[key]
            self.status_text.set_text("")
            self.redraw()
            return

        self.status_text.set_text(f"Loading {self.ticker}...")
        self.fig.canvas.draw_idle()
        try:
            plt.pause(0.001)
        except Exception:
            pass
        try:
            df = fetch_data(self.ticker, self.period)
        except Exception as exc:
            self.status_text.set_text(f"Error: {exc}")
            self.fig.canvas.draw_idle()
            return
        self.cache[key] = df
        self.df = df
        self.status_text.set_text("")
        self.redraw()

    def redraw(self):
        if self.df is None or self.df.empty:
            return
        axes = (self.ax_price, self.ax_rsi, self.ax_macd, self.ax_vol)
        render_chart(axes, self.df, self.ticker, self.period, self.state)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive BIST30 technical-indicator dashboard.")
    parser.add_argument("--ticker", default=BIST30_INDEX,
                         help=f"Starting ticker, e.g. THYAO.IS. Default: {BIST30_INDEX} (BIST30 index).")
    parser.add_argument("--period", default="6mo", choices=PERIODS, help="Starting lookback period.")
    parser.add_argument("--list-tickers", action="store_true", help="Print BIST30 constituent tickers and exit.")
    parser.add_argument("--save", default=None, help="Render one static snapshot to this path instead of an interactive window.")
    parser.add_argument("--no-show", action="store_true", help="With --save, skip opening the interactive window.")
    args = parser.parse_args()

    if args.list_tickers:
        print(f"BIST30 index: {BIST30_INDEX}")
        print("Constituents:")
        for t in BIST30_TICKERS:
            print(f"  {t}")
        return

    if args.save:
        if args.no_show:
            matplotlib.use("Agg")
        df = fetch_data(args.ticker, args.period)
        fig = plt.figure(figsize=(17, 10), facecolor=BG)
        left, width = 0.06, 0.91
        ax_price = fig.add_axes([left, 0.55, width, 0.37], facecolor=PANEL_BG)
        ax_rsi = fig.add_axes([left, 0.40, width, 0.13], facecolor=PANEL_BG)
        ax_macd = fig.add_axes([left, 0.24, width, 0.14], facecolor=PANEL_BG)
        ax_vol = fig.add_axes([left, 0.08, width, 0.14], facecolor=PANEL_BG)
        for ax in (ax_rsi, ax_macd, ax_vol):
            ax.sharex(ax_price)
        state = {"sma": True, "ema": True, "bb": True, "rsi": True, "macd": True, "volume": True}
        render_chart((ax_price, ax_rsi, ax_macd, ax_vol), df, args.ticker, args.period, state)
        fig.savefig(args.save, dpi=150, facecolor=BG)
        print(f"Saved snapshot to {args.save}")
        if not args.no_show:
            plt.show()
        return

    dash = Dashboard(initial_ticker=args.ticker, initial_period=args.period)
    dash.show()


if __name__ == "__main__":
    main()
