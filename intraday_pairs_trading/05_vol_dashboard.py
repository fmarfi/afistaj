"""
Interactive parameter lab for the VOLATILITY MEAN-REVERSION strategy
(_engine_vol.py) -- pairs trading and single-asset PRICE mean reversion
were both tried and dropped (0/16 BIST stocks show price-level mean
reversion at any tested resolution). Their REALIZED VOLATILITY does,
overwhelmingly (16/16, see _engine_vol.py's docstring). This dashboard
lets you screen tickers by that signal, trade ONE ticker or a BASKET of
the top-N qualifying ones simultaneously (diversifying away from the
"got lucky with one ticker" overfitting problem confirmed on real data),
adjust every threshold live, and see the resulting trade log with real
quoted entry/exit prices and Wilder's +DI/-DI at entry.

Run: python intraday_pairs_trading/05_vol_dashboard.py
Then open http://127.0.0.1:5055 in a browser.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request

import _engine as eng
import _engine_vol as vol

app = Flask(__name__)
_ohlc_cache = {}

BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8a86"

FIELDS = [
    ("in_sample_fraction", "In-sample fraction", 0.0, 1.0, 0.05),
    ("adf_crit", "ADF critical value", -6.0, -1.0, 0.05),
    ("hurst_cutoff", "Hurst cutoff", 0.1, 0.6, 0.01),
    ("vol_window", "Realized-vol window (bars)", 4, 60, 1),
    ("session_lookback", "Session lookback (sessions)", 2, 15, 1),
    ("z_window", "Vol z-score window (bars)", 10, 150, 5),
    ("di_period", "+DI/-DI smoothing period (bars)", 2, 40, 1),
    ("momentum_bars", "Momentum fallback (bars, used only if DI unavailable)", 1, 20, 1),
    ("entry_z", "Entry vol z-score", 0.5, 6.0, 0.1),
    ("exit_z", "Exit vol z-score (normalized)", -2.0, 2.0, 0.1),
    ("stop_z", "Stop vol z-score (still expanding)", 1.0, 15.0, 0.1),
]
INT_FIELDS = {"session_lookback", "z_window", "vol_window", "momentum_bars", "di_period"}

INTERVAL_OPTIONS = [
    ("5m", "5 minutes (max 60d)"),
    ("15m", "15 minutes (max 60d)"),
    ("30m", "30 minutes (max 60d)"),
    ("60m", "60 minutes (max 730d, ~2y)"),
]
PERIOD_OPTIONS_BY_INTERVAL = {
    "5m": ["10d", "30d", "60d"],
    "15m": ["10d", "30d", "60d"],
    "30m": ["10d", "30d", "60d"],
    "60m": ["30d", "90d", "180d", "365d", "730d"],
}

CAPITAL_FIELDS = [("starting_capital", "Starting capital (TRY)", 1000, 10_000_000, 1000)]
BASKET_FIELDS = [("n_tickers", "Basket size (1 = single ticker, >1 = trade top-N simultaneously)", 1, 16, 1)]

DEFAULT_BACKTEST_PARAMS = {
    "in_sample_fraction": 0.6, "adf_crit": -2.86, "hurst_cutoff": 0.45,
    "vol_window": vol.VOL_WINDOW, "session_lookback": 5, "z_window": 60,
    "di_period": vol.DI_PERIOD, "momentum_bars": vol.MOMENTUM_BARS,
    "entry_z": 1.5, "exit_z": vol.EXIT_RV_Z, "stop_z": 3.5,
}
DEFAULT_CAPITAL_PARAMS = {"starting_capital": 100_000.0}
DEFAULT_BASKET_PARAMS = {"n_tickers": 1}
DEFAULT_DATA_PARAMS = {"interval": eng.BIST_INTERVAL, "period": eng.BIST_PERIOD}


def get_ohlc(interval, period):
    cache_key = (interval, period)
    if _ohlc_cache.get("key") != cache_key:
        _ohlc_cache["ohlc"] = vol.download_intraday_ohlc(eng.BIST_UNIVERSE, interval=interval, period=period)
        _ohlc_cache["key"] = cache_key
    return _ohlc_cache["ohlc"]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_equity_chart(equity_curve, split_idx, starting_capital, trades, timestamps, title_suffix=""):
    """Shared by single-ticker and basket modes -- trade markers on the equity curve either way."""
    equity_oos = equity_curve[split_idx:]
    ts_oos = timestamps[split_idx:]
    ts_to_pos = {ts: i for i, ts in enumerate(ts_oos)}

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(equity_oos, color=BLUE, linewidth=1.3, zorder=2)
    ax.axhline(starting_capital, color=GRAY, linewidth=0.8, linestyle="--")
    for t in trades:
        entry_pos = ts_to_pos.get(t["entry_time"])
        exit_pos = ts_to_pos.get(t["exit_time"])
        if entry_pos is None or exit_pos is None:
            continue
        win = t["pnl_currency"] > 0
        ax.scatter(entry_pos, equity_oos[entry_pos], marker="^", s=24,
                    facecolors="none", edgecolors=GRAY, linewidths=1.0, zorder=3)
        ax.scatter(exit_pos, equity_oos[exit_pos], marker="o", s=22,
                    color=AQUA if win else RED, zorder=3)
    ax.set_title(f"Equity curve, starting capital {starting_capital:,.0f} TRY (no trading costs){title_suffix}  "
                 f"– △ entry, ● exit")
    return fig_to_base64(fig)


def make_single_ticker_charts(result, entry_z, exit_z, stop_z):
    results = result["results"]
    oos = results[results["rv_z"].notna()].reset_index(drop=True)
    split_idx = result["split_idx"]

    pnl_chart = make_equity_chart(result["equity_curve"], split_idx, result["starting_capital"],
                                   result["trades"], results["timestamp"].to_numpy())

    fig2, ax2 = plt.subplots(figsize=(8, 2.6))
    x = np.arange(len(oos))
    ax2.plot(x, oos["rv_z"].to_numpy(), color=BLUE, linewidth=0.7)
    ax2.axhline(entry_z, color=GRAY, linestyle="--", linewidth=1, label="entry")
    ax2.axhline(exit_z, color=AQUA, linestyle="--", linewidth=1, label="exit")
    ax2.axhline(stop_z, color=RED, linestyle=":", linewidth=1, label="stop")
    ax2.fill_between(x, ax2.get_ylim()[0], ax2.get_ylim()[1],
                      where=~oos["qualified"].to_numpy(dtype=bool), color=RED, alpha=0.08)
    ax2.set_title("Realized-volatility z-score (shaded = disqualified)")
    ax2.legend(loc="upper right", fontsize=7)

    fig3, ax3 = plt.subplots(figsize=(8, 2.2))
    ax3.plot(x, oos["rv"].to_numpy(), color=RED, linewidth=0.9)
    ax3.set_title("Realized volatility (raw, the signal being traded)")

    fig4, ax4 = plt.subplots(figsize=(8, 2.4))
    ax4.plot(x, oos["plus_di"].to_numpy(), color=AQUA, linewidth=0.8, label="+DI (up-move dominant)")
    ax4.plot(x, oos["minus_di"].to_numpy(), color=RED, linewidth=0.8, label="-DI (down-move dominant)")
    ax4.set_title("Wilder's +DI / -DI (decides trade direction on a vol spike)")
    ax4.legend(loc="upper right", fontsize=7)

    return pnl_chart, fig_to_base64(fig2), fig_to_base64(fig3), fig_to_base64(fig4)


def build_trade_cards(trades, ticker_lookup=None):
    """
    Single-instrument trade cards, one price row per card. ticker_lookup:
    pass a dict-like (or None to read t['ticker']) so the SAME renderer
    works for single-ticker mode (one ticker, passed once) and basket
    mode (each trade already carries its own 'ticker' field).
    """
    has_currency = bool(trades) and "pnl_currency" in trades[0]
    show_ticker = bool(trades) and "ticker" in trades[0]
    cols = "1.1fr 1.5fr 0.7fr 0.6fr 0.6fr 0.6fr 0.6fr 1fr 1fr" if show_ticker else "1.7fr 0.7fr 0.6fr 0.6fr 0.6fr 1fr 1fr"

    header_spans = (
        "<span>Ticker</span><span>Entry &rarr; Exit</span><span>Dir.</span><span>+DI</span><span>-DI</span>"
        "<span>Hold</span><span>Reason</span><span>PnL</span>"
        if show_ticker else
        "<span>Entry &rarr; Exit</span><span>Dir.</span><span>+DI</span><span>-DI</span>"
        "<span>Hold</span><span>Reason</span><span>PnL</span>"
    )
    header = f'<div class="trade-cards-header" style="grid-template-columns: {cols};">{header_spans}</div>'

    cards = []
    for t in trades:
        pnl_value, pnl_text = (t["pnl_currency"], f"{t['pnl_currency']:+,.0f} TRY") if has_currency else (t["pnl"], f"{t['pnl']:+.5f} (raw units)")
        pnl_class = "pos" if pnl_value > 0 else "neg"
        change_pct = (t["exit_price"] / t["entry_price"] - 1) * 100
        di_text = (f"{t['entry_plus_di']:.0f}", f"{t['entry_minus_di']:.0f}") if not np.isnan(t.get("entry_plus_di", np.nan)) else ("–", "–")
        ticker_span = f"<span>{t['ticker']}</span>" if show_ticker else ""
        sizing_note = (
            f"""<p class="hint">Base unit: <strong>{t['dollar_per_unit']:,.0f} TRY</strong> out of
            {t['starting_capital']:,.0f} TRY allocated to this {'ticker' if show_ticker else 'strategy'} &mdash;
            the strategy's own GARCH volatility scale then adjusts actual exposure automatically, capped at that budget.</p>"""
            if has_currency else ""
        )
        ticker_label = t.get("ticker", "")
        cards.append(f"""<details class="trade-card">
  <summary style="grid-template-columns: {cols};">
    {ticker_span}
    <span>{t['entry_time']} &rarr; {t['exit_time']}</span>
    <span>{t['direction']}</span>
    <span>{di_text[0]}</span>
    <span>{di_text[1]}</span>
    <span>{t['hold_bars']}</span>
    <span>{t['exit_reason']}</span>
    <span class="{pnl_class}">{pnl_text}</span>
  </summary>
  <div class="trade-detail">
    <table>
      <tr><th></th><th>Entry price</th><th>Exit price</th><th>Change</th></tr>
      <tr><td>{ticker_label}</td><td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td>
          <td class="{'pos' if change_pct >= 0 else 'neg'}">{change_pct:+.2f}%</td></tr>
    </table>
    <p class="hint">Entry vol z-score {t['entry_rv_z']:.2f} &rarr; exit {t['exit_rv_z']:.2f}.
    Direction was set by Wilder's +DI/-DI at entry ({di_text[0]} vs {di_text[1]}): -DI dominant means
    downward movement had led coming in (contrarian long, betting on a bounce); +DI dominant means the
    opposite (contrarian short).</p>
    {sizing_note}
  </div>
</details>""")
    return header + f'<div class="trade-cards">{"".join(cards)}</div>'


PAGE_CSS = f"""
  :root {{ --surface:#fcfcfb; --surface-2:#f3f2ef; --border:#e4e2dd; --text:#0b0b0b; --text-2:#52514e; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface:#1a1a19; --surface-2:#232320; --border:#34332f; --text:#ffffff; --text-2:#c3c2b7; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--surface); color: var(--text); font-family: -apple-system, sans-serif; margin: 0; padding: 28px; }}
  h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-2); font-size: 0.85rem; margin-bottom: 20px; }}
  .hint {{ color: var(--text-2); font-size: 0.78rem; margin: 4px 0 0; }}
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
  .tile-value {{ font-size: 1.15rem; font-weight: 600; overflow-wrap: break-word; }}
  h2 {{ font-size: 0.95rem; margin: 24px 0 10px; border-top: 1px solid var(--border); padding-top: 20px; }}
  img {{ max-width: 100%; border-radius: 6px; border: 1px solid var(--border); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
  th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-2); font-weight: 500; }}
  .neg {{ color: {RED}; }}
  .pos {{ color: {AQUA}; }}
  .banner {{ background: color-mix(in srgb, {RED} 12%, var(--surface)); border: 1px solid {RED}; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.85rem; }}
  .error {{ background: color-mix(in srgb, {RED} 15%, var(--surface)); border: 1px solid {RED}; border-radius: 8px; padding: 12px 16px; }}
  {eng.TRADE_CARD_CSS}
"""


def render_page(params, result=None, basket_result=None, error=None, screening=None):
    field_html = "".join(
        f"""<label>{label}<input type="number" name="{key}" value="{params[key]}" min="{lo}" max="{hi}" step="any"></label>"""
        for key, label, lo, hi, step in FIELDS
    )

    interval = params["interval"]
    period_options = "".join(
        f'<option value="{p}" {"selected" if p == params["period"] else ""}>{p}</option>'
        for p in PERIOD_OPTIONS_BY_INTERVAL[interval]
    )
    interval_options = "".join(
        f'<option value="{val}" {"selected" if val == interval else ""}>{label}</option>'
        for val, label in INTERVAL_OPTIONS
    )
    data_field_html = f"""
        <label>Bar interval ("how fine")
          <select name="interval" onchange="this.form.period.value=''; this.form.submit()">{interval_options}</select>
        </label>
        <label>History length ("how long")
          <select name="period">{period_options}</select>
        </label>"""

    is_basket = params["n_tickers"] > 1
    ticker_field_html = ""
    if screening is not None:
        selected_ticker = params.get("ticker", "auto")
        qualifying = screening[screening["qualifies"]]
        other = screening[~screening["qualifies"]]

        def option(row):
            sel = "selected" if row["ticker"] == selected_ticker else ""
            return f'<option value="{row["ticker"]}" {sel}>{row["ticker"]} (ADF={row["adf_stat"]:.2f}, H={row["H"]:.2f})</option>'

        qualifying_options = "".join(option(r) for _, r in qualifying.iterrows())
        other_options = "".join(option(r) for _, r in other.iterrows())
        auto_sel = "selected" if selected_ticker == "auto" else ""
        disabled = "disabled" if is_basket else ""
        note = " (ignored while basket size > 1 -- top-N qualifying tickers are picked automatically)" if is_basket else ""
        ticker_field_html = f"""
        <label style="grid-column: 1 / -1">Ticker{note} ({len(qualifying)}/{len(screening)} confirmed vol-mean-reverting)
          <select name="ticker" {disabled}>
            <option value="auto" {auto_sel}>Auto (strongest confirmed ticker by ADF stat)</option>
            <optgroup label="Confirmed">{qualifying_options or '<option disabled>none qualify</option>'}</optgroup>
            <optgroup label="Other screened">{other_options}</optgroup>
          </select>
        </label>"""

    def _val_str(v):
        return f"{v:g}" if isinstance(v, float) else str(v)  # avoid a trailing "100000.0"

    capital_field_html = "".join(
        f"""<label>{label}<input type="number" name="{key}"
        value="{_val_str(params[key])}" min="{lo}" max="{hi}" step="any"></label>"""
        for key, label, lo, hi, step in CAPITAL_FIELDS
    )
    basket_field_html = "".join(
        f"""<label>{label}<input type="number" name="{key}" value="{params[key]}" min="{lo}" max="{hi}" step="1"></label>"""
        for key, label, lo, hi, step in BASKET_FIELDS
    )

    body = f"""<title>Volatility Mean-Reversion Lab</title>
<style>{PAGE_CSS}</style>
<h1>Volatility Mean-Reversion Lab</h1>
<div class="subtitle">Individual BIST stock prices don't mean-revert (0/16 pass ADF+Hurst at any resolution) &mdash;
their realized volatility does, overwhelmingly (16/16). This trades THAT: vol spikes trigger a contrarian bet
(direction from Wilder's +DI/-DI) on the stock or a basket of stocks, sized down while it's stormy, exited once
volatility normalizes. See _engine_vol.py.</div>

<form method="get">
  <fieldset>
    <legend>Data ("how long")</legend>
    <div class="field-grid">{data_field_html}</div>
  </fieldset>
  <fieldset>
    <legend>Ticker selection</legend>
    <div class="field-grid">{ticker_field_html}</div>
  </fieldset>
  <fieldset>
    <legend>Basket ("trade more than one at once")</legend>
    <div class="field-grid">{basket_field_html}</div>
  </fieldset>
  <fieldset>
    <legend>Statistical tests &amp; backtest rules</legend>
    <div class="field-grid">{field_html}</div>
  </fieldset>
  <fieldset>
    <legend>Capital &amp; position sizing</legend>
    <div class="field-grid">{capital_field_html}</div>
  </fieldset>
  <button type="submit">Re-run backtest</button>
</form>
"""

    if error:
        body += f'<div class="error"><strong>Error:</strong> {error}</div>'
        return body

    if result is None and basket_result is None:
        body += "<p>Set parameters above and click Re-run to see results.</p>"
        return body

    r = basket_result if is_basket else result
    stats = r["stats"]
    sharpe = stats["sharpe (naive)"]
    sharpe_flag = sharpe > 3.0
    trades = r["trades"]

    if sharpe_flag:
        body += f"""<div class="banner"><strong>Sharpe of {sharpe:.2f} is implausibly high.</strong>
        With zero transaction costs this is most likely a microstructure artifact, not genuine edge -- check
        the cost-sensitivity table below for the approximate breakeven cost before trusting it.</div>"""

    gain_class = "pos" if r["final_capital"] >= r["starting_capital"] else "neg"
    identity_label = ("Basket", ", ".join(r["tickers"])) if is_basket else ("Ticker", r["ticker"])
    tiles = [
        (identity_label[0], identity_label[1], ""),
        ("Starting capital", f"{r['starting_capital']:,.0f} TRY", ""),
        ("Final capital (no cost)", f"{r['final_capital']:,.0f} TRY", gain_class),
        ("Total return", f"{r['total_return_pct']:+.1f}%", gain_class),
        ("Sharpe (no cost)", f"{sharpe:.2f}", ""),
        ("Ann. return", f"{stats['annualized return']*100:.1f}%", ""),
        ("Max drawdown", f"{stats['max drawdown']*100:.1f}%", ""),
        ("Trades", r["n_trades"], ""),
        ("Stop-losses", r["n_stops"], ""),
        ("% disqualified", f"{r['pct_disqualified']:.0f}%", ""),
    ]
    body += '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value {css_class}">{value}</div></div>'
        for label, value, css_class in tiles
    ) + "</div>"

    if is_basket:
        n = len(r["tickers"])
        pnl_chart = make_equity_chart(r["equity_curve"], r["split_idx"], r["starting_capital"], trades,
                                       r["per_ticker_results"][r["tickers"][0]]["results"]["timestamp"].to_numpy(),
                                       title_suffix=f", {n} tickers")
        body += f"""
        <h2>Portfolio equity curve</h2><img src="data:image/png;base64,{pnl_chart}">
        <p class="hint">{r['per_ticker_capital']:,.0f} TRY allocated to EACH of the {n} tickers (evenly split,
        {r['starting_capital']:,.0f} TRY total) &mdash; the sum across all of them never exceeds your total
        capital even if every position happens to be open at once. Each ticker's own GARCH volatility scale
        adjusts its slice independently.</p>
        <h2>Per-ticker breakdown</h2>
        <table><tr><th>Ticker</th><th>Trades</th><th>Sharpe (own)</th><th>Ann. return (own)</th></tr>{"".join(
            f"<tr><td>{tk}</td><td>{tr['n_trades']}</td><td>{tr['stats']['sharpe (naive)']:.2f}</td>"
            f"<td>{tr['stats']['annualized return']*100:+.1f}%</td></tr>"
            for tk, tr in r["per_ticker_results"].items()
        )}</table>
        """
    else:
        pnl_chart, z_chart, rv_chart, di_chart = make_single_ticker_charts(r, params["entry_z"], params["exit_z"], params["stop_z"])
        body += f"""
        <h2>Equity curve</h2><img src="data:image/png;base64,{pnl_chart}">
        <p class="hint">Base unit: starting capital &divide; {eng.SCALE_MAX:g} =
        <strong>{r['dollar_per_unit']:,.2f} TRY</strong>, then scaled bar by bar by the strategy's own
        GARCH volatility forecast (smaller when choppy, larger when calm), capped so exposure never exceeds
        your capital. Not full share/lot accounting &mdash; "roughly how much money," not an exact blotter.</p>
        <h2>Volatility z-score (the signal)</h2><img src="data:image/png;base64,{z_chart}">
        <h2>Realized volatility (raw)</h2><img src="data:image/png;base64,{rv_chart}">
        <h2>+DI / -DI (the direction decision)</h2><img src="data:image/png;base64,{di_chart}">
        """

    cost_rows = "".join(
        f"<tr class='{'neg' if row['sharpe (naive)']<=0 else ''}'><td>{bps}</td>"
        f"<td>{row['annualized return']*100:.1f}%</td><td>{row['sharpe (naive)']:.2f}</td></tr>"
        for bps, row in r["cost_table"].iterrows()
    )
    body += f"""<h2>Cost sensitivity</h2>
    <table><tr><th>bps/switch</th><th>Ann. return</th><th>Sharpe</th></tr>{cost_rows}</table>"""

    body += "<h2>Positions opened (trade log)</h2>"
    if trades:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df["pnl"] > 0).mean() * 100
        body += f"""<p style="font-size:0.85rem;color:var(--text-2)">{len(trades)} trades ·
        win rate {win_rate:.0f}% · avg hold {trades_df['hold_bars'].mean():.0f} bars ·
        exit reasons: {", ".join(f"{k} {int(v)}" for k, v in trades_df['exit_reason'].value_counts().items())} ·
        click a row for the real entry/exit price and +DI/-DI at entry.</p>"""
        body += build_trade_cards(trades)
    else:
        body += "<p>No trades were entered with these parameters — try lowering entry_z.</p>"

    body += f"""<h2>Ticker screening (in-sample, top 12)</h2>
    <table><tr><th>Ticker</th><th>ADF stat</th><th>H</th><th>Qualifies</th></tr>{"".join(
        f"<tr><td>{r2['ticker']}</td><td>{r2['adf_stat']:.3f}</td>"
        f"<td>{r2['H']:.3f}</td><td>{'✓' if r2['qualifies'] else ''}</td></tr>"
        for _, r2 in r['screening'].head(12).iterrows()
    )}</table>"""

    return body


@app.route("/", methods=["GET"])
def index():
    backtest_params = {
        key: request.args.get(key, type=int if key in INT_FIELDS else float, default=DEFAULT_BACKTEST_PARAMS[key])
        for key, *_ in FIELDS
    }
    capital_params = {
        key: request.args.get(key, type=float, default=DEFAULT_CAPITAL_PARAMS[key])
        for key, *_ in CAPITAL_FIELDS
    }
    basket_params = {
        key: request.args.get(key, type=int, default=DEFAULT_BASKET_PARAMS[key])
        for key, *_ in BASKET_FIELDS
    }

    interval = request.args.get("interval", default=DEFAULT_DATA_PARAMS["interval"])
    if interval not in PERIOD_OPTIONS_BY_INTERVAL:
        interval = DEFAULT_DATA_PARAMS["interval"]
    period = request.args.get("period") or PERIOD_OPTIONS_BY_INTERVAL[interval][-1]
    if period not in PERIOD_OPTIONS_BY_INTERVAL[interval]:
        period = PERIOD_OPTIONS_BY_INTERVAL[interval][-1]
    data_params = {"interval": interval, "period": period}

    ticker_param = request.args.get("ticker", default="auto")
    all_params = {**data_params, **backtest_params, **capital_params, **basket_params, "ticker": ticker_param}

    try:
        ohlc = get_ohlc(interval, period)
        screening_result = vol.screen_only(ohlc, backtest_params)
        screening = screening_result[0]

        if basket_params["n_tickers"] > 1:
            basket_result = vol.run_full_backtest_basket(ohlc, backtest_params, n_tickers=basket_params["n_tickers"])
            basket_result = vol.add_capital_pnl_basket(basket_result, capital_params["starting_capital"])
            return render_page(all_params, basket_result=basket_result, screening=screening)
        else:
            selected_ticker = None if ticker_param == "auto" else ticker_param
            result = vol.run_full_backtest(ohlc, backtest_params,
                                            screening_result=screening_result, selected_ticker=selected_ticker)
            result = eng.add_capital_pnl(result, capital_params["starting_capital"])
            return render_page(all_params, result=result, screening=screening)
    except Exception as e:
        return render_page(all_params, error=str(e))


if __name__ == "__main__":
    print("Starting Volatility Mean-Reversion Lab at http://127.0.0.1:5080")
    print("(Ctrl+C to stop)")
    app.run(port=5080, debug=False)
