"""
STEP 4 (optional) of the intraday workflow -- see WORKFLOW.md.

A local Flask app for adjusting backtest AND statistical-test parameters
live and seeing results recompute, including the actual positions the
strategy opened (a trade-by-trade log, not just aggregate stats).

This is genuinely different from 03_generate_dashboard.py: that script
renders one static report for the DEFAULT parameters. This one lets you
change entry_z, stop_z, session_lookback, z_window, the ADF critical
value, the Hurst cutoff, and in_sample_fraction, then re-runs the full
screen -> select -> walk-forward -> trade-log pipeline (via _engine.py --
the exact same code 01/02/03 use) and shows you the new result. Price
data is downloaded once and cached on disk; changing parameters re-runs
the computation, not a new network fetch.

Run: python intraday_pairs_trading/04_interactive_dashboard.py
Then open http://127.0.0.1:5050 in a browser.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string

import _engine as eng

app = Flask(__name__)
_price_cache = {}  # in-process cache keyed by (interval, period): avoid re-hitting yfinance on every form submit

BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8a86"

# (key, label, min, max, step) -- step is only used for the +/- spinner
# buttons now; validation uses step="any" in the HTML (see render_page)
# to avoid floating-point step-mismatch false rejections, e.g. -2.5 being
# refused because it doesn't land on an exact float multiple of the step
# starting from min.
FIELDS = [
    ("in_sample_fraction", "In-sample fraction", 0.0, 1.0, 0.05),
    ("adf_crit", "ADF critical value", -6.0, -1.0, 0.05),
    ("hurst_cutoff", "Hurst cutoff", 0.1, 0.6, 0.01),
    ("session_lookback", "Session lookback (sessions)", 2, 15, 1),
    ("z_window", "Z-score window (bars)", 10, 150, 5),
    ("entry_z", "Entry z-score", 0.5, 4.0, 0.1),
    ("stop_z", "Stop-loss z-score", 1.0, 8.0, 0.1),
]

# yfinance's real limits per bar interval -- how far back history can go.
# "how long" is fundamentally capped by this, not by anything in _engine.py.
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

CAPITAL_FIELDS = [
    ("starting_capital", "Starting capital (TRY)", 1000, 10_000_000, 1000),
]


def get_prices(interval, period):
    cache_key = (interval, period)
    if _price_cache.get("key") != cache_key:
        _price_cache["prices"] = eng.download_intraday_universe(eng.BIST_UNIVERSE, interval=interval, period=period)
        _price_cache["key"] = cache_key
    return _price_cache["prices"]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_charts(result):
    results = result["results"]
    oos = results[results["z"].notna()].reset_index(drop=True)
    split_idx = result["split_idx"]

    starting_capital = result["starting_capital"]
    equity_oos = result["equity_curve"][split_idx:]
    fig1, ax1 = plt.subplots(figsize=(8, 2.8))
    ax1.plot(equity_oos, color=BLUE, linewidth=1.3, zorder=2)
    ax1.axhline(starting_capital, color=GRAY, linewidth=0.8, linestyle="--")

    # Mark every trade on the equity curve itself: an outlined triangle
    # where it was opened, a filled dot (green = profit, red = loss)
    # where it closed -- so "the profit looks unbelievable" can be
    # checked trade by trade instead of taken on faith from one aggregate
    # number. oos's positional index lines up 1:1 with equity_oos because
    # both start at split_idx and z is only ever non-NaN from there on.
    ts_to_pos = {ts: i for i, ts in enumerate(oos["timestamp"])}
    for t in result["trades"]:
        entry_pos = ts_to_pos.get(t["entry_time"])
        exit_pos = ts_to_pos.get(t["exit_time"])
        if entry_pos is None or exit_pos is None:
            continue
        win = t["pnl_currency"] > 0
        ax1.scatter(entry_pos, equity_oos[entry_pos], marker="^", s=28,
                     facecolors="none", edgecolors=GRAY, linewidths=1.1, zorder=3)
        ax1.scatter(exit_pos, equity_oos[exit_pos], marker="o", s=26,
                     color=AQUA if win else RED, zorder=3)
    ax1.set_title(f"Equity curve, starting capital {starting_capital:,.0f} TRY (no trading costs)  "
                  f"– △ entry, ● exit (green=profit, red=loss)")

    fig2, ax2 = plt.subplots(figsize=(8, 2.8))
    x = np.arange(len(oos))
    ax2.plot(x, oos["z"].to_numpy(), color=BLUE, linewidth=0.7)
    ax2.axhline(result["params"]["entry_z"], color=GRAY, linestyle="--", linewidth=1)
    ax2.axhline(-result["params"]["entry_z"], color=GRAY, linestyle="--", linewidth=1)
    ax2.axhline(result["params"]["stop_z"], color=RED, linestyle=":", linewidth=1)
    ax2.axhline(-result["params"]["stop_z"], color=RED, linestyle=":", linewidth=1)
    ax2.fill_between(x, ax2.get_ylim()[0], ax2.get_ylim()[1],
                      where=~oos["qualified"].to_numpy(dtype=bool), color=RED, alpha=0.08)
    ax2.set_title("Z-score (shaded = disqualified)")

    beta_by_session = oos.groupby("session")["beta"].first()
    fig3, ax3 = plt.subplots(figsize=(8, 2.4))
    ax3.plot(beta_by_session.index, beta_by_session.values, color=BLUE, marker="o", markersize=3)
    ax3.axhline(0, color=GRAY, linewidth=0.8)
    ax3.set_title("Hedge ratio per session")

    return fig_to_base64(fig1), fig_to_base64(fig2), fig_to_base64(fig3)


def build_trade_cards(trades_df, ticker_a, ticker_b):
    """Thin wrapper around _engine's shared card renderer -- see its docstring for why cards, not a wider table."""
    return eng.build_trade_cards_html(trades_df.to_dict("records"), ticker_a, ticker_b)


def render_page(params, result=None, error=None, screening=None):
    # step="any" everywhere: the browser's native step-mismatch validation
    # ("please enter a valid value") is unreliable with floats because it
    # compares against exact multiples of `step` from `min`, and binary
    # floating point can't represent values like 0.05 exactly -- e.g.
    # -2.5 could get rejected as "invalid" purely from rounding error even
    # though it's clearly a sane value in range. min/max are kept as
    # loose guidance (still enforced), just not the step granularity.
    field_html = ""
    for key, label, lo, hi, step in FIELDS:
        val = params[key]
        field_html += f"""
        <label>{label}
          <input type="number" name="{key}" value="{val}" min="{lo}" max="{hi}" step="any">
        </label>"""

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

    pair_field_html = ""
    if screening is not None:
        selected_pair = params.get("pair", "auto")
        qualifying = screening[screening["qualifies"]]
        other = screening[~screening["qualifies"]]

        def option(row):
            value = f"{row['ticker_a']}|{row['ticker_b']}"
            label = f"{row['pair']} (ADF={row['adf_stat']:.2f}, H={row['H']:.2f})"
            sel = "selected" if value == selected_pair else ""
            return f'<option value="{value}" {sel}>{label}</option>'

        qualifying_options = "".join(option(r) for _, r in qualifying.iterrows())
        other_options = "".join(option(r) for _, r in other.head(20).iterrows())
        auto_sel = "selected" if selected_pair == "auto" else ""
        pair_field_html = f"""
        <label style="grid-column: 1 / -1">Pair ({len(qualifying)}/{len(screening)} confirmed at current thresholds)
          <select name="pair">
            <option value="auto" {auto_sel}>Auto (best confirmed pair by ADF stat)</option>
            <optgroup label="Confirmed pairs">{qualifying_options or '<option disabled>none qualify at these thresholds</option>'}</optgroup>
            <optgroup label="Other screened pairs (did not qualify)">{other_options}</optgroup>
          </select>
        </label>"""

    capital_field_html = ""
    for key, label, lo, hi, step in CAPITAL_FIELDS:
        val = params[key]
        val_str = f"{val:g}" if isinstance(val, float) else str(val)  # avoid a trailing "100000.0"
        capital_field_html += f"""
        <label>{label}
          <input type="number" name="{key}" value="{val_str}" min="{lo}" max="{hi}" step="any">
        </label>"""

    body = f"""<title>Intraday Backtest Lab</title>
<style>
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
  .tile-value {{ font-size: 1.15rem; font-weight: 600; }}
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
</style>
<h1>Intraday Backtest Lab</h1>
<div class="subtitle">Adjust parameters and re-run the screen → select → walk-forward → trade-log pipeline live. Same engine as 01/02/03 (_engine.py) — see WORKFLOW.md.</div>

<form method="get">
  <fieldset>
    <legend>Data ("how long")</legend>
    <div class="field-grid">{data_field_html}</div>
  </fieldset>
  <fieldset>
    <legend>Pair selection</legend>
    <div class="field-grid">{pair_field_html}</div>
  </fieldset>
  <fieldset>
    <legend>Statistical tests &amp; backtest rules</legend>
    <div class="field-grid">{field_html}</div>
  </fieldset>
  <fieldset>
    <legend>Capital &amp; position sizing ("how much money")</legend>
    <div class="field-grid">{capital_field_html}</div>
  </fieldset>
  <button type="submit">Re-run backtest</button>
</form>
"""

    if error:
        body += f'<div class="error"><strong>Error:</strong> {error}</div>'
        return body

    if result is None:
        body += "<p>Set parameters above and click Re-run to see results.</p>"
        return body

    stats = result["stats"]
    sharpe = stats["sharpe (naive)"]
    sharpe_flag = sharpe > 3.0
    trades_df = pd.DataFrame(result["trades"]) if result["trades"] else pd.DataFrame()

    if sharpe_flag:
        body += f"""<div class="banner"><strong>Sharpe of {sharpe:.2f} is implausibly high.</strong>
        With zero transaction costs on 5-minute bars this is most likely bid-ask bounce, not genuine
        edge -- check the cost-sensitivity table below for the approximate breakeven cost. See WORKFLOW.md, Checkpoint 4.</div>"""

    gain_class = "pos" if result["final_capital"] >= result["starting_capital"] else "neg"
    tiles = [
        ("Pair", f"{result['ticker_a']}/{result['ticker_b']}", ""),
        ("Starting capital", f"{result['starting_capital']:,.0f} TRY", ""),
        ("Final capital (no cost)", f"{result['final_capital']:,.0f} TRY", gain_class),
        ("Total return", f"{result['total_return_pct']:+.1f}%", gain_class),
        ("Sharpe (no cost)", f"{sharpe:.2f}", ""),
        ("Ann. return", f"{stats['annualized return']*100:.1f}%", ""),
        ("Max drawdown", f"{stats['max drawdown']*100:.1f}%", ""),
        ("Trades", result["n_trades"], ""),
        ("Stop-losses", result["n_stops"], ""),
        ("% disqualified", f"{result['pct_disqualified']:.0f}%", ""),
    ]
    body += '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value {css_class}">{value}</div></div>'
        for label, value, css_class in tiles
    ) + "</div>"

    pnl_chart, z_chart, beta_chart = make_charts(result)
    body += f"""
    <h2>Equity curve</h2><img src="data:image/png;base64,{pnl_chart}">
    <p class="hint">Currency figures use one base unit (starting capital &divide; {eng.SCALE_MAX:g} =
    <strong>{result['dollar_per_unit']:,.2f} TRY</strong>) that the strategy's own GARCH volatility scale
    then adjusts bar by bar &mdash; smaller exposure when choppy, larger when calm, capped so it never
    exceeds your capital. Not full per-leg share/lot accounting &mdash; treat it as
    "roughly how much money," not an exact trade blotter.</p>
    <h2>Z-score</h2><img src="data:image/png;base64,{z_chart}">
    <h2>Hedge ratio</h2><img src="data:image/png;base64,{beta_chart}">
    """

    cost_rows = "".join(
        f"<tr class='{'neg' if row['sharpe (naive)']<=0 else ''}'><td>{bps}</td>"
        f"<td>{row['annualized return']*100:.1f}%</td><td>{row['sharpe (naive)']:.2f}</td></tr>"
        for bps, row in result["cost_table"].iterrows()
    )
    body += f"""<h2>Cost sensitivity</h2>
    <table><tr><th>bps/switch</th><th>Ann. return</th><th>Sharpe</th></tr>{cost_rows}</table>"""

    body += "<h2>Positions opened (trade log)</h2>"
    if len(trades_df):
        win_rate = (trades_df["pnl"] > 0).mean() * 100
        body += f"""<p style="font-size:0.85rem;color:var(--text-2)">{len(trades_df)} trades ·
        win rate {win_rate:.0f}% · avg hold {trades_df['hold_bars'].mean():.0f} bars ·
        exit reasons: {", ".join(f"{k} {int(v)}" for k, v in trades_df['exit_reason'].value_counts().items())} ·
        click a row to see the actual entry/exit price of both legs.</p>"""
        body += build_trade_cards(trades_df, result['ticker_a'], result['ticker_b'])
    else:
        body += "<p>No trades were entered with these parameters — try lowering entry_z or raising session_lookback.</p>"

    body += f"""<h2>Pair screening (in-sample, top 8)</h2>
    <table><tr><th>Pair</th><th>Beta</th><th>ADF stat</th><th>H</th><th>Qualifies</th></tr>{"".join(
        f"<tr><td>{r['pair']}</td><td>{r['beta']:.3f}</td><td>{r['adf_stat']:.3f}</td>"
        f"<td>{r['H']:.3f}</td><td>{'✓' if r['qualifies'] else ''}</td></tr>"
        for _, r in result['screening'].head(8).iterrows()
    )}</table>"""

    return body


INT_FIELDS = {"session_lookback", "z_window"}  # every other field in FIELDS is a float

DEFAULT_CAPITAL_PARAMS = {"starting_capital": 100_000.0}
DEFAULT_DATA_PARAMS = {"interval": eng.BIST_INTERVAL, "period": eng.BIST_PERIOD}


@app.route("/", methods=["GET"])
def index():
    backtest_params = {
        key: request.args.get(key, type=int if key in INT_FIELDS else float, default=eng.DEFAULTS[key])
        for key, *_ in FIELDS
    }
    capital_params = {
        key: request.args.get(key, type=float, default=DEFAULT_CAPITAL_PARAMS[key])
        for key, *_ in CAPITAL_FIELDS
    }

    interval = request.args.get("interval", default=DEFAULT_DATA_PARAMS["interval"])
    if interval not in PERIOD_OPTIONS_BY_INTERVAL:
        interval = DEFAULT_DATA_PARAMS["interval"]
    period = request.args.get("period") or PERIOD_OPTIONS_BY_INTERVAL[interval][-1]
    if period not in PERIOD_OPTIONS_BY_INTERVAL[interval]:
        period = PERIOD_OPTIONS_BY_INTERVAL[interval][-1]
    data_params = {"interval": interval, "period": period}

    pair_param = request.args.get("pair", default="auto")
    all_params = {**data_params, **backtest_params, **capital_params, "pair": pair_param}

    try:
        prices = get_prices(interval, period)
        screening_result = eng.screen_only(prices, backtest_params)
        screening = screening_result[0]

        selected_pair = None
        if pair_param != "auto":
            ta, tb = pair_param.split("|")
            selected_pair = (ta, tb)

        result = eng.run_full_backtest(prices, backtest_params,
                                        screening_result=screening_result, selected_pair=selected_pair)
        result["params"] = backtest_params
        result = eng.add_capital_pnl(result, capital_params["starting_capital"])
        return render_page(all_params, result=result, screening=screening)
    except Exception as e:
        return render_page(all_params, error=str(e))


if __name__ == "__main__":
    print("Starting Intraday Backtest Lab at http://127.0.0.1:5050")
    print("(Ctrl+C to stop)")
    app.run(port=5050, debug=False)
