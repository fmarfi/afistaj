"""
Interactive parameter lab for the DAILY BIST pairs strategy (examples/06).

This is the daily-bar counterpart of
intraday_pairs_trading/04_interactive_dashboard.py -- same idea (adjust
thresholds, pick among confirmed pairs, size positions in real currency,
see the trade log, all live in a browser) applied to the daily
walk-forward strategy instead of the intraday one. Deliberately kept as
its OWN file reusing examples/06's functions directly: the daily and
intraday strategies use different mechanics (a session/EOD-flatten
concept doesn't apply to daily bars, annualization uses 252 trading days
not ~95 bars/session, etc.), so they are NOT sharing one engine module --
each mission has its own, avoiding the two getting tangled together.

Run: python examples/08_interactive_dashboard.py
Then open http://127.0.0.1:8060 in a browser.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request
from importlib.util import spec_from_file_location, module_from_spec
import pathlib

_spec = spec_from_file_location("wf_bist", pathlib.Path(__file__).parent / "06_walk_forward_bist_pairs.py")
wf = module_from_spec(_spec)
_spec.loader.exec_module(wf)

app = Flask(__name__)
_price_cache = {}  # in-process cache keyed by period: avoid re-hitting yfinance on every form submit

BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8a86"

FIELDS = [
    ("in_sample_fraction", "In-sample fraction", 0.0, 1.0, 0.05),
    ("adf_p_threshold", "ADF p-value threshold (regime re-check)", 0.01, 0.5, 0.01),
    ("hurst_cutoff", "Hurst cutoff (regime re-check)", 0.1, 0.6, 0.01),
    ("lookback", "Lookback (trading days)", 60, 500, 10),
    ("rebalance", "Rebalance every (trading days)", 5, 60, 1),
    ("entry_z", "Entry z-score", 0.5, 4.0, 0.1),
    ("stop_z", "Stop-loss z-score", 1.0, 8.0, 0.1),
]
INT_FIELDS = {"lookback", "rebalance"}

PERIOD_OPTIONS = ["3y", "5y", "8y", "10y"]

CAPITAL_FIELDS = [
    ("starting_capital", "Starting capital (TRY)", 1000, 10_000_000, 1000),
    ("risk_per_trade_pct", "Risk per trade (% of capital)", 0.1, 20.0, 0.1),
]

DEFAULT_CAPITAL_PARAMS = {"starting_capital": 100_000.0, "risk_per_trade_pct": 1.0}
DEFAULT_BACKTEST_PARAMS = {
    "in_sample_fraction": wf.IN_SAMPLE_FRACTION,
    "adf_p_threshold": 0.10,
    "hurst_cutoff": 0.5,
    "lookback": wf.LOOKBACK,
    "rebalance": wf.REBALANCE,
    "entry_z": wf.ENTRY_Z,
    "stop_z": wf.STOP_Z,
}
DEFAULT_PERIOD = "8y"


def get_prices(period):
    if _price_cache.get("period") != period:
        _price_cache["prices"] = wf.download_universe(wf.BIST_UNIVERSE, period=period)
        _price_cache["period"] = period
    return _price_cache["prices"]


def screen_only(prices, params):
    log_prices = np.log(prices)
    n = len(prices)
    split_idx = int(n * params["in_sample_fraction"])
    screening = wf.screen_pairs_in_sample(
        log_prices, split_idx,
        adf_p_threshold=0.05, hurst_cutoff=0.45,  # in-sample SELECTION stays at the stricter default, matching examples/06
    )
    return screening, split_idx, log_prices


def run_backtest(prices, params, screening_result, selected_pair=None):
    screening, split_idx, log_prices = screening_result

    if selected_pair is not None:
        ta, tb = selected_pair
        match = screening[
            ((screening.ticker_a == ta) & (screening.ticker_b == tb)) |
            ((screening.ticker_a == tb) & (screening.ticker_b == ta))
        ]
        if len(match) == 0:
            raise ValueError(f"pair {ta}/{tb} not found among this universe's screened pairs")
        best = match.iloc[0]
    else:
        best = screening.iloc[0]

    ticker_a, ticker_b = best["ticker_a"], best["ticker_b"]
    log_a = log_prices[ticker_a].to_numpy()
    log_b = log_prices[ticker_b].to_numpy()
    n = len(prices)

    z_path, sigma_path, qualified_path, spread_path, beta_path = wf.walk_forward_backtest(
        log_a, log_b, split_idx, lookback=params["lookback"], rebalance=params["rebalance"],
        adf_p_threshold=params["adf_p_threshold"], hurst_cutoff=params["hurst_cutoff"],
    )
    pnl, position, trades, n_stops, n_regime_flat, n_trades = wf.run_signal_and_pnl_with_trades(
        z_path, sigma_path, qualified_path, spread_path, prices.index.to_numpy(), split_idx, n,
        entry_z=params["entry_z"], stop_z=params["stop_z"],
    )
    trades = wf.attach_trade_pnl(trades, pnl, prices.index.to_numpy())
    stats = wf.performance_stats(pnl, split_idx)
    cost_table = wf.cost_sensitivity_table(pnl, position, split_idx)

    results = pd.DataFrame({
        "timestamp": prices.index, "beta": beta_path, "z": z_path,
        "qualified": qualified_path, "sigma": sigma_path, "spread": spread_path,
        "position": position, "pnl": pnl,
    })

    return dict(
        ticker_a=ticker_a, ticker_b=ticker_b, screening=screening, results=results,
        stats=stats, cost_table=cost_table, trades=trades, split_idx=split_idx,
        n_stops=n_stops, n_regime_flat=n_regime_flat, n_trades=n_trades,
        pct_disqualified=round(float((~qualified_path[split_idx:]).mean() * 100), 1),
    )


def direction_label(direction, ticker_a, ticker_b):
    """
    "short spread" / "long spread" names neither actual stock. spread =
    log(ticker_a) - beta*log(ticker_b), so "short spread" (bet it falls)
    = sell ticker_a, buy ticker_b; "long spread" = buy ticker_a, sell
    ticker_b. (Deliberately a local copy, not imported from the intraday
    mission's _engine.py -- see this file's docstring on why.)
    """
    if direction == "short spread":
        return f"sell {ticker_a} / buy {ticker_b}"
    elif direction == "long spread":
        return f"buy {ticker_a} / sell {ticker_b}"
    return direction


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_charts(result, entry_z, stop_z):
    results = result["results"]
    oos = results[results["z"].notna()].reset_index(drop=True)
    split_idx = result["split_idx"]

    starting_capital = result["starting_capital"]
    equity_oos = result["equity_curve"][split_idx:]
    fig1, ax1 = plt.subplots(figsize=(8, 2.8))
    ax1.plot(equity_oos, color=BLUE, linewidth=1.3, zorder=2)
    ax1.axhline(starting_capital, color=GRAY, linewidth=0.8, linestyle="--")

    # Mark every trade on the equity curve: outlined triangle at entry,
    # filled dot at exit (green = profit, red = loss) -- lets a big-looking
    # total return be checked trade by trade instead of taken on faith.
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
    ax2.plot(x, oos["z"].to_numpy(), color=BLUE, linewidth=0.8)
    ax2.axhline(entry_z, color=GRAY, linestyle="--", linewidth=1)
    ax2.axhline(-entry_z, color=GRAY, linestyle="--", linewidth=1)
    ax2.axhline(stop_z, color=RED, linestyle=":", linewidth=1)
    ax2.axhline(-stop_z, color=RED, linestyle=":", linewidth=1)
    ax2.fill_between(x, ax2.get_ylim()[0], ax2.get_ylim()[1],
                      where=~oos["qualified"].to_numpy(dtype=bool), color=RED, alpha=0.08)
    ax2.set_title("Z-score (shaded = disqualified)")

    fig3, ax3 = plt.subplots(figsize=(8, 2.4))
    ax3.plot(x, oos["beta"].to_numpy(), color=BLUE, linewidth=1.0)
    ax3.axhline(0, color=GRAY, linewidth=0.8)
    ax3.set_title("Hedge ratio, re-estimated daily")

    return fig_to_base64(fig1), fig_to_base64(fig2), fig_to_base64(fig3)


def render_page(params, result=None, error=None, screening=None):
    field_html = ""
    for key, label, lo, hi, step in FIELDS:
        val = params[key]
        field_html += f"""
        <label>{label}
          <input type="number" name="{key}" value="{val}" min="{lo}" max="{hi}" step="any">
        </label>"""

    period_options = "".join(
        f'<option value="{p}" {"selected" if p == params["period"] else ""}>{p}</option>'
        for p in PERIOD_OPTIONS
    )
    data_field_html = f"""
        <label>History length
          <select name="period">{period_options}</select>
        </label>"""

    pair_field_html = ""
    if screening is not None:
        selected_pair = params.get("pair", "auto")
        qualifying = screening[screening["qualifies"]]
        other = screening[~screening["qualifies"]]

        def option(row):
            value = f"{row['ticker_a']}|{row['ticker_b']}"
            label = f"{row['pair']} (ADF p={row['adf_p']:.3f}, H={row['H']:.2f})"
            sel = "selected" if value == selected_pair else ""
            return f'<option value="{value}" {sel}>{label}</option>'

        qualifying_options = "".join(option(r) for _, r in qualifying.iterrows())
        other_options = "".join(option(r) for _, r in other.head(20).iterrows())
        auto_sel = "selected" if selected_pair == "auto" else ""
        pair_field_html = f"""
        <label style="grid-column: 1 / -1">Pair ({len(qualifying)}/{len(screening)} confirmed at current thresholds)
          <select name="pair">
            <option value="auto" {auto_sel}>Auto (best confirmed pair by ADF p-value)</option>
            <optgroup label="Confirmed pairs">{qualifying_options or '<option disabled>none qualify at these thresholds</option>'}</optgroup>
            <optgroup label="Other screened pairs (did not qualify)">{other_options}</optgroup>
          </select>
        </label>"""

    capital_field_html = ""
    for key, label, lo, hi, step in CAPITAL_FIELDS:
        val = params[key]
        val_str = f"{val:g}" if isinstance(val, float) else str(val)
        capital_field_html += f"""
        <label>{label}
          <input type="number" name="{key}" value="{val_str}" min="{lo}" max="{hi}" step="any">
        </label>"""

    body = f"""<title>Daily Backtest Lab</title>
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
  .banner {{ background: color-mix(in srgb, {ORANGE} 12%, var(--surface)); border: 1px solid {ORANGE}; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.85rem; }}
  .error {{ background: color-mix(in srgb, {RED} 15%, var(--surface)); border: 1px solid {RED}; border-radius: 8px; padding: 12px 16px; }}
</style>
<h1>Daily Backtest Lab</h1>
<div class="subtitle">Same strategy as examples/06, on daily bars, adjustable live -- pick a confirmed pair, tune thresholds, size in real currency. Not connected to the intraday mission's engine (see this file's docstring).</div>

<form method="get">
  <fieldset>
    <legend>Data</legend>
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
    <legend>Capital &amp; position sizing</legend>
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
    trades_df = pd.DataFrame(result["trades"]) if result["trades"] else pd.DataFrame()

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

    pnl_chart, z_chart, beta_chart = make_charts(result, params["entry_z"], params["stop_z"])
    body += f"""
    <h2>Equity curve</h2><img src="data:image/png;base64,{pnl_chart}">
    <p class="hint">Currency figures use a single conversion factor (capital
    &times; risk% &divide; typical forecast volatility = <strong>{result['dollar_per_unit']:,.2f} TRY per unit</strong>),
    applied uniformly across the run &mdash; not full per-leg share/lot accounting. Treat it as
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
        trade_rows = "".join(
            f"<tr><td>{t['entry_time']}</td><td>{t['exit_time']}</td>"
            f"<td>{direction_label(t['direction'], result['ticker_a'], result['ticker_b'])}</td>"
            f"<td>{t['entry_z']:.2f}</td><td>{t['exit_z']:.2f}</td><td>{t['hold_days']}</td>"
            f"<td>{t['exit_reason']}</td>"
            f"<td class='{'pos' if t['pnl_currency']>0 else 'neg'}'>{t['pnl_currency']:+,.0f} TRY</td></tr>"
            for t in trades_df.to_dict("records")
        )
        win_rate = (trades_df["pnl"] > 0).mean() * 100
        body += f"""<p style="font-size:0.85rem;color:var(--text-2)">{len(trades_df)} trades ·
        win rate {win_rate:.0f}% · avg hold {trades_df['hold_days'].mean():.0f} days ·
        exit reasons: {", ".join(f"{k} {int(v)}" for k, v in trades_df['exit_reason'].value_counts().items())}</p>
        <table><tr><th>Entry</th><th>Exit</th><th>Direction</th><th>Entry z</th><th>Exit z</th>
        <th>Hold (days)</th><th>Exit reason</th><th>PnL</th></tr>{trade_rows}</table>"""
    else:
        body += "<p>No trades were entered with these parameters — try lowering entry_z or the lookback window.</p>"

    body += f"""<h2>Pair screening (in-sample, top 8)</h2>
    <table><tr><th>Pair</th><th>Beta</th><th>ADF stat</th><th>ADF p</th><th>H</th><th>Qualifies</th></tr>{"".join(
        f"<tr><td>{r['pair']}</td><td>{r['beta']:.3f}</td><td>{r['adf_stat']:.3f}</td>"
        f"<td>{r['adf_p']:.4f}</td><td>{r['H']:.3f}</td><td>{'✓' if r['qualifies'] else ''}</td></tr>"
        for _, r in result['screening'].head(8).iterrows()
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
    period = request.args.get("period", default=DEFAULT_PERIOD)
    if period not in PERIOD_OPTIONS:
        period = DEFAULT_PERIOD
    pair_param = request.args.get("pair", default="auto")

    all_params = {"period": period, **backtest_params, **capital_params, "pair": pair_param}

    try:
        prices = get_prices(period)
        screening_result = screen_only(prices, backtest_params)
        screening = screening_result[0]

        selected_pair = None
        if pair_param != "auto":
            ta, tb = pair_param.split("|")
            selected_pair = (ta, tb)

        result = run_backtest(prices, backtest_params, screening_result, selected_pair=selected_pair)
        capital_fields = wf.add_capital_pnl(
            result["results"]["pnl"].to_numpy(), result["trades"], result["results"]["sigma"].to_numpy(),
            result["split_idx"], capital_params["starting_capital"], capital_params["risk_per_trade_pct"],
        )
        result.update(capital_fields)
        return render_page(all_params, result=result, screening=screening)
    except Exception as e:
        return render_page(all_params, error=str(e))


if __name__ == "__main__":
    print("Starting Daily Backtest Lab at http://127.0.0.1:8060")
    print("(Ctrl+C to stop)")
    app.run(port=8060, debug=False)
