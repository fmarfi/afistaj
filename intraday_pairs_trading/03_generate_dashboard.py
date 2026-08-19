"""
STEP 3 of the intraday workflow -- see WORKFLOW.md for the full picture.

Builds a single self-contained HTML dashboard (dashboard.html) from the
outputs of steps 1 and 2. Deliberately "basic": static charts (no
crosshair/tooltip layer), generated once per run rather than live --
enough to review a backtest, not a production monitoring tool.

Run this last, after 01 and 02 have both produced their output files.
"""

import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _engine as eng

# Categorical slots from the dataviz skill's reference palette -- assigned
# by ROLE (walk-forward path, cost/regime shading, disqualified state),
# not cycled arbitrarily.
BLUE = "#2a78d6"     # primary series: walk-forward path / spread
ORANGE = "#eb6834"   # secondary series: naive-cost comparison
AQUA = "#1baf7a"     # status: qualified / good
RED = "#e34948"      # status: disqualified / stop-loss / critical
GRAY = "#8a8a86"     # muted reference lines


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_cumulative_pnl(results, cost_table):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(np.arange(len(results)), np.nancumsum(results["pnl"].to_numpy()),
            color=BLUE, linewidth=1.4, label="no transaction costs")

    breakeven_rows = cost_table[cost_table["sharpe (naive)"] <= 0]
    if len(breakeven_rows):
        breakeven_bps = breakeven_rows.index[0]
    else:
        breakeven_bps = cost_table.index[-1]
    position = results["position"].to_numpy()
    pnl = results["pnl"].to_numpy()
    position_prev = np.roll(position, 1)
    position_prev[0] = 0.0
    turnover = np.abs(position - position_prev)
    cost = turnover * (breakeven_bps / 10000)
    pnl_after_cost = pnl - cost
    ax.plot(np.arange(len(results)), np.nancumsum(np.nan_to_num(pnl_after_cost)),
            color=ORANGE, linewidth=1.4, label=f"at {breakeven_bps} bps/switch (~breakeven)")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_title("Cumulative PnL: no costs vs. approx. breakeven cost")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel("out-of-sample bar")
    return fig_to_base64(fig)


def chart_zscore(results, entry_z, stop_z):
    z = results["z"].to_numpy()
    qualified = results["qualified"].to_numpy(dtype=bool)
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, z, color=BLUE, linewidth=0.7)
    ax.axhline(entry_z, color=GRAY, linestyle="--", linewidth=1)
    ax.axhline(-entry_z, color=GRAY, linestyle="--", linewidth=1)
    ax.axhline(stop_z, color=RED, linestyle=":", linewidth=1)
    ax.axhline(-stop_z, color=RED, linestyle=":", linewidth=1)
    ax.fill_between(x, -6, 6, where=~qualified, color=RED, alpha=0.08, label="disqualified (flat)")
    ax.set_ylim(-6, 6)
    ax.set_title("Intraday z-score (dashed = entry, dotted = hard stop, shaded = regime-break)")
    ax.legend(loc="upper left", frameon=False)
    return fig_to_base64(fig)


def chart_hedge_ratio(results):
    beta_by_session = results.groupby("session")["beta"].first().dropna()
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(beta_by_session.index, beta_by_session.values, color=BLUE, marker="o", markersize=3, linewidth=1.2)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_title("Hedge ratio, recalibrated once per session")
    ax.set_xlabel("session index")
    return fig_to_base64(fig)


def chart_garch_vol(results):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(np.arange(len(results)), results["sigma"].to_numpy(), color=RED, linewidth=0.9)
    ax.set_title("GARCH conditional volatility of the spread (per-session forecast)")
    ax.set_xlabel("out-of-sample bar")
    return fig_to_base64(fig)


def kpi_tile(label, value, status=None, wide_text=False):
    status_class = f" status-{status}" if status else ""
    value_class = "tile-value tile-value-text" if wide_text else "tile-value"
    return f'<div class="tile{status_class}"><div class="tile-label">{label}</div><div class="{value_class}">{value}</div></div>'


def build_html(summary, screening, cost_table, results, daily_pair, session_lookback, trades):
    entry_z, stop_z = 1.5, 3.5
    sharpe = summary["sharpe (naive)"]
    sharpe_flag = sharpe > 3.0
    breakeven_rows = cost_table[cost_table["sharpe (naive)"] <= 0]
    breakeven_bps = breakeven_rows.index[0] if len(breakeven_rows) else None

    # `results` spans the FULL price history; only rows from the
    # out-of-sample start onward have real z/qualified/sigma values --
    # everything before that is unfilled placeholder data (z is NaN there,
    # but `qualified` defaults to False, not NaN, which would misrender as
    # "disqualified" for the entire in-sample warm-up if not filtered out).
    oos_results = results[results["z"].notna()].reset_index(drop=True)

    pnl_chart = chart_cumulative_pnl(oos_results, cost_table)
    z_chart = chart_zscore(oos_results, entry_z, stop_z)
    beta_chart = chart_hedge_ratio(oos_results)
    vol_chart = chart_garch_vol(oos_results)

    tiles = "".join([
        kpi_tile("Pair", summary["pair"], wide_text=True),
        kpi_tile("Sharpe (no cost)", f"{sharpe:.2f}", status="critical" if sharpe_flag else "good"),
        kpi_tile("Ann. return", f"{summary['annualized return']*100:.1f}%"),
        kpi_tile("Ann. vol", f"{summary['annualized vol']*100:.1f}%"),
        kpi_tile("Max drawdown", f"{summary['max drawdown']*100:.1f}%"),
        kpi_tile("Trades", int(summary["trades"])),
        kpi_tile("Stop-losses", int(summary["stop_losses"])),
        kpi_tile("% time disqualified", f"{summary['pct_disqualified']:.0f}%"),
        kpi_tile("Cost breakeven", f"{breakeven_bps} bps" if breakeven_bps is not None else "> tested range",
                 status="warning" if breakeven_bps is not None and breakeven_bps <= 10 else None),
    ])

    warning_banner = ""
    if sharpe_flag:
        warning_banner = f"""
<div class="banner banner-critical">
  <strong>Sharpe of {sharpe:.2f} is implausibly high for a live strategy.</strong>
  With zero transaction costs modeled on 5-minute bars, this is most likely bid-ask
  bounce, not genuine edge. See the cost-sensitivity table below — this backtest's
  edge crosses zero at approximately <strong>{breakeven_bps} bps</strong> of round-trip
  cost per position change. Compare that against a real quoted spread for
  {summary['pair']} before treating this as tradeable. See WORKFLOW.md, Checkpoint 4.
</div>"""

    screening_top = screening.head(10).copy()
    screening_top["qualifies"] = screening_top["qualifies"].map(lambda v: "✓" if v else "")
    screening_rows = "".join(
        f"<tr><td>{r['pair']}</td><td>{r['beta']:.3f}</td><td>{r['adf_stat']:.3f}</td>"
        f"<td>{r['H']:.3f}</td><td class='center'>{r['qualifies']}</td></tr>"
        for _, r in screening_top.iterrows()
    )

    cost_rows = "".join(
        f"<tr class='{'row-negative' if row['sharpe (naive)'] <= 0 else ''}'>"
        f"<td>{bps}</td><td>{row['annualized return']*100:.1f}%</td>"
        f"<td>{row['annualized vol']*100:.1f}%</td><td>{row['sharpe (naive)']:.2f}</td></tr>"
        for bps, row in cost_table.iterrows()
    )

    if trades is not None and len(trades):
        trade_ticker_a, trade_ticker_b = summary["pair"].split("/")
        win_rate = (trades["pnl"] > 0).mean() * 100
        exit_breakdown = ", ".join(f"{k} {int(v)}" for k, v in trades["exit_reason"].value_counts().items())
        trades_summary = (f"{len(trades)} trades &middot; win rate {win_rate:.0f}% &middot; "
                           f"avg hold {trades['hold_bars'].mean():.0f} bars &middot; exit reasons: {exit_breakdown} &middot; "
                           f"click a row to see the actual entry/exit price of both legs.")
        trade_cards = eng.build_trade_cards_html(trades.to_dict("records"), trade_ticker_a, trade_ticker_b)
        trades_section = f"""<h2>Positions opened (trade log)</h2>
<p class="caveats">{trades_summary}</p>
{trade_cards}
<p class="guide"><strong>How to read this:</strong> each card is one completed position, in the order
  it was opened. "Direction" is which way the SPREAD was traded (see WORKFLOW.md's "strategy itself"
  section for what long/short spread means in terms of the two underlying legs), not the raw price
  direction of either stock. "Exit reason" tells you WHY it closed: <em>reversion</em> is the intended,
  healthy outcome (the z-score came back to 0 as expected); <em>hard stop</em> means it moved further
  against the position even after entry (see the STOP_Z threshold in WORKFLOW.md); <em>end of day</em>
  means it was still open when the session ended and was force-flattened per the no-overnight-risk
  rule, regardless of PnL at that moment. A pair with mostly "reversion" exits and only a few stops is
  behaving as designed; many "hard stop" exits suggest ENTRY_Z or STOP_Z may need revisiting. Expand a
  card to see the exact quoted price of both legs at entry and exit -- the real, checkable numbers
  behind the PnL, not just the abstract z-score.</p>"""
    else:
        trades_section = "<h2>Positions opened (trade log)</h2><p>No trades were entered during this run.</p>"

    beta_by_session = results.groupby("session")["beta"].first().dropna()
    beta_range_text = f"{beta_by_session.min():.2f} to {beta_by_session.max():.2f}"
    beta_flag = (beta_by_session.max() - beta_by_session.min() > 0.5) or (beta_by_session.min() < 0 < beta_by_session.max())
    breakeven_text = f"{breakeven_bps} bps" if breakeven_bps is not None else "beyond the range tested (20 bps)"

    guide_pnl = f"""<p class="guide"><strong>How to read this:</strong> the blue line is cumulative
      profit with no trading costs at all; the orange line applies a per-trade cost right around
      this backtest's approximate breakeven ({breakeven_text}/switch, see the cost-sensitivity table
      below). A blue line that climbs steadily while the orange line goes flat or negative is the
      visual version of Checkpoint 4 in WORKFLOW.md: the strategy's apparent edge, and whether it
      actually survives realistic costs, can be two very different stories — compare the shapes,
      not just the endpoints.</p>"""

    guide_z = f"""<p class="guide"><strong>How to read this:</strong> the strategy enters a trade
      when the blue line crosses the dashed gray line (±{entry_z:g}), exits when it crosses back
      through zero, and hard-stops if it reaches the dotted red line (±{stop_z:g}). Pink shading
      marks periods the pair failed its periodic cointegration re-check and was disqualified —
      flat, no new trades, regardless of where the z-score sits. If pink covers most of the chart,
      the statistical relationship this whole strategy depends on wasn't reliably present during
      this window; a strategy that's rarely qualified to trade isn't a strategy you can lean on.</p>"""

    guide_beta = f"""<p class="guide"><strong>How to read this:</strong> each point is the hedge
      ratio re-estimated at the start of one session, using only the {session_lookback} sessions
      before it. A slowly drifting line is expected and healthy — the relationship between the two
      stocks evolving over time is exactly why the ratio gets re-estimated instead of fixed forever.
      Sharp jumps or a sign flip (as happened here: {beta_range_text}) are a different story —
      that's noise from too little data per fit, not a real shift in the relationship, and it's
      flagged automatically in Step 2's console output when it happens.</p>""" if beta_flag else f"""<p class="guide"><strong>How to read this:</strong> each point is the hedge
      ratio re-estimated at the start of one session, using only the {session_lookback} sessions
      before it. A slowly drifting line is expected and healthy; sharp jumps or sign flips would
      indicate the window is too short to pin the relationship down reliably — this run's range
      ({beta_range_text}) stayed reasonably contained.</p>"""

    guide_vol = """<p class="guide"><strong>How to read this:</strong> this is the GARCH model's
      forecast of how much the spread is expected to jump around, refit once per session and held
      constant for that whole session's bars. Position size is scaled DOWN when this line is high
      and UP when it's low, targeting roughly constant risk per trade rather than a fixed bet size
      regardless of conditions — spikes here should correspond to smaller positions being taken in
      the PnL chart above, not larger ones.</p>"""

    guide_cost = """<p class="guide"><strong>How to read this:</strong> each row re-runs the exact
      same trades with a different per-switch cost subtracted (charged once per entry, exit, or
      flip). Rows shown in red have gone Sharpe-negative. Find the row where Sharpe crosses zero,
      then compare that number of bps against what this pair would actually cost to trade — if the
      real spread is wider than the breakeven row, the strategy loses money once you account for
      trading it, no matter how good the zero-cost number looked.</p>"""

    guide_screening = """<p class="guide"><strong>How to read this:</strong> beta is the OLS hedge
      ratio; ADF stat more negative than about -2.86 rejects the "random walk" null (evidence of
      mean reversion); H (Hurst Exponent) below 0.45 independently points the same direction. A row
      only "qualifies" when both agree — two different statistical routes landing on the same
      conclusion is meaningfully stronger evidence than either alone.</p>"""

    return f"""<title>Intraday Pairs Trading Dashboard</title>
<style>
  :root {{
    --surface: #fcfcfb; --surface-2: #f3f2ef; --border: #e4e2dd;
    --text-primary: #0b0b0b; --text-secondary: #52514e;
    --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface: #1a1a19; --surface-2: #232320; --border: #34332f;
      --text-primary: #ffffff; --text-secondary: #c3c2b7;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--surface); color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 32px; line-height: 1.5;
  }}
  h1 {{ font-size: 1.4rem; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 24px; }}
  h2 {{ font-size: 1.05rem; margin: 32px 0 12px; border-top: 1px solid var(--border); padding-top: 24px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }}
  .tile {{
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px;
  }}
  .tile-label {{ font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 4px; }}
  .tile-value {{ font-size: 1.3rem; font-weight: 600; overflow-wrap: break-word; }}
  .tile-value-text {{ font-size: 0.95rem; }}
  .status-good .tile-value {{ color: var(--good); }}
  .status-warning .tile-value {{ color: var(--warning); }}
  .status-critical .tile-value {{ color: var(--critical); }}
  .banner {{ border-radius: 8px; padding: 16px 20px; margin: 20px 0; font-size: 0.92rem; }}
  .banner-critical {{ background: color-mix(in srgb, var(--critical) 12%, var(--surface)); border: 1px solid var(--critical); }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-secondary); font-weight: 500; }}
  td.center {{ text-align: center; }}
  .row-negative {{ color: var(--critical); }}
  .row-positive {{ color: var(--good); }}
  .caveats {{ font-size: 0.85rem; color: var(--text-secondary); }}
  .caveats li {{ margin-bottom: 6px; }}
  .guide {{
    font-size: 0.85rem; color: var(--text-secondary); margin: 10px 0 0;
    background: var(--surface-2); border-left: 3px solid {BLUE}; border-radius: 4px;
    padding: 10px 14px;
  }}
  .guide strong {{ color: var(--text-primary); }}
  {eng.TRADE_CARD_CSS}
</style>

<h1>Intraday Pairs Trading — Backtest Dashboard</h1>
<div class="subtitle">5-minute bars, BIST universe · generated from 01_screen_intraday_pairs.py + 02_intraday_walk_forward.py output · see WORKFLOW.md for the full workflow</div>

{warning_banner}

<div class="tiles">{tiles}</div>

<h2>Cumulative PnL</h2>
<img src="data:image/png;base64,{pnl_chart}" alt="Cumulative PnL chart">
{guide_pnl}

{trades_section}

<h2>Signal: intraday z-score</h2>
<img src="data:image/png;base64,{z_chart}" alt="Z-score chart">
{guide_z}

<h2>Hedge ratio stability</h2>
<img src="data:image/png;base64,{beta_chart}" alt="Hedge ratio chart">
{guide_beta}

<h2>Volatility (GARCH)</h2>
<img src="data:image/png;base64,{vol_chart}" alt="GARCH volatility chart">
{guide_vol}

<h2>Cost sensitivity</h2>
<table>
  <tr><th>bps / switch</th><th>Ann. return</th><th>Ann. vol</th><th>Sharpe</th></tr>
  {cost_rows}
</table>
{guide_cost}

<h2>Pair screening (in-sample, top 10 of {len(screening)})</h2>
<table>
  <tr><th>Pair</th><th>Beta</th><th>ADF stat</th><th>Hurst H</th><th>Qualifies</th></tr>
  {screening_rows}
</table>
{guide_screening}
<p class="caveats">Daily-selected pair for comparison: <strong>{daily_pair[0]}/{daily_pair[1]}</strong> (see 01's console output for its intraday ADF/Hurst numbers).</p>

<h2>Caveats</h2>
<ul class="caveats">
  <li>~60 calendar days of 5-minute history only (yfinance's limit) — an order of magnitude less data than examples/06's 5-year daily backtest. Treat every number here as lower-confidence.</li>
  <li>No transaction costs, bid-ask spread, or slippage in the headline numbers — see the cost-sensitivity table above for what survives realistic costs.</li>
  <li>Positions are forced flat at the end of every session (day-trading discipline, no overnight risk) and whenever the rolling cointegration re-check fails.</li>
  <li>This is a research/backtesting tool, not an execution system — it does not place real orders.</li>
</ul>
"""


def main():
    try:
        results = pd.read_pickle("intraday_pairs_trading/_walk_forward_results.pkl")
        summary = pd.read_csv("intraday_pairs_trading/_backtest_summary.csv").iloc[0]
        cost_table = pd.read_csv("intraday_pairs_trading/_cost_sensitivity.csv", index_col=0)
        screening = pd.read_csv("intraday_pairs_trading/screening_results.csv")
    except FileNotFoundError as e:
        print(f"Missing an earlier step's output ({e}). Run 01 then 02 first.")
        return

    try:
        trades = pd.read_csv("intraday_pairs_trading/_trades.csv")
    except FileNotFoundError:
        trades = None  # 02 saves this only when at least one trade was entered

    # The pair examples/06 selected at the daily timescale, for comparison
    # in the dashboard footer -- see 01_screen_intraday_pairs.py's
    # DAILY_SELECTED_PAIR constant.
    daily_pair = eng.DAILY_SELECTED_PAIR
    session_lookback = eng.DEFAULTS["session_lookback"]

    html = build_html(summary, screening, cost_table, results, daily_pair, session_lookback, trades)
    out_path = "intraday_pairs_trading/dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved {out_path} -- open it directly in a browser.")


if __name__ == "__main__":
    main()
