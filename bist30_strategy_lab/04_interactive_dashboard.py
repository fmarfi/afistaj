"""
Interactive dashboard for this mission: a LEADERBOARD tab (reads 02/03's
cached, DSR-corrected full-universe comparison -- fast, no recompute) and
an EXPLORE tab (live single-backtest parameter tweaking per family, same
engines 02/03 use, same shape as intraday_pairs_trading/04's screen-then-
select UX).

Run 02_run_full_universe_backtests.py and 03_run_validation_leaderboard.py
FIRST to populate the leaderboard tab's cache.

Run: python bist30_strategy_lab/04_interactive_dashboard.py
Then open http://127.0.0.1:5100 in a browser.
"""

import io
import base64
import pickle
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string

import universe as uni
import _common as c
import _engine_meanrev as em
import _engine_momentum as mo
import _engine_pca as pca

app = Flask(__name__)
_price_cache = {}
LEADERBOARD_PKL = pathlib.Path(__file__).parent / "_leaderboard_detail.pkl"

BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8a86"

FAMILIES = {
    "mean_reversion_ou": {"label": "Mean reversion (OU half-life pairs)", "module": em, "run": em.run_full_backtest_meanrev},
    "momentum": {"label": "Momentum (cross-sectional rotation)", "module": mo, "run": mo.run_full_backtest_momentum},
    "pca_stat_arb": {"label": "PCA basket stat-arb", "module": pca, "run": pca.run_full_backtest_pca},
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
FIELDS_BY_FAMILY = {"mean_reversion_ou": MEANREV_FIELDS, "momentum": MOMENTUM_FIELDS, "pca_stat_arb": PCA_FIELDS}


def get_prices(period):
    if _price_cache.get("period") != period:
        _price_cache["prices"] = uni.download_universe(period=period)
        _price_cache["period"] = period
    return _price_cache["prices"]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def equity_chart(result, family_label):
    equity = result["equity_curve"][result["split_idx"]:]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(equity, color=BLUE, linewidth=1.3)
    ax.axhline(result["starting_capital"], color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_title(f"{family_label} -- equity curve, out-of-sample (no trading costs)")
    return fig_to_base64(fig)


def leaderboard_chart(board):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = [AQUA if d > 0.5 else (ORANGE if d > 0.2 else RED) for d in board["dsr"]]
    ax.barh(board["family"], board["dsr"], color=colors)
    ax.set_xlabel("Deflated Sharpe Ratio (P[true Sharpe > 0], trial-corrected)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    return fig_to_base64(fig)


def render_trades_table(trades, limit=200):
    if not trades:
        return "<p>No trades were entered with these parameters -- try loosening entry_z or the trend/regime filter.</p>"
    df = pd.DataFrame(trades)
    cols_priority = ["ticker", "direction", "entry_time", "exit_time", "entry_z", "exit_z",
                      "hold_bars", "exit_reason", "pnl_currency"]
    cols = [col for col in cols_priority if col in df.columns]
    df_show = df[cols].head(limit).copy()
    for col in ("entry_z", "exit_z"):
        if col in df_show:
            df_show[col] = df_show[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    if "pnl_currency" in df_show:
        df_show["pnl_currency"] = df_show["pnl_currency"].map(lambda x: f"{x:+,.0f} TRY")
    header = "".join(f"<th>{col}</th>" for col in df_show.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td class='{'pos' if str(v).startswith('+') else ('neg' if str(v).startswith('-') else '')}'>{v}</td>" for v in row) + "</tr>"
        for row in df_show.itertuples(index=False)
    )
    note = f"<p class='hint'>{len(trades)} trades total, showing first {min(limit, len(trades))}.</p>" if len(trades) > limit else ""
    return f"<table><tr>{header}</tr>{rows}</table>{note}"


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
    <img src="data:image/png;base64,{leaderboard_chart(board)}">
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
      <fieldset><legend>Pair selection</legend><div class="field-grid">{pair_field_html}</div></fieldset>
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
    if "avg_n_held" in result:
        tiles.append(("Avg names held", result["avg_n_held"], ""))
    if "avg_n_active" in result:
        tiles.append(("Avg names active", result["avg_n_active"], ""))

    body += '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value {cls}">{value}</div></div>'
        for label, value, cls in tiles
    ) + "</div>"

    body += f"""<h2>Equity curve</h2><img src="data:image/png;base64,{equity_chart(result, conf['label'])}">
    <p class="hint">Base unit: {result['dollar_per_unit']:,.2f} TRY per abstract position unit
    (starting capital &divide; scale_max={result.get('scale_max', 1.0):g}). Not full per-leg share/lot accounting.</p>"""

    cost_rows = "".join(
        f"<tr class='{'neg' if row['sharpe (naive)'] <= 0 else ''}'><td>{bps}</td>"
        f"<td>{row['annualized return']*100:.1f}%</td><td>{row['sharpe (naive)']:.2f}</td></tr>"
        for bps, row in result["cost_table"].iterrows()
    )
    body += f"""<h2>Cost sensitivity</h2>
    <table><tr><th>bps/switch</th><th>Ann. return</th><th>Sharpe</th></tr>{cost_rows}</table>"""

    body += "<h2>Trade log</h2>" + render_trades_table(result["trades"])

    if "screening" in result:
        body += f"""<h2>Pair screening (in-sample, top 10)</h2>
        <table><tr><th>Pair</th><th>Beta</th><th>ADF stat</th><th>H</th><th>Half-life (d)</th><th>Qualifies</th></tr>{"".join(
            f"<tr><td>{r['pair']}</td><td>{r['beta']:.3f}</td><td>{r['adf_stat']:.3f}</td>"
            f"<td>{r['H']:.3f}</td><td>{r['half_life_days']:.1f}</td><td>{'checkmark' if r['qualifies'] else ''}</td></tr>"
            for _, r in result['screening'].head(10).iterrows()
        )}</table>"""

    return body


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
  img {{ max-width: 100%; border-radius: 6px; border: 1px solid var(--border); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-bottom: 12px; }}
  th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-2); font-weight: 500; }}
  tr.top-family {{ background: color-mix(in srgb, {AQUA} 10%, var(--surface)); }}
  .neg {{ color: {RED}; }}
  .pos {{ color: {AQUA}; }}
  .error {{ background: color-mix(in srgb, {RED} 15%, var(--surface)); border: 1px solid {RED}; border-radius: 8px; padding: 12px 16px; }}
"""


@app.route("/", methods=["GET"])
def index():
    tab = request.args.get("tab", default="leaderboard")

    if tab == "leaderboard":
        content = render_leaderboard_tab()
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
            else:
                result = conf["run"](prices, params=merged_params, bars_per_year=bpy)

            result = c.add_capital_pnl(result, starting_capital)
            params["result"] = result
        except Exception as e:
            params["error"] = str(e)

        content = render_explore_tab(params)

    page = f"""<title>BIST30 Strategy Lab</title>
<style>{PAGE_CSS}</style>
<h1>BIST30 Strategy Lab</h1>
<div class="subtitle">Statistical models, simulation-based validation, and technical strategy comparison
across the real BIST30 universe -- see WORKFLOW.md.</div>
<div class="tabs">
  <a href="?tab=leaderboard" class="{'active' if tab == 'leaderboard' else ''}">Leaderboard</a>
  <a href="?tab=explore" class="{'active' if tab == 'explore' else ''}">Explore</a>
</div>
{content}
"""
    return render_template_string(page)


if __name__ == "__main__":
    print("Starting BIST30 Strategy Lab at http://127.0.0.1:5100")
    print("(Ctrl+C to stop)")
    app.run(port=5100, debug=False)
