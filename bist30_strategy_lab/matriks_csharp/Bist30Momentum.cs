// BIST30 cross-sectional momentum -- the strategy logic, in C#.
//
// This is a port of momentum_standalone.py / _engine_momentum.py. It is
// deliberately PLATFORM-NEUTRAL: no Matriks types, no I/O, no orders, no
// framework base class. It takes a bar timeline plus each symbol's closes
// and returns what the strategy would have held, traded, and earned.
//
// Why the split: the platform adapter (which symbol data comes from where,
// how an order is sent) changes with the platform and with the version of
// its API, while THIS file is the part whose numbers have to match the
// research backtest bar for bar. It is verified against the Python engine
// on the same price panel -- see TestRunner.cs.
//
// IMPORTANT -- this strategy is CROSS-SECTIONAL. It does not look at one
// instrument and decide; it RANKS the whole universe every rebalance and
// holds the top N. A per-symbol backtester that only ever shows a strategy
// one instrument cannot express it: the rank of AKBNK is meaningless
// without the other 29 series at the same bar. The host must therefore
// feed every symbol's closes into Evaluate() before asking what to hold.
//
// Compiles as plain C# 5 (no external packages) so it can be pasted into
// an older strategy editor as-is.

using System;
using System.Collections.Generic;

namespace Bist30
{
    /// <summary>Tunables. Every window is counted in BARS, not days -- 180 hourly bars is ~20 sessions.</summary>
    public class MomentumParameters
    {
        public int LookbackBars = 180;      // trailing rate-of-change ranking window
        public int MaFastBars = 45;         // fast trend-filter moving average
        public int MaSlowBars = 180;        // slow trend-filter moving average
        public int RebalanceBars = 45;      // bars between re-rankings
        public int TopN = 5;                // names held AT ONCE (the basket rotates through more)
        public double InSampleFraction = 0.6;   // untraded warm-up, split on a session boundary
        public double StartingCapital = 100000.0;
    }

    /// <summary>One completed position in one symbol.</summary>
    public class MomentumTrade
    {
        public string Symbol;
        public DateTime EntryTime;
        public DateTime ExitTime;
        public int EntryBar;
        public int ExitBar;
        public int HoldBars;
        public string ExitReason;
        public double EntryPrice;
        public double ExitPrice;
        public double PnlFraction;      // attributed bar by bar over the hold
        public double PnlCurrency;
        public double AmountTraded;     // capital / names held at entry
        public long Shares;             // indicative: amount / entry price, no lot rounding
    }

    /// <summary>Everything one backtest produced.</summary>
    public class MomentumResult
    {
        public double[] BarPnlFraction;     // the book's return per bar
        public double[] EquityCurve;        // currency, sized off STARTING capital (never compounds)
        public int[] NamesHeld;
        public double[] Turnover;
        public List<MomentumTrade> Trades = new List<MomentumTrade>();
        public int SplitBar;                // first out-of-sample bar
        public double BarsPerYear;
        public double AnnualizedReturn;
        public double AnnualizedVolatility;
        public double Sharpe;
        public double MaxDrawdown;
        public double FinalCapital;
        public double TotalReturnPercent;
    }

    /// <summary>
    /// The strategy. Feed it the shared bar timeline and one close-price
    /// array per symbol (same length, double.NaN where that symbol had no
    /// bar), then call Run.
    /// </summary>
    public class CrossSectionalMomentum
    {
        private readonly MomentumParameters _p;

        public CrossSectionalMomentum(MomentumParameters parameters)
        {
            _p = parameters ?? new MomentumParameters();
        }

        /// <summary>Rate of change over LookbackBars, NaN until there is enough history.</summary>
        public static double[] RateOfChange(double[] close, int lookback)
        {
            double[] roc = new double[close.Length];
            for (int t = 0; t < close.Length; t++)
            {
                int back = t - lookback;
                if (back < 0 || IsBad(close[t]) || IsBad(close[back]) || close[back] == 0.0)
                    roc[t] = double.NaN;
                else
                    roc[t] = close[t] / close[back] - 1.0;
            }
            return roc;
        }

        /// <summary>
        /// Trailing simple moving average. NaN until the window is full AND
        /// only over windows with no gaps -- the same rule pandas' rolling
        /// mean applies, so a symbol that was not trading yet cannot sneak a
        /// short-window average into the ranking.
        /// </summary>
        public static double[] MovingAverage(double[] close, int window)
        {
            double[] ma = new double[close.Length];
            for (int t = 0; t < close.Length; t++)
            {
                if (t + 1 < window) { ma[t] = double.NaN; continue; }
                double sum = 0.0;
                bool ok = true;
                for (int k = t - window + 1; k <= t; k++)
                {
                    if (IsBad(close[k])) { ok = false; break; }
                    sum += close[k];
                }
                ma[t] = ok ? sum / window : double.NaN;
            }
            return ma;
        }

        /// <summary>
        /// The names to hold after a rebalance: ranked by trailing return,
        /// keeping only those the trend filter confirms. `bar` must be the
        /// bar the decision is ALLOWED to see -- the caller passes the
        /// PREVIOUS bar, never the one being traded on.
        /// </summary>
        public static List<string> RankUniverse(IList<string> symbols,
                                                Dictionary<string, double[]> roc,
                                                Dictionary<string, double[]> maFast,
                                                Dictionary<string, double[]> maSlow,
                                                int bar, int topN)
        {
            List<string> eligible = new List<string>();
            List<double> scores = new List<double>();
            for (int i = 0; i < symbols.Count; i++)
            {
                string symbol = symbols[i];
                double score = roc[symbol][bar];
                double fast = maFast[symbol][bar];
                double slow = maSlow[symbol][bar];
                bool trending = !IsBad(fast) && !IsBad(slow) && fast > slow;
                if (IsBad(score) || !trending)
                    continue;
                eligible.Add(symbol);
                scores.Add(score);
            }

            // Stable descending sort: equal scores keep universe order, so a
            // tie resolves the same way it does in the Python engine.
            int[] order = new int[eligible.Count];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            Array.Sort(order, delegate(int a, int b)
            {
                int byScore = scores[b].CompareTo(scores[a]);
                return byScore != 0 ? byScore : a.CompareTo(b);
            });

            List<string> selected = new List<string>();
            for (int i = 0; i < order.Length && selected.Count < topN; i++)
                selected.Add(eligible[order[i]]);
            return selected;
        }

        /// <summary>
        /// First bar of the out-of-sample window, landing on a session
        /// boundary rather than mid-day, so a warm-up never ends halfway
        /// through a trading session.
        /// </summary>
        public static int SplitOnSessionBoundary(DateTime[] timestamps, double inSampleFraction)
        {
            int[] session = SessionIds(timestamps);
            int sessionCount = session[session.Length - 1] + 1;
            int splitSession = (int)(sessionCount * inSampleFraction);
            for (int t = 0; t < session.Length; t++)
                if (session[t] >= splitSession)
                    return t;
            return session.Length;
        }

        public static int[] SessionIds(DateTime[] timestamps)
        {
            int[] session = new int[timestamps.Length];
            int current = 0;
            for (int t = 1; t < timestamps.Length; t++)
            {
                if (timestamps[t].Date != timestamps[t - 1].Date) current++;
                session[t] = current;
            }
            return session;
        }

        /// <summary>Real bar density, measured from the data rather than assumed (hourly BIST ~2143, daily 252).</summary>
        public static double BarsPerYear(DateTime[] timestamps)
        {
            if (timestamps.Length < 2) return double.NaN;
            Dictionary<DateTime, bool> days = new Dictionary<DateTime, bool>();
            for (int t = 0; t < timestamps.Length; t++) days[timestamps[t].Date] = true;
            if (days.Count == 0) return double.NaN;
            return ((double)timestamps.Length / days.Count) * 252.0;
        }

        /// <summary>
        /// The whole backtest. Mirrors walk_forward_momentum: rebalance every
        /// RebalanceBars using ONLY the previous bar's data, hold equally
        /// weighted, attribute each trade's PnL bar by bar over its own hold.
        /// </summary>
        public MomentumResult Run(DateTime[] timestamps, Dictionary<string, double[]> closes)
        {
            List<string> symbols = new List<string>(closes.Keys);
            symbols.Sort(StringComparer.Ordinal);   // deterministic universe order for tie-breaks
            int n = timestamps.Length;

            Dictionary<string, double[]> logReturn = new Dictionary<string, double[]>();
            Dictionary<string, double[]> roc = new Dictionary<string, double[]>();
            Dictionary<string, double[]> maFast = new Dictionary<string, double[]>();
            Dictionary<string, double[]> maSlow = new Dictionary<string, double[]>();
            Dictionary<string, double[]> contribution = new Dictionary<string, double[]>();
            for (int i = 0; i < symbols.Count; i++)
            {
                string s = symbols[i];
                double[] close = closes[s];
                if (close.Length != n)
                    throw new ArgumentException("close series for " + s + " is not aligned to the bar timeline");
                logReturn[s] = LogReturns(close);
                roc[s] = RateOfChange(close, _p.LookbackBars);
                maFast[s] = MovingAverage(close, _p.MaFastBars);
                maSlow[s] = MovingAverage(close, _p.MaSlowBars);
                contribution[s] = new double[n];
            }

            MomentumResult result = new MomentumResult();
            result.BarPnlFraction = new double[n];
            result.NamesHeld = new int[n];
            result.Turnover = new double[n];
            result.SplitBar = SplitOnSessionBoundary(timestamps, _p.InSampleFraction);
            result.BarsPerYear = BarsPerYear(timestamps);

            Dictionary<string, int> holdings = new Dictionary<string, int>();   // symbol -> entry bar
            List<string> holdingOrder = new List<string>();                     // stable iteration
            List<MomentumTrade> open = new List<MomentumTrade>();
            int lastRebalanceBar = -1;

            for (int t = result.SplitBar; t < n; t++)
            {
                bool rebalance = (lastRebalanceBar < 0) || (t - lastRebalanceBar >= _p.RebalanceBars);
                if (rebalance)
                {
                    lastRebalanceBar = t;
                    // t-1, never t: the bar being traded on has not closed
                    // when the decision is made.
                    List<string> selected = RankUniverse(symbols, roc, maFast, maSlow, t - 1, _p.TopN);

                    List<string> dropped = new List<string>();
                    for (int i = 0; i < holdingOrder.Count; i++)
                        if (!selected.Contains(holdingOrder[i])) dropped.Add(holdingOrder[i]);
                    List<string> added = new List<string>();
                    for (int i = 0; i < selected.Count; i++)
                        if (!holdings.ContainsKey(selected[i])) added.Add(selected[i]);

                    result.Turnover[t] = (double)(dropped.Count + added.Count) / Math.Max(_p.TopN, 1);

                    for (int i = 0; i < dropped.Count; i++)
                    {
                        string symbol = dropped[i];
                        result.Trades.Add(CloseTrade(symbol, holdings[symbol], t, timestamps,
                                                     closes[symbol], "rebalance drop"));
                        holdings.Remove(symbol);
                        holdingOrder.Remove(symbol);
                    }
                    for (int i = 0; i < added.Count; i++)
                    {
                        holdings[added[i]] = t;
                        holdingOrder.Add(added[i]);
                    }
                }

                if (holdingOrder.Count > 0)
                {
                    double weight = 1.0 / holdingOrder.Count;
                    for (int i = 0; i < holdingOrder.Count; i++)
                    {
                        string symbol = holdingOrder[i];
                        double r = logReturn[symbol][t];
                        if (!IsBad(r))
                        {
                            contribution[symbol][t] = weight * r;
                            result.BarPnlFraction[t] += weight * r;
                        }
                    }
                }
                result.NamesHeld[t] = holdingOrder.Count;
            }

            for (int i = 0; i < holdingOrder.Count; i++)
            {
                string symbol = holdingOrder[i];
                result.Trades.Add(CloseTrade(symbol, holdings[symbol], n - 1, timestamps,
                                             closes[symbol], "end of backtest"));
            }

            // Per-trade PnL: this symbol's own contribution summed over
            // exactly its holding period, NOT a fixed stake times a price
            // move -- the basket's weight drifts as names come and go.
            for (int i = 0; i < result.Trades.Count; i++)
            {
                MomentumTrade trade = result.Trades[i];
                double[] contrib = contribution[trade.Symbol];
                double sum = 0.0;
                for (int t = trade.EntryBar + 1; t <= trade.ExitBar && t < n; t++) sum += contrib[t];
                trade.PnlFraction = sum;
            }

            AttachCapital(result);
            ComputeStatistics(result);
            return result;
        }

        private MomentumTrade CloseTrade(string symbol, int entryBar, int exitBar,
                                         DateTime[] timestamps, double[] close, string reason)
        {
            MomentumTrade trade = new MomentumTrade();
            trade.Symbol = symbol;
            trade.EntryBar = entryBar;
            trade.ExitBar = exitBar;
            trade.EntryTime = timestamps[entryBar];
            trade.ExitTime = timestamps[exitBar];
            trade.HoldBars = exitBar - entryBar;
            trade.ExitReason = reason;
            trade.EntryPrice = close[entryBar];
            trade.ExitPrice = close[exitBar];
            return trade;
        }

        /// <summary>
        /// Currency sizing. The book is never more than fully invested, so
        /// the base unit is the whole starting capital and each held name
        /// gets 1/n of it -- TopN=5 means 20,000 of 100,000 per name, not
        /// 100,000. Sizing is off STARTING capital and never compounds.
        /// </summary>
        private void AttachCapital(MomentumResult result)
        {
            double capital = _p.StartingCapital;
            int n = result.BarPnlFraction.Length;
            result.EquityCurve = new double[n];
            double running = capital;
            for (int t = 0; t < n; t++)
            {
                running += result.BarPnlFraction[t] * capital;
                result.EquityCurve[t] = running;
            }
            result.FinalCapital = n > 0 ? result.EquityCurve[n - 1] : capital;
            result.TotalReturnPercent = capital > 0 ? (result.FinalCapital / capital - 1.0) * 100.0 : 0.0;

            for (int i = 0; i < result.Trades.Count; i++)
            {
                MomentumTrade trade = result.Trades[i];
                int namesHeld = result.NamesHeld[trade.EntryBar];
                trade.AmountTraded = capital / Math.Max(namesHeld, 1);
                trade.PnlCurrency = trade.PnlFraction * capital;
                trade.Shares = (!IsBad(trade.EntryPrice) && trade.EntryPrice > 0)
                    ? (long)(trade.AmountTraded / trade.EntryPrice) : 0;
            }
        }

        /// <summary>Out-of-sample only -- the in-sample warm-up is not traded and must not flatter the statistics.</summary>
        private void ComputeStatistics(MomentumResult result)
        {
            int start = result.SplitBar;
            int count = result.BarPnlFraction.Length - start;
            if (count <= 0) return;

            double mean = 0.0;
            for (int t = start; t < result.BarPnlFraction.Length; t++) mean += result.BarPnlFraction[t];
            mean /= count;

            double variance = 0.0;
            for (int t = start; t < result.BarPnlFraction.Length; t++)
            {
                double d = result.BarPnlFraction[t] - mean;
                variance += d * d;
            }
            variance /= count;                     // population sigma, matching numpy's default
            double sigma = Math.Sqrt(variance);

            result.AnnualizedReturn = mean * result.BarsPerYear;
            result.AnnualizedVolatility = sigma * Math.Sqrt(result.BarsPerYear);
            result.Sharpe = result.AnnualizedVolatility > 0
                ? result.AnnualizedReturn / result.AnnualizedVolatility : double.NaN;

            double cumulative = 0.0, peak = 0.0, worst = 0.0;
            for (int t = start; t < result.BarPnlFraction.Length; t++)
            {
                cumulative += result.BarPnlFraction[t];
                if (cumulative > peak) peak = cumulative;
                double drawdown = peak - cumulative;
                if (drawdown > worst) worst = drawdown;
            }
            result.MaxDrawdown = worst;
        }

        private static double[] LogReturns(double[] close)
        {
            double[] r = new double[close.Length];
            r[0] = double.NaN;
            for (int t = 1; t < close.Length; t++)
            {
                if (IsBad(close[t]) || IsBad(close[t - 1]) || close[t - 1] <= 0 || close[t] <= 0)
                    r[t] = double.NaN;
                else
                    r[t] = Math.Log(close[t] / close[t - 1]);
            }
            return r;
        }

        private static bool IsBad(double value)
        {
            return double.IsNaN(value) || double.IsInfinity(value);
        }
    }
}
