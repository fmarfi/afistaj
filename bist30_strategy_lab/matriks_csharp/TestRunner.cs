// Verification harness for Bist30Momentum.cs: runs the C# strategy on the
// SAME price panel the Python engine ran on and prints the same statistics,
// so the port can be checked number against number instead of by reading.
//
// Not part of the Matriks strategy -- exclude this file when pasting into
// the platform.
//
//   csc /out:test.exe Bist30Momentum.cs TestRunner.cs
//   test.exe panel.csv
//
// panel.csv is what momentum_standalone.py --save-prices writes: first
// column timestamps, one column per symbol.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace Bist30
{
    public class TestRunner
    {
        public static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("usage: test.exe <panel.csv> [topN]");
                return 2;
            }

            List<DateTime> stamps = new List<DateTime>();
            Dictionary<string, List<double>> columns = new Dictionary<string, List<double>>();
            List<string> header = null;

            using (StreamReader reader = new StreamReader(args[0]))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (line.Length == 0) continue;
                    string[] cells = line.Split(',');
                    if (header == null)
                    {
                        header = new List<string>();
                        for (int i = 1; i < cells.Length; i++)
                        {
                            header.Add(cells[i].Trim());
                            columns[cells[i].Trim()] = new List<double>();
                        }
                        continue;
                    }
                    stamps.Add(ParseStamp(cells[0]));
                    for (int i = 1; i < cells.Length; i++)
                    {
                        double value;
                        string cell = cells[i].Trim();
                        if (cell.Length == 0 ||
                            !double.TryParse(cell, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
                            value = double.NaN;
                        columns[header[i - 1]].Add(value);
                    }
                }
            }

            Dictionary<string, double[]> closes = new Dictionary<string, double[]>();
            for (int i = 0; i < header.Count; i++) closes[header[i]] = columns[header[i]].ToArray();

            MomentumParameters parameters = new MomentumParameters();
            if (args.Length > 1) parameters.TopN = int.Parse(args[1], CultureInfo.InvariantCulture);

            CrossSectionalMomentum strategy = new CrossSectionalMomentum(parameters);
            MomentumResult result = strategy.Run(stamps.ToArray(), closes);

            Console.WriteLine("bars              : " + stamps.Count + " x " + header.Count + " symbols");
            Console.WriteLine("bars per year     : " + result.BarsPerYear.ToString("F1", CultureInfo.InvariantCulture));
            Console.WriteLine("split bar         : " + result.SplitBar + "  (" +
                              stamps[result.SplitBar].ToString("yyyy-MM-dd HH:mm") + ")");
            Console.WriteLine("final capital     : " + result.FinalCapital.ToString("N2", CultureInfo.InvariantCulture));
            Console.WriteLine("total return %    : " + result.TotalReturnPercent.ToString("F4", CultureInfo.InvariantCulture));
            Console.WriteLine("annualized return : " + result.AnnualizedReturn.ToString("F4", CultureInfo.InvariantCulture));
            Console.WriteLine("annualized vol    : " + result.AnnualizedVolatility.ToString("F4", CultureInfo.InvariantCulture));
            Console.WriteLine("sharpe            : " + result.Sharpe.ToString("F3", CultureInfo.InvariantCulture));
            Console.WriteLine("max drawdown      : " + result.MaxDrawdown.ToString("F4", CultureInfo.InvariantCulture));
            Console.WriteLine("trades            : " + result.Trades.Count);

            result.Trades.Sort(delegate(MomentumTrade a, MomentumTrade b)
            {
                int byTime = a.EntryTime.CompareTo(b.EntryTime);
                return byTime != 0 ? byTime : string.CompareOrdinal(a.Symbol, b.Symbol);
            });
            Console.WriteLine();
            Console.WriteLine("first 5 trades:");
            for (int i = 0; i < Math.Min(5, result.Trades.Count); i++)
            {
                MomentumTrade t = result.Trades[i];
                Console.WriteLine("  " + t.Symbol.PadRight(10) +
                                  t.EntryTime.ToString("yyyy-MM-dd HH:mm") + " -> " +
                                  t.ExitTime.ToString("yyyy-MM-dd HH:mm") +
                                  "  hold " + t.HoldBars +
                                  "  amount " + t.AmountTraded.ToString("N0", CultureInfo.InvariantCulture) +
                                  "  pnl " + t.PnlCurrency.ToString("N0", CultureInfo.InvariantCulture));
            }
            return 0;
        }

        private static DateTime ParseStamp(string text)
        {
            text = text.Trim();
            // The panel carries an ISO stamp with a +03:00 offset; the bars
            // are Istanbul wall time and the strategy only needs the local
            // calendar day, so keep the wall clock and drop the offset.
            DateTimeOffset offset;
            if (DateTimeOffset.TryParse(text, CultureInfo.InvariantCulture,
                                        DateTimeStyles.None, out offset))
                return offset.DateTime;
            return DateTime.Parse(text, CultureInfo.InvariantCulture);
        }
    }
}
