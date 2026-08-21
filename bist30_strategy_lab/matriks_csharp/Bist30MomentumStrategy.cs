using System;
using System.Collections.Generic;
using System.Linq;
using Matriks.Data.Symbol;
using Matriks.Engines;
using Matriks.Indicators;
using Matriks.Symbols;
using Matriks.AlgoTrader;
using Matriks.Trader.Core;
using Matriks.Trader.Core.Fields;
using Matriks.Trader.Core.TraderModels;
using Matriks.Lean.Algotrader.AlgoBase;
using Matriks.Lean.Algotrader.Models;
using Matriks.Lean.Algotrader.Trading;
using Matriks.Data.Tick;
using Matriks.Enumeration;

//===========================================ACIKLAMA===========================================//
// BIST30 KESITSEL MOMENTUM (cross-sectional momentum) -- long only, top-N rotasyon.            //
//                                                                                              //
// Bu strateji TEK bir hisseye bakip karar vermez: her rebalans barinda TUM listeyi siralar     //
// ve en guclu N tanesini esit agirlikla tutar. Bu yuzden AddSymbol ile 30 sembolun tamami      //
// eklenir ve karar, referans sembolun bari kapandiginda bir kez verilir.                       //
//                                                                                              //
// Siralama olcusu : Lookback bar onceye gore getiri (ROC).                                     //
// Trend filtresi  : hizli hareketli ortalama > yavas hareketli ortalama (whipsaw korumasi).    //
// Karar ani       : HER ZAMAN bir onceki barin verisiyle (index-1). Islem yapilan barin        //
//                   kapanisi karara girmez -- ileriye bakma (look-ahead) yoktur.               //
//                                                                                              //
// Bu dosya KENDI KENDINE YETERLIDIR: siralama matematigi iceride, baska dosyaya bagli degil.   //
// Ayni matematik Python arastirma motoruyla (momentum_standalone.py) kurus kurusuna            //
// dogrulanmistir -- bkz. matriks_csharp/Bist30Momentum.cs ve TestRunner.cs.                    //
//                                                                                              //
// PARAMETRELER: arastirma panosunun Explore sekmesindeki Momentum alanlariyla birebir ayni     //
// (ayni isim, ayni birim = BAR, ayni varsayilan, ayni aralik). Hepsi backtest, optimizasyon    //
// ve canli calismada UI'dan degistirilebilir.                                                  //
//==============================================================================================//

namespace Matriks.Lean.Algotrader
{
	// Sinif adi dosya adiyla birebir ayni olmalidir (buyuk/kucuk harf duyarli).
	public class Bist30MomentumStrategy : MatriksAlgo
	{
		// ---------------------------------------------------------------- parametreler

		[Parameter("AEFES,AKBNK,ASELS,ASTOR,BIMAS,DSTKF,EKGYO,ENKAI,EREGL,FROTO,GARAN,GUBRF,ISCTR,KCHOL,KRDMD,MGROS,PETKM,PGSUS,SAHOL,SASA,SISE,TAVHL,TCELL,THYAO,TOASO,TRALT,TTKOM,TUPRS,VAKBN,YKBNK")]
		public string SymbolList;   // virgulle ayrilmis evren; ilk sembol ayni zamanda referans semboldur

		[Parameter(SymbolPeriod.Day)]
		public SymbolPeriod SymbolPeriod;   // arastirma 60 dakikalik barda yapildi -- UI'dan secin

		// Parametreler ve varsayilanlari, arastirma panosunun (04_interactive_
		// dashboard.py, Explore sekmesi) Momentum alanlariyla BIREBIR aynidir:
		// ayni isimler, ayni birim (BAR), ayni varsayilanlar, ayni gecerli
		// araliklar. Panoda ne gorup ayarliyorsaniz burada da odur.
		//
		//   ROC lookback (bars)    20..500   varsayilan 180
		//   Fast MA (bars)          5..200   varsayilan  45
		//   Slow MA (bars)         20..500   varsayilan 180
		//   Rebalance every (bars)  5..200   varsayilan  45
		//   Basket size (top N)     1..15    varsayilan   5
		//
		// Bar sayilaridir, gun degil: 60 dakikalik BIST barinda seans basina
		// ~9 bar vardir, yani 180 bar ~ 20 seans. SymbolPeriod'u Gunluk
		// secerseniz ayni sayilar 180 GUN demeye gelir -- o durumda panoda da
		// yapacaginiz gibi degerleri kendiniz kucultun (20 / 5 / 20 / 5).
		[Parameter(180)]
		public int LookbackMom;     // ROC lookback (bars)

		[Parameter(45)]
		public int MaFast;          // Fast MA (bars)

		[Parameter(180)]
		public int MaSlow;          // Slow MA (bars)

		[Parameter(45)]
		public int Rebalance;       // Rebalance every (bars)

		[Parameter(5)]
		public int TopN;            // AYNI ANDA tutulan isim sayisi (sepet doner, daha fazla isme dokunur)

		[Parameter(100000)]
		public decimal Capital;     // sepetin tamamina ayrilan tutar; her isme Capital/TopN duser

		[Parameter(true)]
		public bool VerboseDebug;   // her rebalansta secilenleri Debug sekmesine yaz

		// ---------------------------------------------------------------- ciktilar

		[Output]
		public decimal HeldCount;   // su an tutulan isim sayisi

		[Output]
		public decimal LastRebalanceBar;

		// ---------------------------------------------------------------- ic durum

		private List<string> _symbols;                  // evren, verilen sirada (esitlik bozma sirasi)
		private Dictionary<string, int> _symbolIds;     // sembol -> SymbolId
		private Dictionary<string, decimal> _positions; // sembol -> elde tutulan adet
		private int _referenceSymbolId;
		private int _lastRebalanceIndex = -1;

		// Parametrelerin dogrulanmis hali -- OnInit'te bir kez cozulur, boylece
		// backtest ile canli calisma ayni sayilari kullanir.
		private int _lookbackBars;
		private int _maFastBars;
		private int _maSlowBars;
		private int _rebalanceBars;
		private int _requiredBars;

		public override void OnInit()
		{
			_symbols = ParseSymbolList(SymbolList);
			_symbolIds = new Dictionary<string, int>();
			_positions = new Dictionary<string, decimal>();

			// Panodaki gibi dogrudan bar cinsinden; tek yaptigimiz sifir/negatif
			// bir degerin siralamayi bozmasini engellemek.
			_lookbackBars = Math.Max(1, LookbackMom);
			_maFastBars = Math.Max(1, MaFast);
			_maSlowBars = Math.Max(1, MaSlow);
			_rebalanceBars = Math.Max(1, Rebalance);
			_requiredBars = Math.Max(_lookbackBars, Math.Max(_maFastBars, _maSlowBars));

			// Kesitsel bir strateji tum evreni ayni anda gormek zorundadir:
			// AKBNK'in sirasi, digerlerinin ayni bardaki getirisi bilinmeden
			// anlamsizdir. Bu yuzden 30 sembolun tamami eklenir.
			for (int i = 0; i < _symbols.Count; i++)
			{
				AddSymbol(_symbols[i], SymbolPeriod);
				_symbolIds[_symbols[i]] = GetSymbolId(_symbols[i]);
			}

			// Listenin ilk sembolu "referans"tir: OnDataUpdate her sembol icin
			// ayri ayri tetiklenir, ama karar bar basina BIR KEZ verilmelidir.
			_referenceSymbolId = _symbolIds[_symbols[0]];

			// Sadece yeni bar kapanislarinda calis -- bar ici her tikta degil.
			WorkWithPermanentSignal(true);

			Debug("BIST30 momentum hazir: " + _symbols.Count + " sembol | lookback " + _lookbackBars +
				  " bar, MA " + _maFastBars + "/" + _maSlowBars + " bar, rebalans " + _rebalanceBars +
				  " bar, TopN " + TopN + " (isinma: " + _requiredBars + " bar).");
		}

		public override void OnDataUpdate(BarDataEventArgs barData)
		{
			// Her sembol icin ayri tetikleniyoruz; karari yalnizca referans
			// sembolun barinda ver, yoksa ayni bar 30 kez rebalans edilir.
			if (barData.SymbolId != _referenceSymbolId)
				return;

			int index = barData.BarDataIndex;

			// Karar bir ONCEKI barin verisiyle alinir. Islem yapilan barin
			// kapanisi karara giremez -- arastirma motorundaki kural da budur.
			int signalIndex = index - 1;
			if (signalIndex < 1)
				return;

			// Butun pencereler dolmadan siralama yapma.
			if (signalIndex < _requiredBars)
				return;

			bool timeToRebalance = (_lastRebalanceIndex < 0) ||
								   (index - _lastRebalanceIndex >= _rebalanceBars);
			if (!timeToRebalance)
				return;

			_lastRebalanceIndex = index;
			LastRebalanceBar = index;

			List<string> selected = RankUniverse(signalIndex);
			RebalancePortfolio(selected, index);

			HeldCount = _positions.Count;
		}

		/// <summary>
		/// Evreni siralar: trend filtresini gecen ve yeterli gecmisi olan
		/// isimler icinde ROC'u en yuksek TopN tanesi. Esitlikte evren
		/// sirasi korunur, boylece ayni veri ayni sepeti verir.
		/// </summary>
		private List<string> RankUniverse(int signalIndex)
		{
			List<string> eligible = new List<string>();
			List<double> scores = new List<double>();

			for (int i = 0; i < _symbols.Count; i++)
			{
				string symbol = _symbols[i];
				double[] closes = ReadCloses(symbol, signalIndex);
				if (closes == null)
					continue;

				double roc = RateOfChange(closes, _lookbackBars);
				double fast = MovingAverage(closes, _maFastBars);
				double slow = MovingAverage(closes, _maSlowBars);

				bool trending = IsUsable(fast) && IsUsable(slow) && fast > slow;
				if (!IsUsable(roc) || !trending)
					continue;

				eligible.Add(symbol);
				scores.Add(roc);
			}

			int[] order = new int[eligible.Count];
			for (int i = 0; i < order.Length; i++) order[i] = i;
			Array.Sort(order, delegate(int a, int b)
			{
				int byScore = scores[b].CompareTo(scores[a]);
				return byScore != 0 ? byScore : a.CompareTo(b);   // stabil: esitlikte evren sirasi
			});

			List<string> selected = new List<string>();
			for (int i = 0; i < order.Length && selected.Count < TopN; i++)
				selected.Add(eligible[order[i]]);
			return selected;
		}

		/// <summary>
		/// Sepeti hedefe getirir: dusenleri sat, yeni girenleri al, kalanlari
		/// esit agirliga yeniden boyutlandir. Arastirma motoru her bar 1/n
		/// esit agirlik varsayar; emir seviyesinde bunun karsiligi, her
		/// rebalansta pozisyonlari esitlemektir.
		/// </summary>
		private void RebalancePortfolio(List<string> selected, int index)
		{
			decimal amountPerName = (selected.Count > 0) ? Capital / selected.Count : 0m;

			// 1) Artik secilmeyenleri tamamen kapat.
			List<string> current = new List<string>(_positions.Keys);
			for (int i = 0; i < current.Count; i++)
			{
				string symbol = current[i];
				if (selected.Contains(symbol))
					continue;
				decimal quantity = _positions[symbol];
				if (quantity > 0)
				{
					SendMarketOrder(symbol, quantity, OrderSide.Sell);
					if (VerboseDebug) Debug("CIKIS  " + symbol + "  adet " + quantity);
				}
				_positions.Remove(symbol);
			}

			// 2) Secilenleri hedef adede getir.
			for (int i = 0; i < selected.Count; i++)
			{
				string symbol = selected[i];
				double price = LastClose(symbol, index);
				if (!IsUsable(price) || price <= 0)
					continue;

				decimal target = Math.Floor(amountPerName / (decimal)price);   // BIST tam adet isler
				decimal held = _positions.ContainsKey(symbol) ? _positions[symbol] : 0m;
				decimal delta = target - held;

				if (delta > 0)
				{
					SendMarketOrder(symbol, delta, OrderSide.Buy);
					if (VerboseDebug) Debug("GIRIS  " + symbol + "  adet " + delta +
											"  tutar " + Math.Round(delta * (decimal)price, 0));
				}
				else if (delta < 0)
				{
					SendMarketOrder(symbol, -delta, OrderSide.Sell);
					if (VerboseDebug) Debug("AZALT  " + symbol + "  adet " + (-delta));
				}

				if (target > 0)
					_positions[symbol] = target;
				else
					_positions.Remove(symbol);
			}

			if (VerboseDebug)
				Debug("Rebalans @bar " + index + " -> " + string.Join(", ", selected.ToArray()));
		}

		// ---------------------------------------------------------------- veri erisimi

		/// <summary>
		/// Bir sembolun siralama icin gereken son kapanislari, en eskiden
		/// yeniye. Yetersiz gecmis varsa (yeni halka arz, isim degisikligi)
		/// null doner ve o isim o rebalansta siralamaya hic girmez -- ayni
		/// davranis arastirma motorunda da vardir.
		/// </summary>
		private double[] ReadCloses(string symbol, int signalIndex)
		{
			int needed = _requiredBars + 1;
			int start = signalIndex - needed + 1;
			if (start < 0)
				return null;

			var bars = GetBarData(symbol, SymbolPeriod);
			if (bars == null)
				return null;

			double[] closes = new double[needed];
			for (int i = 0; i < needed; i++)
			{
				double value = ToDouble(bars.Close[start + i]);
				if (!IsUsable(value) || value <= 0)
					return null;        // penceresinde bosluk olan isim siralanamaz
				closes[i] = value;
			}
			return closes;
		}

		private double LastClose(string symbol, int index)
		{
			var bars = GetBarData(symbol, SymbolPeriod);
			if (bars == null)
				return double.NaN;
			return ToDouble(bars.Close[index]);
		}

		// ---------------------------------------------------------------- matematik
		// (momentum_standalone.py ile birebir ayni tanimlar)

		/// <summary>Pencerenin sonundaki getiri: son kapanis / lookback bar onceki kapanis - 1.</summary>
		private static double RateOfChange(double[] closes, int lookback)
		{
			int last = closes.Length - 1;
			int back = last - lookback;
			if (back < 0 || closes[back] == 0)
				return double.NaN;
			return closes[last] / closes[back] - 1.0;
		}

		/// <summary>Son `window` barin duz ortalamasi; pencere dolmamissa NaN.</summary>
		private static double MovingAverage(double[] closes, int window)
		{
			if (window <= 0 || closes.Length < window)
				return double.NaN;
			double sum = 0.0;
			for (int i = closes.Length - window; i < closes.Length; i++)
				sum += closes[i];
			return sum / window;
		}

		private static bool IsUsable(double value)
		{
			return !double.IsNaN(value) && !double.IsInfinity(value);
		}

		private static double ToDouble(object value)
		{
			// GetBarData'nin dizileri decimal de olabilir double da; ikisini de kabul et.
			try { return Convert.ToDouble(value); }
			catch { return double.NaN; }
		}

		private static List<string> ParseSymbolList(string list)
		{
			List<string> symbols = new List<string>();
			if (string.IsNullOrEmpty(list))
				return symbols;
			string[] parts = list.Split(new char[] { ',', ';', ' ' });
			for (int i = 0; i < parts.Length; i++)
			{
				string symbol = parts[i].Trim().ToUpperInvariant();
				if (symbol.Length > 0 && !symbols.Contains(symbol))
					symbols.Add(symbol);
			}
			return symbols;
		}

		public override void OnStopped()
		{
		}
	}
}
