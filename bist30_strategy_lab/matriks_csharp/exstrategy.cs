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
using Matriks.IntermediaryInstitutionAnalysis.Enums;
using Newtonsoft.Json;

//===========================================AÇIKLAMA===========================================//
// Bu stratejide Bollinger ve RSI indikatörleri birlikte kullanılmıştır. Aşırı alım/satım	    //
// bölgeleri tespit edilmeye çalışılmaktadır. Düşeceğini düşündüğünüz bir enstrüman için	    //
// (kod içerisindeki alım emri yorum satırına dönüştürülerek ya da silinerek) satış amaçlı,	    //
// yükseleceğini düşündüğünüz bir hisse için (satış emri silinerek) alım amaçlı	kullanılabilir.	//
// Eğer RSI(varsayılan periyot 11) oversold bandının üstüne kırarsa (varsayilan 30) 		    //
// ve son kapanış fiyatı BollingerDown(aşağı) bandından daha düşük bir değerse alım		        //
// emri gönderilir. Eğer RSI overbought (70) bandının altına kırarsa ve son kapanış fiyatı 	    //
// BollingerUp (yukarı) bandından daha yüksek bir değerse sat emri gönderilir.			        //
// Emirler piyasa fiyatından gönderilecektir.                                            	    //
// Emir gönderimi ile birlikte strateji raporunda Debug sekmesine "Alış emri gönderildi."	    //
// ve "Satış emri gönderildi." ifadesi yazdırılmaktadır.                                  	    //

namespace Matriks.Lean.Algotrader

{	//Strateji ismini burada deklare ediyoruz.
	//Dosya daki isimle stratejide yazılan isim tamamen aynı olmalıdır. (küçük büyük harf duyarlı)

	public class BolRsiStrategy : MatriksAlgo
	{
		//canlı, backtest ve backtest optimization kısımlarında değiştirilebilir olması 
		//istenilen parametreler bu bölümde yazılır

		[SymbolParameter("AKBNK")]
		public string Symbol;//Sembol ismi

		[Parameter(SymbolPeriod.Day)]
		public SymbolPeriod SymbolPeriod;//Stratejiyi çalıştırmak istediğimiz bar periyodu

		[Parameter(1)]
		public decimal BuyOrderQuantity;//Alım miktarı için kullanacağımız parametre

		[Parameter(1)]
		public decimal SellOrderQuantity;//Satım miktarı için kullanacağımız parametre

		[Parameter(11)] //RSI periyodu için kullanacağımız parametre
		public int RsiPeriod;

		[Parameter(MovMethod.E)]
		public MovMethod MovMethod;//BOLLINGER indikatörünü hesaplamak için kullanacağımız hareketli ortalama metodu parametre

		[Parameter(15)]
		public int BolPeriod; //BOLLINGER periyodu için kullanacağımız parametre

		[Parameter(2)]
		public decimal StandartDeviation;//BOLLINGER indikatörünü hesaplamak için kullanacağımız standart sapma değeri

		[Output]
		public decimal Kapanis;
		[Output]
		public decimal Bollinger_Ust_Bant;
		[Output]
		public decimal Bollinger_Alt_Bant;
		[Output]
		public decimal RSI;

		// indikator tanımları.
		BOLLINGER bollinger;//BOLLINGER indikatörü türünde bollinger isminde bir obje tanımlıyoruz
		RSI rsi;//RSI indikatörü türünde rsi isminde bir obje tanımlıyoruz
		 
		public override void OnInit()
		{
			rsi = RSIIndicator(Symbol, SymbolPeriod, OHLCType.Close, RsiPeriod);
			bollinger = BollingerIndicator(Symbol, SymbolPeriod, OHLCType.Close, BolPeriod, StandartDeviation, MovMethod);

			AddSymbol(Symbol, SymbolPeriod);

			// Algoritmanın kalıcı veya geçici sinyal ile çalışıp çalışmayacağını belirleyen fonksiyondur.
			// true geçerseniz algoritma sadece yeni bar açılışlarında çalışır, bu fonksiyonu çağırmazsanız 
			//veya false geçerseniz her işlem olduğunda algoritma tetiklenir.
			WorkWithPermanentSignal(true);

			//Eger emri bir al bir sat seklinde gonderilmesi isteniyor bu true set edilir. 
			//Alttaki satırı silerek veya false geçerek emirlerin sirayla gönderilmesini engelleyebilirsiniz. 
			//SendOrderSequential(true);
		}

		// Eklenen sembollerin bardata'ları ve indikatorler güncellendikçe bu fonksiyon tetiklenir. 
		public override void OnDataUpdate(BarDataEventArgs barData)
		{
			/*Debug("************");
			Debug("Close = " + barData.BarData.Close);
			Debug("bollinger ust bant = " + Math.Round(bollinger.Bollingerup.CurrentValue, 4));
			Debug("bollinger alt bant = " + Math.Round(bollinger.BollingerDown.CurrentValue, 4));
			Debug("rsi = " + Math.Round(rsi.CurrentValue, 2));*/
			if (CrossAbove(rsi, rsi.DownLevel) && bollinger.BollingerDown.CurrentValue > barData.BarData.Close)
			{
				SendMarketOrder(Symbol, BuyOrderQuantity, (OrderSide.Buy));
				Debug("Close = " + barData.BarData.Close);
				Debug("BOLLINGER DOWN = " + bollinger.BollingerDown.CurrentValue);
				Debug("rsi = " + Math.Round(rsi.CurrentValue, 2));
				Debug("Alış Emri Gönderildi");
			}
			if (CrossBelow(rsi, rsi.UpLevel) && bollinger.Bollingerup.CurrentValue < barData.BarData.Close)
			{
				SendMarketOrder(Symbol, SellOrderQuantity, (OrderSide.Sell));
				Debug("Close = " + barData.BarData.Close);
				Debug("BOLLINGER UP = " + Math.Round(bollinger.Bollingerup.CurrentValue, 4));
				Debug("rsi = " + Math.Round(rsi.CurrentValue, 2));
				Debug("Satış Emri Gönderildi");
			}

			Kapanis = barData.BarData.Close;
			Bollinger_Ust_Bant = Math.Round(bollinger.Bollingerup.CurrentValue, 2);
			Bollinger_Alt_Bant = Math.Round(bollinger.BollingerDown.CurrentValue, 2);
			RSI = Math.Round(rsi.CurrentValue, 2);
		}

		/// <summary>
		/// Strateji durdurulduğunda bu fonksiyon tetiklenir.
		/// </summary>
		public override void OnStopped()
		{
		}
    }
}