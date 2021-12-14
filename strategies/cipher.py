import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.series import Series
from functools import reduce
import pandas_ta as pta
import technical.indicators as technicali

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

rangeUpper = 60
rangeLower = 5
def valuewhen(dataframe, condition, source, occurrence):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    copy = copy.sort_values(by=[condition, 'colFromIndex'], ascending=False).reset_index(drop=True)
    copy['valuewhen'] = np.where(copy[condition] > 0, copy[source].shift(-occurrence), 100)
    copy['valuewhen'] = copy['valuewhen'].fillna(100)
    copy['barrsince'] = copy['colFromIndex'] - copy['colFromIndex'].shift(-occurrence)
    copy.loc[
        (
            (rangeLower <= copy['barrsince']) &
            (copy['barrsince']  <= rangeUpper)
        )
    , "in_range"] = 1
    copy['in_range'] = copy['in_range'].fillna(0)
    copy = copy.sort_values(by=['colFromIndex'], ascending=True).reset_index(drop=True)
    return copy['valuewhen'], copy['in_range']

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()   
    ema1 = ta.EMA(df, timeperiod=5)
    ema2 = ta.EMA(df, timeperiod=35)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def wavetrend(df:DataFrame) -> Series:

    # inputs
    n1 = 9  # Channel Length
    n2 = 12  # Average Length
    ap = (df['high'] + df['low'] + df['close']) / 3  # HLC3

    # wavetrend calculation
    esa = ta.EMA(ap, timeperiod=3)
    d = ta.EMA(abs(ap - esa), timeperiod=9)
    ci = (ap - esa) / (0.015 * d)
    wt1 = ta.EMA(ci, timeperiod=n2)
    wt2 = ta.SMA(wt1, timeperiod=3)
    wtVWAP = wt1 - wt2

    return pd.Series(data={
        "wt1": wt1,
        "wt2": wt2,
        "wtVWAP": wtVWAP
        })

def hlc3(df):
    return (df['high'] + df['low'] + df['close']) / 3

def fractalize(osrc):
    fcopy = osrc.copy()
    top_fractal = fcopy.shift(4) < fcopy.shift (2) & fcopy.shift(3) < fcopy.shift(2) & fcopy.shift(2) > fcopy.shift(1) & fcopy.shift(2) > fcopy
    bot_fractal = fcopy.shift(4) > fcopy.shift (2) & fcopy.shift(3) > fcopy.shift(2) & fcopy.shift(2) < fcopy.shift(1) & fcopy.shift(2) < fcopy
   
class cipher(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # # ROI table:
    '''minimal_roi = {
        "0": 0.438,
        "86": 0.162,
        "241": 0.061,
        "435": 0
    }'''
    buy_params = {
        "buy_aroondown": 34,
        "buy_aroonup": 82,
        "buy_arrondown_cat": False,
        "buy_arronup_cat": False,
        "buy_gold_cat1": False,
        "buy_gold_cat2": False,
        "buy_mfi": 17,
        "buy_mfi_cat": True,
        "buy_schaff_cat": False,
        "buy_schafff_val": 33,
        "buy_vwap_cat": False,
        "buy_wt_cat": True,
        "buy_wt_oversold": -64,
        "ewo_high": 0.453,
        "fast_ewo": 48,
        "rsi_buy": 37,
        "slow_ewo": 191,
        "use_bull": False,
        "use_hidden_bull": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_aroondown": 56,
        "sell_aroonup": 35,
        "sell_arrondown_cat": True,
        "sell_arronup_cat": False,
        "sell_mfi": 60,
        "sell_mfi_cat": False,
        "sell_schaff_cat": False,
        "sell_schafff_val": 72,
        "sell_vwap_cat": False,
        "sell_wt_cat": True,
        "sell_wt_overbought": 40,
        "use_bear": True,
        "use_hidden_bear": False,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.281,
        "75": 0.127,
        "195": 0.049,
        "460": 0
    }
    
    stoploss = -0.258

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.073
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    use_custom_stoploss = False

    # Optimal timeframe for the strategy.
    timeframe = '15m'
    #inf_1h = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 30
    oslevel3  = -100

    period = 10
    #buyval
    buy_mfi = IntParameter(10, 90, default = 50, space = 'buy' )
    fast_ewo = IntParameter(35, 75, default = 50, space = 'buy' )
    slow_ewo = IntParameter(130, 210, default = 200, space = 'buy' )
    buy_wt_oversold = IntParameter(-65, -30, default = -53, space = 'buy' )
    buy_aroonup = IntParameter(40, 100, default = 77, space = 'buy')
    buy_aroondown = IntParameter(0, 40, default = 24, space = 'buy' )
    buy_schafff_val = IntParameter(20, 40, default = 25, space = 'buy' )

    #sell val
    sell_mfi = IntParameter(10, 90, default = 50, space = 'sell' )
    sell_wt_overbought = IntParameter(30, 65, default = 53, space = 'sell' )
    sell_aroonup = IntParameter(0, 75, default = 24, space = 'sell' )
    sell_aroondown = IntParameter(40, 100, default = 68, space = 'sell')
    sell_schafff_val = IntParameter(65, 90, default = 75, space = 'sell' )




    #buy cat
    buy_mfi_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_vwap_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_arrondown_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_arronup_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_wt_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_schaff_cat = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_gold_cat1 = CategoricalParameter([True, False], default = True, space = 'buy' )
    buy_gold_cat2 = CategoricalParameter([True, False], default = True, space = 'buy' )
    #sell cat
    sell_mfi_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_wt_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_vwap_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_arronup_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_arrondown_cat = CategoricalParameter([True, False], default = True, space = 'sell' )
    sell_schaff_cat = CategoricalParameter([True, False], default = True, space = 'sell' )

    #RSIDIV
    
    use_bull = CategoricalParameter([True, False], default = True, space = 'buy' )
    use_hidden_bull = CategoricalParameter([True, False], default = True, space = 'buy' )
    use_bear = CategoricalParameter([True, False], default = True, space = 'sell' )
    use_hidden_bear = CategoricalParameter([True, False], default = True, space = 'sell' )
    ewo_high = DecimalParameter(0, 7.0, default=5.835, space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=50, space='buy', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #MFI

        

        dataframe['mfi'] = ta.MFI(dataframe, timeperiod = 60)

        #VWAP
        
        dataframe['vwap'] = qtpylib.rolling_vwap(dataframe)
        #wavtrend
        wavetrendi = wavetrend(dataframe)
        dataframe['wt1'] = wavetrendi['wt1']
        dataframe['wt2'] = wavetrendi['wt2']
        dataframe['wtVWAP'] = wavetrendi['wtVWAP']
        
        wbl = 60
        dataframe['oscwt'] = dataframe['wt2'].fillna(0)

        dataframe['wtmin'] = dataframe['oscwt'].rolling(wbl).min()
        dataframe['wtprevmin'] = np.where(dataframe['wtmin'] > dataframe['wtmin'].shift(), dataframe['wtmin'].shift(), dataframe['wtmin'])
        dataframe.loc[
            (dataframe['oscwt'] == dataframe['wtprevmin']),
            'wtplfound'] = 1
        
        dataframe['wtplfound'] = dataframe['wtplfound'].fillna(0)

        dataframe['wtmax'] = dataframe['oscwt'].rolling(wbl).max()
        dataframe['wtprevmax'] = np.where(dataframe['wtmax'] < dataframe['wtmax'].shift(), dataframe['wtmax'].shift(), dataframe['wtmax'])
        dataframe.loc[
            (dataframe['oscwt'] == dataframe['wtprevmax']),

        'wtphfound'] = 1
        dataframe['wtphfound'] = dataframe['wtphfound'].fillna(0)
        
        #--Regular WT bullish

        dataframe['wtvaluwhen_plfound_osc'], dataframe['wtinrange_plfound_osc'] = valuewhen(dataframe, 'wtplfound', 'oscwt', 1)
        dataframe.loc[
            (
                (dataframe['oscwt'] > dataframe['wtvaluwhen_plfound_osc'])&
                ( dataframe['wtinrange_plfound_osc'] == 1)
            
            ),


        'wtoschl'] = 1
      
        dataframe['wtvaluewhen_plfound_low'], dataframe['wtinrange_plFound_low'] = valuewhen(dataframe, 'wtplfound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] < dataframe['wtvaluewhen_plfound_low'])
            , 'wtpriceLL'] = 1
        #bullCond = plotBull and priceLL and oscHL and plFound
        dataframe.loc[
            (
                (dataframe['wtpriceLL'] == 1) &
                (dataframe['wtoschl'] == 1) &
                (dataframe['wtplfound'] == 1)
            )
            , 'wtbullCond'] = 1
        
    # // Hidden Bullish
        # // Osc: Lower Low
        #
        # oscLL = osc[lbR] < valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe['wtvaluewhen_plFound_osc'], dataframe['wtinrange_plFound_osc'] = valuewhen(dataframe, 'wtplfound', 'oscwt', 1)
        dataframe.loc[
            (
                (dataframe['oscwt'] < dataframe['wtvaluewhen_plFound_osc']) &
                (dataframe['wtinrange_plFound_osc'] == 1)
             )
        , 'wtoscLL'] = 1
        #
        # // Price: Higher Low
        #
        # priceHL = low[lbR] > valuewhen(plFound, low[lbR], 1)
        dataframe['wtvaluewhen_plFound_low'], dataframe['wtinrange_plFound_low'] = valuewhen(dataframe,'wtplfound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] > dataframe['wtvaluewhen_plFound_low'])
            , 'wtpriceHL'] = 1
        # hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound
        dataframe.loc[
            (
                (dataframe['wtpriceHL'] == 1) &
                (dataframe['wtoscLL'] == 1) &
                (dataframe['wtplfound'] == 1)
            )
            , 'wthiddenBullCond'] = 1


         # // Regular Bearish
        # // Osc: Lower High
        #
        # oscLH = osc[lbR] < valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['wtvaluewhen_phFound_osc'], dataframe['wtinrange_phFound_osc'] = valuewhen(dataframe, 'wtphfound', 'oscwt', 1)
        dataframe.loc[
            (
                (dataframe['oscwt'] < dataframe['wtvaluewhen_phFound_osc']) &
                (dataframe['wtinrange_phFound_osc'] == 1)
             )
        , 'wtoscLH'] = 1
        #
        # // Price: Higher High
        #
        # priceHH = high[lbR] > valuewhen(phFound, high[lbR], 1)
        dataframe['wtvaluewhen_phFound_high'], dataframe['wtinrange_phFound_high'] = valuewhen(dataframe, 'wtphfound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] > dataframe['wtvaluewhen_phFound_high'])
            , 'wtpriceHH'] = 1
        #
        # bearCond = plotBear and priceHH and oscLH and phFound
        dataframe.loc[
            (
                (dataframe['wtpriceHH'] == 1) &
                (dataframe['wtoschl'] == 1) &
                (dataframe['wtphfound'] == 1)
            )
            , 'wtbearCond'] = 1

        # // Hidden Bearish
        # // Osc: Higher High
        #
        # oscHH = osc[lbR] > valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['wtvaluewhen_phFound_osc'], dataframe['wtinrange_phFound_osc'] = valuewhen(dataframe, 'wtphfound', 'oscwt', 1)
        dataframe.loc[
            (
                (dataframe['oscwt'] > dataframe['wtvaluewhen_phFound_osc']) &
                (dataframe['wtinrange_phFound_osc'] == 1)
             )
        , 'wtoscHH'] = 1
        #
        # // Price: Lower High
        #
        # priceLH = high[lbR] < valuewhen(phFound, high[lbR], 1)
        dataframe['wtvaluewhen_phFound_high'], dataframe['wtinrange_phFound_high'] = valuewhen(dataframe, 'wtphfound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] < dataframe['wtvaluewhen_phFound_high'])
            , 'wtpriceLH'] = 1
        #
        # hiddenBearCond = plotHiddenBear and priceLH and oscHH and phFound
        dataframe.loc[
            (
                (dataframe['wtpriceLH'] == 1) &
                (dataframe['wtoscHH'] == 1) &
                (dataframe['wtphfound'] == 1)
            )
            , 'wthiddenBearCond'] = 1

        #MACD

        #RSI+MFI

        #stohrsi
        

        dataframe['schaff'] = pta.stc(dataframe)
 

        #aaron
        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']




        #Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        #RSIDIV
        len = 14
        src = dataframe['close']
        lbL = 10#5
        dataframe['osc'] = ta.RSI(src, len)
        dataframe['osc'] = dataframe['osc'].fillna(0)

        # plFound = na(pivotlow(osc, lbL, lbR)) ? false : true
        dataframe['min'] = dataframe['osc'].rolling(lbL).min()
        dataframe['prevMin'] = np.where(dataframe['min'] > dataframe['min'].shift(), dataframe['min'].shift(), dataframe['min'])
        dataframe.loc[
            (dataframe['osc'] == dataframe['prevMin'])
        , 'plFound'] = 1
        dataframe['plFound'] = dataframe['plFound'].fillna(0)

        # phFound = na(pivothigh(osc, lbL, lbR)) ? false : true
        dataframe['max'] = dataframe['osc'].rolling(lbL).max()
        dataframe['prevMax'] = np.where(dataframe['max'] < dataframe['max'].shift(), dataframe['max'].shift(), dataframe['max'])
        dataframe.loc[
            (dataframe['osc'] == dataframe['prevMax'])
        , 'phFound'] = 1
        dataframe['phFound'] = dataframe['phFound'].fillna(0)


        #------------------------------------------------------------------------------
        # Regular Bullish
        # Osc: Higher Low
        # oscHL = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
             )
        , 'oscHL'] = 1

        # Price: Lower Low
        # priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)
        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(dataframe, 'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] < dataframe['valuewhen_plFound_low'])
            , 'priceLL'] = 1
        #bullCond = plotBull and priceLL and oscHL and plFound
        dataframe.loc[
            (
                (dataframe['priceLL'] == 1) &
                (dataframe['oscHL'] == 1) &
                (dataframe['plFound'] == 1)
            )
            , 'bullCond'] = 1

        # plot(
        #      plFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bullish",
        #      linewidth=2,
        #      color=(bullCond ? bullColor : noneColor)
        #      )
        #
        # plotshape(
        #      bullCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bullish Label",
        #      text=" Bull ",
        #      style=shape.labelup,
        #      location=location.absolute,
        #      color=bullColor,
        #      textcolor=textColor
        #      )

        # //------------------------------------------------------------------------------
        # // Hidden Bullish
        # // Osc: Lower Low
        #
        # oscLL = osc[lbR] < valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
             )
        , 'oscLL'] = 1
        #
        # // Price: Higher Low
        #
        # priceHL = low[lbR] > valuewhen(plFound, low[lbR], 1)
        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(dataframe,'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] > dataframe['valuewhen_plFound_low'])
            , 'priceHL'] = 1
        # hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound
        dataframe.loc[
            (
                (dataframe['priceHL'] == 1) &
                (dataframe['oscLL'] == 1) &
                (dataframe['plFound'] == 1)
            )
            , 'hiddenBullCond'] = 1
        #
        # plot(
        #      plFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bullish",
        #      linewidth=2,
        #      color=(hiddenBullCond ? hiddenBullColor : noneColor)
        #      )
        #
        # plotshape(
        #      hiddenBullCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bullish Label",
        #      text=" H Bull ",
        #      style=shape.labelup,
        #      location=location.absolute,
        #      color=bullColor,
        #      textcolor=textColor
        #      )
        #
        # //------------------------------------------------------------------------------
        # // Regular Bearish
        # // Osc: Lower High
        #
        # oscLH = osc[lbR] < valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
             )
        , 'oscLH'] = 1
        #
        # // Price: Higher High
        #
        # priceHH = high[lbR] > valuewhen(phFound, high[lbR], 1)
        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] > dataframe['valuewhen_phFound_high'])
            , 'priceHH'] = 1
        #
        # bearCond = plotBear and priceHH and oscLH and phFound
        dataframe.loc[
            (
                (dataframe['priceHH'] == 1) &
                (dataframe['oscLH'] == 1) &
                (dataframe['phFound'] == 1)
            )
            , 'bearCond'] = 1
        #
        # plot(
        #      phFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bearish",
        #      linewidth=2,
        #      color=(bearCond ? bearColor : noneColor)
        #      )
        #
        # plotshape(
        #      bearCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bearish Label",
        #      text=" Bear ",
        #      style=shape.labeldown,
        #      location=location.absolute,
        #      color=bearColor,
        #      textcolor=textColor
        #      )
        #
        # //------------------------------------------------------------------------------
        # // Hidden Bearish
        # // Osc: Higher High
        #
        # oscHH = osc[lbR] > valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
             )
        , 'oscHH'] = 1
        #
        # // Price: Lower High
        #
        # priceLH = high[lbR] < valuewhen(phFound, high[lbR], 1)
        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] < dataframe['valuewhen_phFound_high'])
            , 'priceLH'] = 1
        #
        # hiddenBearCond = plotHiddenBear and priceLH and oscHH and phFound
        dataframe.loc[
            (
                (dataframe['priceLH'] == 1) &
                (dataframe['oscHH'] == 1) &
                (dataframe['phFound'] == 1)
            )
            , 'hiddenBearCond'] = 1
        #
        # plot(
        #      phFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bearish",
        #      linewidth=2,
        #      color=(hiddenBearCond ? hiddenBearColor : noneColor)
        #      )
        #
        # plotshape(
        #      hiddenBearCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bearish Label",
        #      text=" H Bear ",
        #      style=shape.labeldown,
        #      location=location.absolute,
        #      color=bearColor,
        #      textcolor=textColor
        #  )"""


       # dataframe['lastrsi'] = valuewhen(dataframe, 'wtplfound', 'osc'[2],0)

        return dataframe
    


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
         bconditions = []
         if self.buy_gold_cat1.value:
             bconditions.append((dataframe['wtbullCond'] == 1)&
                                 (dataframe['wtprevmin'] <= self.oslevel3)&
                                 (dataframe['wt2'] > self.oslevel3)&
                                 ((dataframe['wtprevmin'] - dataframe['wt2']) <= -5)
                                 
             
             
             
             )
         if self.buy_gold_cat2.value:
             bconditions.append((dataframe['bullCond'] == 1)&
                                 (dataframe['wtprevmin'] <= self.oslevel3)&
                                 (dataframe['wt2'] > self.oslevel3)&
                                 ((dataframe['wtprevmin'] - dataframe['wt2']) <= -5)
                                 
             
             
             
             )

         #if self.buy_mfi_cat.value == True:
         bconditions.append(qtpylib.crossed_above(dataframe['mfi'], self.buy_mfi.value))

         if self.use_bull.value:
            bconditions.append(
                    (
                        (dataframe['bullCond'] > 0) &
                        (dataframe['EWO'] > self.ewo_high.value) &
                        #(dataframe['osc'] < self.rsi_buy.value) &
                        (dataframe['volume'] > 0)
                    )
                )
         if self.use_hidden_bull.value:
            bconditions.append(
                (
                    (dataframe['hiddenBullCond'] > 0) &
                    (dataframe['EWO'] > self.ewo_high.value) &
                    #(dataframe['osc'] < self.rsi_buy.value) &
                    (dataframe['volume'] > 0)
                )
            )


         #if self.buy_wt_cat.value:
         bconditions.append(qtpylib.crossed_above(dataframe['wt1'], dataframe['wt2'])&
                                ((dataframe['wt2']-dataframe['wt1']) <=0)&
                                (dataframe['wt2']<= self.buy_wt_oversold.value)
             
             
             )

         if self.buy_vwap_cat.value:
             bconditions.append(dataframe["close"] <= dataframe['vwap'])

         if self.buy_arronup_cat.value == True:
            bconditions.append(qtpylib.crossed_above(dataframe['aroonup'], self.buy_aroonup.value))

         if self.buy_arrondown_cat.value == True:
            bconditions.append(qtpylib.crossed_above(dataframe['aroondown'], self.buy_aroondown.value))

         if self.buy_schaff_cat.value:
             bconditions.append(qtpylib.crossed_above(dataframe['schaff'], self.buy_schafff_val.value))

         bconditions.append(dataframe['volume'] > 0)


         if bconditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, bconditions),
                'buy'] = 1
            return dataframe 
    

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        sconditions = []
        if self.sell_mfi_cat.value == True:
              sconditions.append(qtpylib.crossed_below(dataframe['mfi'], self.sell_mfi.value))

        if self.use_bear.value:
            sconditions.append(
                (
                    (dataframe['bearCond'] > 0) &
                    (dataframe['volume'] > 0)
                )
            )

        if self.use_hidden_bear.value:
            sconditions.append(
                (
                    (dataframe['hiddenBearCond'] > 0) &
                    (dataframe['volume'] > 0)
                )
            )
        
        #if self.buy_wt_cat.value:
        sconditions.append(qtpylib.crossed_below(dataframe['wt1'], dataframe['wt2'])&
                                ((dataframe['wt2']-dataframe['wt1']) >=0)&
                                (dataframe['wt2']>= self.sell_wt_overbought.value)
             
             
             )
        
        if self.sell_vwap_cat.value:
             sconditions.append(dataframe["close"] >=  dataframe['vwap'])
        
        if self.sell_arrondown_cat.value == True:
            sconditions.append(qtpylib.crossed_above(dataframe['aroondown'], self.sell_aroondown.value))
        if self.sell_arronup_cat.value == True:
            sconditions.append(qtpylib.crossed_below(dataframe['aroonup'], self.sell_aroonup.value))

        #if self.sell_schaff_cat.value:
        sconditions.append(qtpylib.crossed_below(dataframe['schaff'], self.sell_schafff_val.value))
        
        sconditions.append(dataframe['volume'] > 0)
        

        if sconditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, sconditions),
                'sell'] = 1
            return dataframe