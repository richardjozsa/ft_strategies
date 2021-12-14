import technical.indicators as technical

from pandas import DataFrame
import numpy as np  # noqa
import pandas as pd  # noqa

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)





class TKcros(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # # ROI table:
    minimal_roi = {
	    "0": 0.284,
        "42": 0.037,
        "219": 0.019,
        "572": 0
    }

    stoploss = -0.5

    #hyperobtalbe

   
    
    
    
    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    use_custom_stoploss = False

    # Optimal timeframe for the strategy.
    timeframe = '15m'
    inf_1h = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False

    }
    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    buyema = IntParameter(100, 200, default = 120, space ='buy', optimize = True)
    
    buytema = IntParameter(3, 20, default = 9, space = 'buy', optimize = True)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
 
      
      ichi = technical.ichimoku(dataframe)
      dataframe['tenkan'] = ichi['tenkan_sen']
      dataframe['kijun'] = ichi['kijun_sen']
      dataframe['span_a'] = ichi['senkou_span_a']
      dataframe['span_b'] = ichi['senkou_span_b']
      dataframe['cloud_green']=ichi['cloud_green']
      dataframe['cloud_red']=ichi['cloud_red']
      dataframe['ema200'] = ta.EMA(dataframe, timeperiod = self.buyema.value)
      dataframe['tema'] = ta.TEMA(dataframe, timeperiod= self.buytema.value)      

      return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['tenkan'].shift(1)<dataframe['kijun'].shift(1)) &
                (dataframe['tenkan']>dataframe['kijun']) &
                (qtpylib.crossed_above(dataframe['tema'], dataframe['ema200'])) &
                (dataframe['cloud_red']==True)&
                (dataframe['volume'] > 0)

                
            ),
        
        'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(
            (dataframe['tenkan'].shift(1)>dataframe['kijun'].shift(1)) &
            (dataframe['tenkan']<dataframe['kijun'])&
            

            (dataframe['volume'] > 0)
        
        
        
        ), 'sell'] = 1
        return dataframe