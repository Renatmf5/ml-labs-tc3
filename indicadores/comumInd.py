import pandas_ta as ta
import pandas as pd

def calcular_volatilidade_candles(df, timeframe):
    if timeframe == '1h':
        df['ATR_Candle'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    elif timeframe == '4h':
        df['ATR_Candle'] = ta.atr(df['high'], df['low'], df['close'], length=20)
    return df

def calcular_indicadores_tendencia(df, timeframe):
    if timeframe == '1h':
        df['ema_9'] = df['close'] - ta.ema(df['close'], length=9)
        df['ema_21'] = df['close'] -ta.ema(df['close'], length=21)
        df['ema_50'] =  df['close'] - ta.ema(df['close'], length=50)
        df['ema_200'] = df['close'] - ta.ema(df['close'], length=200)
        df['sma_50'] = df['close'] - ta.sma(df['close'], length=50)
        df['sma_100'] = df['close'] - ta.sma(df['close'], length=100)
        df['sma_200'] = df['close'] - ta.sma(df['close'], length=200)
        
        df['ema_9_21_diff'] = df['ema_9'] - df['ema_21']
        df['ema_9_50_diff'] = df['ema_9'] - df['ema_50']
        df['ema_21_50_diff'] = df['ema_21'] - df['ema_50']
        df['ema_21_200_diff'] = df['ema_21'] - df['ema_200']
        df['ema_50_200_diff'] = df['ema_50'] - df['ema_200']
        
        df['sma_50_100_diff'] = df['sma_50'] - df['sma_100']
        df['sma_50_200_diff'] = df['sma_50'] - df['sma_200']
        
        
        ichimoku_df, *_ = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    elif timeframe == '4h':
        df['ema_21'] = df['close'] - ta.ema(df['close'], length=21)
        df['ema_50'] = df['close'] - ta.ema(df['close'], length=50)
        df['ema_100'] = df['close'] - df['close'] - ta.ema(df['close'], length=100)
        df['ema_200'] = df['close'] - ta.ema(df['close'], length=200)
        df['sma_100'] = df['close'] - ta.sma(df['close'], length=100)
        df['sma_200'] = df['close'] - ta.sma(df['close'], length=200)
        
        df['ema_9_21_diff'] = df['ema_9'] - df['ema_21']
        df['ema_9_50_diff'] = df['ema_9'] - df['ema_50']
        df['ema_21_50_diff'] = df['ema_21'] - df['ema_50']
        df['ema_21_200_diff'] = df['ema_21'] - df['ema_200']
        df['ema_50_200_diff'] = df['ema_50'] - df['ema_200']
        
        df['sma_100_200_diff'] = df['sma_100'] - df['sma_200']
        
        ichimoku_df, *_ = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=18, kijun=52, senkou=104)
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=14, multiplier=4)
        adx = ta.adx(df['high'], df['low'], df['close'], length=20)
    
    df['ichimoku_a'] = df['close'] - ichimoku_df['ISA_9']
    df['ichimoku_b'] = df['close'] - ichimoku_df['ISB_26']
    df['ichimoku_base'] = df['close'] - ichimoku_df['ITS_9']
    df['ichimoku_conversion'] = df['close'] - ichimoku_df['IKS_26']
    df['ichimoku_lagging'] = df['close'] - df['close'].shift(26)
    df['supertrend'] = df['close'] - supertrend['SUPERT_10_3.0']
    df['adx'] = adx['ADX_14']
    
    # Calcular diferenças entre os componentes do Ichimoku Cloud
    df['ichimoku_conversion_base_diff'] = ichimoku_df['IKS_26'] - ichimoku_df['ITS_9']
    df['ichimoku_a_b_diff'] = ichimoku_df['ISA_9'] - ichimoku_df['ISB_26']
    
    # Garantir que o índice do DataFrame seja um índice de tempo
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    df['vwap'] = df['close'] - ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    return df

def calcular_indicadores_momentum(df, timeframe):
    if timeframe == '1h':
        df['rsi'] = ta.rsi(df['close'], length=14)
        stoch_rsi = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    elif timeframe == '4h':
        df['rsi'] = ta.rsi(df['close'], length=21)
        stoch_rsi = ta.stochrsi(df['close'], length=21, rsi_length=21, k=5, d=5)
        macd = ta.macd(df['close'], fast=24, slow=52, signal=9)
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=50)
    
    df['stoch_rsi_k'] = stoch_rsi['STOCHRSIk_14_14_3_3']
    df['stoch_rsi_d'] = stoch_rsi['STOCHRSId_14_14_3_3']
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    return df

def calcular_indicadores_volatilidade(df, timeframe):
    if timeframe == '1h':
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bbands = ta.bbands(df['close'], length=20, std=2)
        donchian = ta.donchian(high=df['high'], low=df['low'], close=df['close'], length=20)
    elif timeframe == '4h':
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=20)
        bbands = ta.bbands(df['close'], length=50, std=2)
        donchian = ta.donchian(high=df['high'], low=df['low'], close=df['close'], length=50)
    
    df['bb_upper'] = df['close'] - bbands['BBU_20_2.0']
    df['bb_middle'] = df['close'] - bbands['BBM_20_2.0']
    df['bb_lower'] = df['close'] - bbands['BBL_20_2.0']
    
    # Diferença entre as próprias bandas de Bollinger
    df['bb_upper_middle_diff'] = bbands['BBU_20_2.0'] - bbands['BBM_20_2.0']
    df['bb_middle_lower_diff'] = bbands['BBM_20_2.0'] - bbands['BBL_20_2.0']
    df['bb_upper_lower_diff'] = bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']
    
    # Identifica colunas reais do Donchian Channel
    donchian_cols = list(donchian.columns)
    
    df['donchian_upper'] = df['close'] - donchian.get(donchian_cols[0], None)
    df['donchian_lower'] = df['close'] - donchian.get(donchian_cols[1], None)
    
    # Diferença entre os próprios valores do Donchian Channel
    df['donchian_upper_lower_diff'] = df['donchian_upper'] - df['donchian_lower']
    
    return df

def calcular_indicadores_volume(df, timeframe):
    df['obv'] = ta.obv(df['close'], df['volume'])
    if timeframe == '1h':
        mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        df['mfi'] = mfi.astype('float64')  # Converte para float64
        df['vwma'] = df['close'] - ta.vwma(df['close'], df['volume'], length=20)
    elif timeframe == '4h':
        mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=21)
        df['mfi'] = mfi.astype('float64')  # Converte para float64
        df['vwma'] = df['close'] - ta.vwma(df['close'], df['volume'], length=50)
    
    return df