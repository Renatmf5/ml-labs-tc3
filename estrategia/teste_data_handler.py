import pandas as pd
from indicadores import *
from services.binance_client import client
from sklearn.preprocessing import StandardScaler
   
def process_candles(symbol, data_path, timeframe):
    
    data_path = calcular_retorno_candle(data_path)
    data_path = calcular_volatilidade_candles(data_path, timeframe)
    data_path = calcular_indicadores_tendencia(data_path, timeframe)
    data_path = calcular_indicadores_momentum(data_path, timeframe)
    data_path = calcular_indicadores_volatilidade(data_path, timeframe)
    data_path = calcular_indicadores_volume(data_path, timeframe)
    data_path = calcular_volatilidade_adp_volumes_direcional(data_path, timeframe)
    data_path = data_path.dropna()
    data_path = gerar_sinal(data_path, timeframe)
    # Dropar colunas open_time open high low close
    #data_path = data_path.drop(columns=['open_time', 'open', 'high', 'low', 'close'])
    
    # Salvar o dataframe no formato csv e parquet da raiz desse projeto
    data_path.to_parquet(f'{symbol}_real_test.parquet', index=False)



def get_historical_kliness(symbol, interval, limit=500):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        klines = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
        ])
        klines = klines[['open_time', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades', 'taker_buy_base_asset_volume']]
            # Converter colunas para tipos numéricos
        klines['open'] = pd.to_numeric(klines['open'], errors='coerce')
        klines['high'] = pd.to_numeric(klines['high'], errors='coerce')
        klines['low'] = pd.to_numeric(klines['low'], errors='coerce')
        klines['close'] = pd.to_numeric(klines['close'], errors='coerce')
        klines['volume'] = pd.to_numeric(klines['volume'], errors='coerce')
        klines['number_of_trades'] = pd.to_numeric(klines['number_of_trades'], errors='coerce')
        klines['taker_buy_base_asset_volume'] = pd.to_numeric(klines['taker_buy_base_asset_volume'], errors='coerce')
        
        # Converter timestamp para data e hora
        klines['open_time'] = pd.to_datetime(klines['open_time'], unit='ms')
        
        process_candles(symbol=symbol, data_path=klines, timeframe=interval)
        
        return 
    except Exception as e:
        print(f"Erro ao obter dados históricos: {e}")
        return None
      
