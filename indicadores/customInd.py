import pandas as pd
import numpy as np

def calcular_retorno_candle(df):
    df['retorno_candle'] = df['close'].pct_change().round(2)
    return df

def gerar_sinal(df, timeframe):
    # criar condição de valor de alvo_base com base no timeframe
    if timeframe == '1d':
        alvo_base = 2
    elif timeframe == '4h':
        alvo_base = 3
    elif timeframe == '1h':
        alvo_base = 4
    sinais = []
    tempos = []
    tamanho_alvo = []
    for i in range(len(df)):
        sinal = None
        linhas = 0
        #quando volatilidade nao tiver valor use alvo base com 2 casas decimais
        if pd.isna(df['volatilidade'].iloc[i]):
            alvo_ajustado = alvo_base
        else:
            alvo_ajustado = alvo_base * df['volatilidade'].iloc[i]
            
        for j in range(i, len(df)-1):
            retorno_high = (df['high'].iloc[j+1] / df['close'].iloc[i]) - 1
            retorno_low = (df['low'].iloc[j+1] / df['close'].iloc[i]) - 1
            
            linhas += 1
            if retorno_high >= alvo_ajustado:
                sinal = 1
                break
            elif retorno_low <= -alvo_ajustado:
                sinal = 0
                break
            
        sinais.append(sinal)
        tempos.append(linhas)
        tamanho_alvo.append(alvo_ajustado)
    df['signal'] = sinais
    df['tempo'] = tempos
    df['tamanho_alvo'] = tamanho_alvo
    
    # Calcular a média de tempo
    media_tempo = df['tempo'].mean()
    
    # Calcular o percentil 75 dos tempos
    percentil_35 = np.percentile(df['tempo'], 70)
    
    # Atualizar sinais para o valor 2 para os sinais que levaram mais tempo que a média
    df.loc[df['tempo'] > percentil_35, 'signal'] = 2
    
    #dropar colunas tempo
    df = df.drop(columns=['tempo']) 
    
    #dropar coluna tamanho_alvo
    df = df.drop(columns=['tamanho_alvo'])
    
    df = df.dropna()
    df['signal'] = df['signal'].astype('int64')
    return df

def calcular_volatilidade_adp_volumes_direcional(df, timeframe):
    # Calcular a volatilidade adaptada para volumes em BTC e USDT
    if timeframe == '1d':
        df['mean_volume'] = df['volume'].rolling(window=7).mean()
        df['std_volume'] = df['volume'].rolling(window=7).std()
        # Criar a proporção entre 'taker_buy_base_asset_volume' (BTC) e volume total (BTC)
        df['proportion_taker_BTC'] = df['taker_buy_base_asset_volume'] / df['volume']
        # Calcular a média histórica da proporção BTC e USDT
        df['mean_proportion_BTC'] = df['proportion_taker_BTC'].rolling(window=14).mean()
        # Calcular o desvio padrão histórico das proporções
        df['std_proportion_BTC'] = df['proportion_taker_BTC'].rolling(window=14).std()
        df['volatilidade'] = df['close'].pct_change().rolling(window=30).std()

    if timeframe == '4h':
        df['mean_volume'] = df['volume'].rolling(window=14).mean()
        df['std_volume'] = df['volume'].rolling(window=14).std()
        df['proportion_taker_BTC'] = df['taker_buy_base_asset_volume'] / df['volume']
        df['mean_proportion_BTC'] = df['proportion_taker_BTC'].rolling(window=28).mean()
        df['std_proportion_BTC'] = df['proportion_taker_BTC'].rolling(window=28).std()
        df['volatilidade'] = df['close'].pct_change().rolling(window=42).std()
        
    if timeframe == '1h':
        df['mean_volume'] = df['volume'].rolling(window=24).mean()
        df['std_volume'] = df['volume'].rolling(window=24).std()
        df['proportion_taker_BTC'] = df['taker_buy_base_asset_volume'] / df['volume']
        df['mean_proportion_BTC'] = df['proportion_taker_BTC'].rolling(window=48).mean()
        df['std_proportion_BTC'] = df['proportion_taker_BTC'].rolling(window=48).std()
        df['volatilidade'] = df['close'].pct_change().rolling(window=180).std()
        
    # Calcular o z-score (desvio em relação à média) para identificar desvios significativos
    df['z_score_BTC'] = (df['proportion_taker_BTC'] - df['mean_proportion_BTC']) / df['std_proportion_BTC']
    
    # Ao final tornar a coluna index devolta para um id sequencial
    df = df.reset_index(drop=True)
    return df
