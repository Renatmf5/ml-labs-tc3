import pandas as pd
from modelosMLDP import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from datetime import timedelta
from estrategia.teste_data_handler import *
import boto3
import joblib
import io



class Backtester:
    def __init__(self, estrategia, ticker, bucket_name):
        self.estrategia = estrategia
        self.ticker = ticker
        self.bucket_name = bucket_name
        self.trades_df = pd.DataFrame()
        self.Simulation_real_time_df = pd.DataFrame()
        self.cumulative_returns = [0]  # Lista para armazenar o retorno acumulado de cada fatia
        self.initial_investment = 1000  # Valor inicial do investimento
        self.current_investment = self.initial_investment  # Valor atual do investimento
        self.scaler_path_ppo = os.path.join("modelosMLDP", "dataModels", "ppo_scaler.pkl")
        self.scaler_key_ppo = f'models/{ticker}/scaler/{ticker}_ppo_scaler.pkl'
        
    def execute_model_trainer(self, type, train_data, test_data):
        if type == 'completa':
           # trata XGboost
           self.XGboost_model = XGBoostModelTrainer(self.ticker, self.bucket_name)
           
           #features_to_remove_XGBoost = ['timestamp', 'open', 'high', 'low', 'close', 'target']
           selected_columns_XGBoost = ['ema_20_diff','atr_14','macd_diff','macd_signal_diff', 'macd_hist','rsi_14','adx_14', 'bb_upper_diff','bb_middle_diff', 'bb_lower_diff', 'wma_14_diff', 'cci_20','stc','roc_10','mean_proportion_BTC','std_proportion_BTC','passado_1','passado_2','passado_3','proportion_taker_BTC', 'z_score_BTC' ,'proximo_topo_curto','proximo_fundo_curto','proximo_topo_medio','proximo_fundo_medio','proximo_topo_longo','proximo_fundo_longo','target']
          
           # Selecionar apenas as colunas especificadas
           #train_data_XGBoost = train_data.drop(columns=features_to_remove_XGBoost)
           train_data_XGBoost = train_data[selected_columns_XGBoost]
           
           #test_data_XGBoost = test_data.drop(columns=features_to_remove_XGBoost)
           test_data_XGBoost = test_data[selected_columns_XGBoost]
           scaler_XGBoost = self.XGboost_model.train_model_regressor(train_data_XGBoost)         
           
           # Trata LSTM
           
           self.LSTM_model = LSTMClassificationModel(self.ticker, self.bucket_name)
            
           #features_to_remove_LSTM = ['timestamp', 'open', 'high', 'low', 'close', 'signal']
           selected_columns_LSTM = ['macd_diff_lstm','macd_signal_lstm','macd_hist_lstm','rsi_14', 'willr_14','donchian_lower','donchian_mid', 'donchian_high','donchuan_lower_diff', 'donchian_mid_diff', 'aroon_up','aroon_down','chop','fisher','zscore','mean_proportion_BTC','std_proportion_BTC','passado_1','passado_2','passado_3','proportion_taker_BTC', 'z_score_BTC',  'proximo_topo_curto','proximo_fundo_curto','proximo_topo_medio','proximo_fundo_medio','proximo_topo_longo','proximo_fundo_longo','signal']
           train_data_LSTM = train_data[selected_columns_LSTM]
           test_data_LSTM = test_data[selected_columns_LSTM]
           scaler_LSTM = self.LSTM_model.train_lstm_classification(train_data_LSTM)
           
           
           # Trata MLP
           
           self.MLP_model = MLPClassificationModel(self.ticker, self.bucket_name, 2)
           selected_columns_MLP = ['sma_5_diff','sma_20_diff','sma_50_diff', 'stoch_k','stoch_d','vwap_diff', 'mfi','tsi_stoch','dmi_plus','dmi_minus','adx','psar','cmo','obv','kc_upper','kc_mid','kc_lower','kc_upper_diff','kc_mid_diff','mean_proportion_BTC','std_proportion_BTC','passado_1','passado_2','passado_3','proportion_taker_BTC', 'z_score_BTC','proximo_topo_curto','proximo_fundo_curto','proximo_topo_medio','proximo_fundo_medio','proximo_topo_longo','proximo_fundo_longo','signal']
           train_data_MLP = train_data[selected_columns_MLP ]
           test_data_MLP = test_data[selected_columns_MLP]
           scaler_MLP = self.MLP_model.train_mlp_classification(train_data_MLP)
           
           # Chama o modelo de ensemble
           self.ensemble_model = StackingEnsembleModel(
                self.bucket_name,
                self.ticker,
                model_lstm=self.LSTM_model,
                model_xgb=self.XGboost_model,
                model_mlp=self.MLP_model,
                scaler_lstm=scaler_LSTM,
                scaler_xgb=scaler_XGBoost,
                #target_scaler_xgb=target_scaler_xgb,
                scaler_mlp=scaler_MLP
            )
           
           meta_X_train, min_length_train, scaler_ensemble = self.ensemble_model.treinar_meta_modelo(train_data_LSTM, train_data_XGBoost, train_data_MLP, train_data['signal'])
           
           meta_X_test, meta_predictions,meta_predictions_proba, min_length_test = self.ensemble_model.prever_meta_modelo(test_data_LSTM, test_data_XGBoost, test_data_MLP, test_data['signal'], scaler_ensemble)
           ensemble_accuracy = self.ensemble_model.avaliar_meta_modelo(test_data['signal'],  meta_predictions, min_length_test)
           print(f'Acurácia do ensemble: {ensemble_accuracy}')
           
           # Trata RL_PPO_model
           
           meta_X_train = scaler_ensemble.transform(meta_X_train)
           meta_X_test = scaler_ensemble.transform(meta_X_test)
           
           # Selecionar as colunas finais para o RL PPO
           selected_columns_RL_PPO = ['sma_5_diff', 'sma_20_diff', 'sma_50_diff', 'ema_20_diff', 'mean_proportion_BTC', 'std_proportion_BTC', 'proportion_taker_BTC', 'z_score_BTC','cci_20', 'stc', 'roc_10', 'cmo', 'obv', 'mfi', 'tsi_stoch', 'dmi_plus', 'dmi_minus', 'adx',
            'pred_lstm_proba', 'pred_xgb_proba', 'pred_mlp_proba', 'ensemble_signal', 'low', 'high', 'close']
           
           # Separar as colunas de meta_X_train
           train_data = train_data[:min_length_train]
           
           # Selecionar as colunas de interesse para escalonamento
           train_data_features = train_data[['sma_5_diff', 'sma_20_diff', 'sma_50_diff', 'ema_20_diff', 'mean_proportion_BTC', 'std_proportion_BTC', 'proportion_taker_BTC', 'z_score_BTC','cci_20', 'stc', 'roc_10', 'cmo', 'obv', 'mfi', 'tsi_stoch', 'dmi_plus', 'dmi_minus', 'adx']]

            # Escalonar os dados com MinMaxScaler
           scaler_ppo = MinMaxScaler(feature_range=(0, 1))
           train_data_scaled = scaler_ppo.fit_transform(train_data_features)
           
           # Converter de volta para DataFrame
           train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data_features.columns)

           train_data_scaled['pred_lstm_proba'] = meta_X_train[:, 0]
           train_data_scaled['pred_xgb_proba'] = meta_X_train[:, 1]
           train_data_scaled['pred_mlp_proba'] = meta_X_train[:, 2]
           train_data_scaled['ensemble_signal'] = train_data['signal'].values
           # Adicionar as colunas 'low', 'high' e 'close' ao test_data escalonado
           train_data_scaled['low'] = train_data['low'].values
           train_data_scaled['high'] = train_data['high'].values
           train_data_scaled['close'] = train_data['close'].values
           
           
           # Separar as colunas de meta_X_test
           test_data = test_data[:min_length_test]
           
           test_data_features = test_data[['sma_5_diff', 'sma_20_diff', 'sma_50_diff', 'ema_20_diff', 'mean_proportion_BTC', 'std_proportion_BTC', 'proportion_taker_BTC', 'z_score_BTC','cci_20', 'stc', 'roc_10', 'cmo', 'obv', 'mfi', 'tsi_stoch', 'dmi_plus', 'dmi_minus', 'adx']]

            # Escalonar os dados do test_data com o mesmo scaler usado no train_data
           test_data_scaled = scaler_ppo.transform(test_data_features)

            # Converter de volta para DataFrame para facilitar a manipulação
           test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data_features.columns)

            # Adicionar as colunas de probabilidades do ensemble ao test_data escalonado
           test_data_scaled['pred_lstm_proba'] = meta_X_test[:, 0]
           test_data_scaled['pred_xgb_proba'] = meta_X_test[:, 1]
           test_data_scaled['pred_mlp_proba'] = meta_X_test[:, 2]

            # Adicionar a coluna 'ensemble_signal' ao test_data escalonado
           test_data_scaled['ensemble_signal'] = meta_predictions_proba

            # Adicionar as colunas 'low', 'high' e 'close' ao test_data escalonado
           test_data_scaled['low'] = test_data['low'].values
           test_data_scaled['high'] = test_data['high'].values
           test_data_scaled['close'] = test_data['close'].values


            # Criar o conjunto de dados final para o RL PPO
           test_data_RL_PPO = test_data_scaled[selected_columns_RL_PPO]
           train_data_RL_PPO = train_data_scaled[selected_columns_RL_PPO]

           
           RL_PPO_model = CryptoTradingModel(self.ticker, self.bucket_name)
           RL_PPO_model.train_ppo(train_data_RL_PPO)
           
           metrics_df = RL_PPO_model.predict_ppo(test_data_RL_PPO)
           print(metrics_df)
           
           # Salvar o scaler
           joblib.dump(scaler_ppo, self.scaler_path_ppo)
           
           # Fazer upload do modelo treinado e do scaler para o S3
           s3_client = boto3.client('s3')
           s3_client.upload_file(self.scaler_path_ppo, self.bucket_name, self.scaler_key_ppo)
           
           
           
                     
           
           return metrics_df
    
           
    def backtesting_optimized(self):
        
        self.trades_df = self.estrategia.df
        self.type = self.estrategia.type
        self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
        self.trades_df = self.trades_df.sort_values(by='timestamp')
        self.trades_df = self.trades_df.dropna()

        # Definir o início e o fim do período de treinamento inicial (6 meses)        
        start_date = self.trades_df['timestamp'].min()
        end_date_train = start_date + timedelta(days=365/2)
        
        # Definir o início do período de teste (1 mês após o período de treinamento)
        start_date_test = end_date_train
        end_date_test = start_date_test + timedelta(days=30)
        
        results = []
        
        for i in range(12):
            # Filtrar os dados de treinamento e teste
            train_data = self.trades_df[(self.trades_df['timestamp'] >= start_date) & (self.trades_df['timestamp'] < end_date_train)]
            test_data = self.trades_df[(self.trades_df['timestamp'] >= start_date_test) & (self.trades_df['timestamp'] < end_date_test)]
            

            if test_data.empty:
                break
            
            # Executar o treinamento e a predição
            metrics_df = self.execute_model_trainer(self.type,train_data, test_data)
            
            """
            y_test = test_data['signal'].values
            y_test = y_test[:len(meta_predictions)]
            
            # Obter o relatório de classificação como dicionário
            report_dict = classification_report(y_test, meta_predictions, target_names=['Classe 0', 'Classe 1'], output_dict=True)
            
            # Extrair precisão e recall para cada classe
            precision0 = report_dict['Classe 0']['precision']
            recall0 = report_dict['Classe 0']['recall']
            precision1 = report_dict['Classe 1']['precision']
            recall1 = report_dict['Classe 1']['recall']
            # Armazenar os resultados
            results.append({
                'test_start': start_date_test,
                'test_end': end_date_test,
                'precision0': precision0,
                'recall0': recall0,
                'precision1': precision1,
                'recall1': recall1,
                'accuracy': accuracy_score(y_test, meta_predictions)
            })
            """
            # Adicionar colunas de período de teste ao DataFrame de métricas
            metrics_df['start_time'] = start_date_test
            metrics_df['end_time'] = end_date_test

            # Verificar se metrics_df é bidimensional
            if metrics_df.ndim == 2:
                # Armazenar os resultados
                results.append(metrics_df)
            else:
                print(f"metrics_df não é bidimensional: {metrics_df.shape}")

            # Atualizar as datas para a próxima iteração
            end_date_train += timedelta(days=30)
            start_date_test += timedelta(days=30)
            end_date_test += timedelta(days=30)

        # Concatenar todos os DataFrames de resultados
        if results:
            results_df = pd.concat(results, ignore_index=True)
            print(results_df)
        else:
            print("Nenhum resultado para concatenar.")
        
        
        return results_df
           
    def backtesting_real_time(self):
        self.trades_df = self.estrategia.df
        self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
        self.trades_df = self.trades_df.sort_values(by='timestamp')
        self.trades_df = self.trades_df.dropna()

        # Definir o início e o fim do período de treinamento inicial (2 anos)
        start_date = self.trades_df['timestamp'].min()
        end_date_train = start_date + timedelta(days=365/2)
        
        # Definir o início do período de teste (1 mês após o período de treinamento)
        start_date_test = end_date_train
        end_date_test = start_date_test + timedelta(days=30)
        
        trades = []
        position_open = False
        entry_price = 0
        entry_index = 0
        entry_timestamp = None
        tax = 0.0012
        
        # Inicializar Simulation_real_time_df fora do loop
        self.Simulation_real_time_df = pd.DataFrame()
        
        for i in range(12):
            # Filtrar os dados de treinamento e teste
            train_data = self.trades_df[(self.trades_df['timestamp'] >= start_date) & (self.trades_df['timestamp'] < end_date_train)]
            test_data = self.trades_df[(self.trades_df['timestamp'] >= start_date_test) & (self.trades_df['timestamp'] < end_date_test)]
            
            if test_data.empty:
                break
            
            # Executar o treinamento e a predição
            result, scaler = self.execute_model_trainer('classificação_xgboost',train_data, test_data)
            y_pred = self.model.predict_classifier(result, scaler)
            test_data = test_data.copy()
            test_data.loc[:, 'predicted_signal'] = y_pred  # Usar .loc para evitar o aviso
            
            for i in range(len(test_data)):
                if not position_open and test_data['predicted_signal'].iloc[i] in [1, 0]:
                    # Abrir uma nova posição
                    position_open = True
                    entry_price = test_data['close'].iloc[i]
                    entry_index = i
                    entry_signal = test_data['predicted_signal'].iloc[i]
                    entry_timestamp = test_data['timestamp'].iloc[i]
                    target = 4 * test_data['volatilidade'].iloc[i].round(4)
                elif position_open:
                    # Verificar se a posição deve ser fechada
                    if entry_signal == 1:  # Posição de compra
                        if (test_data['high'].iloc[i] - entry_price) / entry_price >= target:
                            # Saída com sucesso (stop gain)
                            trade_return = (target-tax)
                            self.current_investment *= (1 + trade_return)
                            trades.append({
                                'entry_timestamp': entry_timestamp,
                                'entry_index': entry_index,
                                'exit_index': i,
                                'entry_price': entry_price,
                                'exit_price': test_data['high'].iloc[i],
                                'return': trade_return,
                                'signal': entry_signal,
                                'success': True,
                                'exit_timestamp': test_data['timestamp'].iloc[i],
                                'investment_value': self.current_investment
                            })
                            position_open = False
                        elif (test_data['low'].iloc[i] - entry_price) / entry_price <= -target:
                            # Saída com perda (stop loss)
                            trade_return = -(target+tax)
                            self.current_investment *= (1 + trade_return)
                            trades.append({
                                'entry_timestamp': entry_timestamp,
                                'entry_index': entry_index,
                                'exit_index': i,
                                'entry_price': entry_price,
                                'exit_price': test_data['low'].iloc[i],
                                'return': trade_return,
                                'signal': entry_signal,
                                'success': False,
                                'exit_timestamp': test_data['timestamp'].iloc[i],
                                'investment_value': self.current_investment
                            })
                            position_open = False
                    elif entry_signal == 0:  # Posição de venda
                        if (entry_price - test_data['low'].iloc[i]) / entry_price >= target:
                            # Saída com sucesso (stop gain)
                            trade_return = (target-tax)
                            self.current_investment *= (1 + trade_return)
                            trades.append({
                                'entry_timestamp': entry_timestamp,
                                'entry_index': entry_index,
                                'exit_index': i,
                                'entry_price': entry_price,
                                'exit_price': test_data['low'].iloc[i],
                                'return': trade_return,
                                'signal': entry_signal,
                                'success': True,
                                'exit_timestamp': test_data['timestamp'].iloc[i],
                                'investment_value': self.current_investment
                            })
                            position_open = False
                        elif (entry_price - test_data['high'].iloc[i]) / entry_price <= -target:
                            # Saída com perda (stop loss)
                            trade_return = -(target + tax)
                            self.current_investment *= (1 + trade_return)
                            trades.append({
                                'entry_timestamp': entry_timestamp,
                                'entry_index': entry_index,
                                'exit_index': i,
                                'entry_price': entry_price,
                                'exit_price': test_data['high'].iloc[i],
                                'return': trade_return,
                                'signal': entry_signal,
                                'success': False,
                                'exit_timestamp': test_data['timestamp'].iloc[i],
                                'investment_value': self.current_investment
                            })
                            position_open = False
            
            # Atualizar as datas para a próxima iteração
            end_date_train += timedelta(days=30)
            start_date_test += timedelta(days=30)
            end_date_test += timedelta(days=30)
            
        # Concatenar os novos trades ao DataFrame principal
        self.Simulation_real_time_df = pd.DataFrame(trades)
            
        
        return self.Simulation_real_time_df
   
    def ler_parametros_scaler_do_s3(self, symbol):
        s3 = boto3.client('s3')
        try:
            response = s3.get_object(Bucket=self.bucket_name, Key=f'models/{symbol}/scaler/{symbol}_scaler.pkl')
            content = response['Body'].read()
            scaler = joblib.load(io.BytesIO(content))
            if isinstance(scaler, StandardScaler):
                return scaler
            else:
                print("O objeto carregado não é um StandardScaler.")
                return None
        except Exception as e:
            print(f"Erro ao obter parâmetros do scaler do S3: {e}")
            return None
    # Ler modelo do xgboost salvo no bucket do S3
    def ler_modelo_xgboost_do_s3(self, symbol):
        s3 = boto3.client('s3')
        try:
            response = s3.get_object(Bucket=self.bucket_name, Key=f'models/{symbol}/xgboost/xgboost_classifier_model.pkl')
            content = response['Body'].read()
            model = joblib.load(io.BytesIO(content))
            if isinstance(model, XGBClassifier):
                return model
            else:
                print("O objeto carregado não é um XGBClassifier.")
                return None
        except Exception as e:
            print(f"Erro ao obter modelo do S3: {e}")
            return None 
    
    def backtest_compara_future_model(self):
        results1 = []
        # Carregar o modelo treinado localmente
        model_path = os.path.join("modelosMLDP", "dataModels", "xgboost_classifier_model.pkl")
        self.classifier_model = joblib.load(model_path)
        
        # Carregar dados históricos
        get_historical_kliness(symbol='BTCUSDT', interval='1h', limit=1500)
        
        # Carregar o DataFrame de teste
        test_data_parquet = pd.read_parquet('BTCUSDT_real_test.parquet')
                
        # Carregar o DataFrame de treino
        train_data = pd.read_parquet('data_frame_3A.parquet')
        
        # Ajustar os dados de teste para que comecem a partir da última data do conjunto de treino
        last_train_timestamp = train_data['timestamp'].max()
        test_data_parquet = test_data_parquet[test_data_parquet['open_time'] >= last_train_timestamp]
        test_data_parquet_test = test_data_parquet.copy()
        
        # Remover colunas especificadas
        test_data_parquet = test_data_parquet.drop(columns=['open_time', 'open', 'high', 'low', 'close'])
        
        # Treinar o modelo localmente
        teste_data, scaler = self.execute_model_trainer('classificação_xgboost',train_data, train_data)
                
        # Obter sinais
        y_test_parquet_signal = test_data_parquet['signal']
        test_data_parquet = test_data_parquet.drop(columns=['signal'])
        
        scaled = self.ler_parametros_scaler_do_s3(self.ticker)
        model_trained = self.ler_modelo_xgboost_do_s3(self.ticker)
        
        #escalar dados de teste X
        test_data_parquet = scaled.transform(test_data_parquet)
                       
        # Predizer usando o modelo local
        Y_pred_local_parquet = model_trained.predict(test_data_parquet)
                
        # Comparar as predições
        accuracy_local_parquet = accuracy_score(y_test_parquet_signal, Y_pred_local_parquet)
       
        report_dict = classification_report(y_test_parquet_signal, Y_pred_local_parquet, target_names=['Classe 0', 'Classe 1', 'Classe 2'], output_dict=True)
            
        # Extrair precisão e recall para cada classe
        precision0 = report_dict['Classe 0']['precision']
        recall0 = report_dict['Classe 0']['recall']
        precision1 = report_dict['Classe 1']['precision']
        recall1 = report_dict['Classe 1']['recall']
        precision2 = report_dict['Classe 2']['precision']
        recall2 = report_dict['Classe 2']['recall']
        
        results1.append({
            'precision0': precision0,
            'recall0': recall0,
            'precision1': precision1,
            'recall1': recall1,
            'precision2': precision2,
            'recall2': recall2,
            'accuracy_local_parquet': accuracy_local_parquet
        })
              
        
        # Imprimir lista resultados em formato de tabela
        results_df = pd.DataFrame(results1)
        print(results_df)
        
        test_data_parquet_test['predicted_signal'] = Y_pred_local_parquet
        self.backtest_trading_with_real_data(test_data_parquet_test)      
        
        return
    
    def backtest_trading_with_real_data(self, test_data):
        trades = []
        position_open = False
        entry_price = 0
        entry_index = 0
        entry_timestamp = None
        tax = 0.0012
        for i in range(len(test_data)):
            if not position_open and test_data['predicted_signal'].iloc[i] in [1, 0]:
                # Abrir uma nova posição
                position_open = True
                entry_price = test_data['close'].iloc[i]
                entry_index = i
                entry_signal = test_data['predicted_signal'].iloc[i]
                entry_timestamp = test_data['open_time'].iloc[i]
                target = 4 * test_data['volatilidade'].iloc[i].round(4)
            elif position_open:
                # Verificar se a posição deve ser fechada
                if entry_signal == 1:  # Posição de compra
                    if (test_data['high'].iloc[i] - entry_price) / entry_price >= target:
                        # Saída com sucesso (stop gain)
                        trade_return = (target-tax)
                        self.current_investment *= (1 + trade_return)
                        trades.append({
                            'entry_timestamp': entry_timestamp,
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': test_data['high'].iloc[i],
                            'return': trade_return,
                            'signal': entry_signal,
                            'success': True,
                            'exit_timestamp': test_data['open_time'].iloc[i],
                            'investment_value': self.current_investment
                        })
                        position_open = False
                    elif (test_data['low'].iloc[i] - entry_price) / entry_price <= -target:
                        # Saída com perda (stop loss)
                        trade_return = -(target+tax)
                        self.current_investment *= (1 + trade_return)
                        trades.append({
                            'entry_timestamp': entry_timestamp,
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': test_data['low'].iloc[i],
                            'return': trade_return,
                            'signal': entry_signal,
                            'success': False,
                            'exit_timestamp': test_data['open_time'].iloc[i],
                            'investment_value': self.current_investment
                        })
                        position_open = False
                elif entry_signal == 0:  # Posição de venda
                    if (entry_price - test_data['low'].iloc[i]) / entry_price >= target:
                        # Saída com sucesso (stop gain)
                        trade_return = (target-tax)
                        self.current_investment *= (1 + trade_return)
                        trades.append({
                            'entry_timestamp': entry_timestamp,
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': test_data['low'].iloc[i],
                            'return': trade_return,
                            'signal': entry_signal,
                            'success': True,
                            'exit_timestamp': test_data['open_time'].iloc[i],
                            'investment_value': self.current_investment
                        })
                        position_open = False
                    elif (entry_price - test_data['high'].iloc[i]) / entry_price <= -target:
                        # Saída com perda (stop loss)
                        trade_return = -(target + tax)
                        self.current_investment *= (1 + trade_return)
                        trades.append({
                            'entry_timestamp': entry_timestamp,
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': test_data['high'].iloc[i],
                            'return': trade_return,
                            'signal': entry_signal,
                            'success': False,
                            'exit_timestamp': test_data['open_time'].iloc[i],
                            'investment_value': self.current_investment
                        })
                        position_open = False
                        
        # Concatenar os novos trades ao DataFrame principal
        self.Simulation_real_time_df = pd.DataFrame(trades)
        return self.Simulation_real_time_df