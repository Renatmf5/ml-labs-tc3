import pandas as pd
from modelosMLDP import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from datetime import timedelta


class Backtester:
    def __init__(self, estrategia):
        self.estrategia = estrategia
        self.trades_df = pd.DataFrame()
        self.Simulation_real_time_df = pd.DataFrame()
        self.cumulative_returns = [0]  # Lista para armazenar o retorno acumulado de cada fatia
        self.initial_investment = 1000  # Valor inicial do investimento
        self.current_investment = self.initial_investment  # Valor atual do investimento
        
    def execute_model_trainer(self, train_data, test_data):
        if self.estrategia.type == 'classificação_xgboost':
           self.model = XGBoostModelTrainer()
           
           features_to_remove = ['timestamp', 'open', 'high', 'low', 'close']
           #,'bb_middle','cci', 'ichimoku_base','stoch_rsi_k','stoch_rsi_d','proportion_taker_BTC','z_score_BTC'
           train_data = train_data.drop(columns=features_to_remove)
           test_data = test_data.drop(columns=features_to_remove)
           feature_importance_df = self.model.train_model_classifier(train_data)
           
           """
           print(f"Top 10 features mais importantes: {feature_importance_df}")
           
           
           y_train = train_data['signal'].values
           y_test = test_data['signal'].values
           y_pred = self.model.predict_classifier(test_data)
           
           # Calcular a matriz de confusão
           cm = confusion_matrix(y_test, y_pred)
           print("Matriz de Confusão:")
           print(cm)
           
           
           # Calcular a acurácia geral
           accuracy = accuracy_score(y_test, y_pred)
           print(f"Acurácia Geral: {accuracy * 100:.2f}%")
           
            # Avaliar o desempenho nos dados de treinamento
           y_train_pred = self.model.predict_classifier(train_data)
           train_accuracy = accuracy_score(y_train, y_train_pred)
           print(f"Acurácia nos Dados de Treinamento: {train_accuracy * 100:.2f}%")
           
           # Calcular a acurácia para cada classe
           report = classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1', 'Classe 2'])
           print("Relatório de Classificação:")
           print(report)
           
           test_data['predicted_signal'] = y_pred
           """
           return test_data
           
    def backtesting_optimized(self):
        
        self.trades_df = self.estrategia.df
        self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
        self.trades_df = self.trades_df.sort_values(by='timestamp')
        self.trades_df = self.trades_df.dropna()

        # Definir o início e o fim do período de treinamento inicial (2 anos)
        start_date = self.trades_df['timestamp'].min()
        end_date_train = start_date + timedelta(days=2*365)
        
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
            test_data = self.execute_model_trainer(train_data, test_data)
            
            y_test = test_data['signal'].values
            y_pred = self.model.predict_classifier(test_data)
            # Obter o relatório de classificação como dicionário
            report_dict = classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1', 'Classe 2'], output_dict=True)
            
            # Extrair precisão e recall para cada classe
            precision0 = report_dict['Classe 0']['precision']
            recall0 = report_dict['Classe 0']['recall']
            precision1 = report_dict['Classe 1']['precision']
            recall1 = report_dict['Classe 1']['recall']
            precision2 = report_dict['Classe 2']['precision']
            recall2 = report_dict['Classe 2']['recall']
            
            # Armazenar os resultados
            results.append({
                'test_start': start_date_test,
                'test_end': end_date_test,
                'precision0': precision0,
                'recall0': recall0,
                'precision1': precision1,
                'recall1': recall1,
                'precision2': precision2,
                'recall2': recall2,
                'accuracy': accuracy_score(y_test, y_pred)
            })
            
            
            # Atualizar as datas para a próxima iteração
            end_date_train += timedelta(days=30)
            start_date_test += timedelta(days=30)
            end_date_test += timedelta(days=30)
        
        # Imprimir lista resultados em formato de tabela
        results_df = pd.DataFrame(results)
        print(results_df)
        
        
        return results
           
    def backtesting_real_time(self):

        self.trades_df = self.estrategia.df
        self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
        self.trades_df = self.trades_df.sort_values(by='timestamp')
        self.trades_df = self.trades_df.dropna()

        # Definir o início e o fim do período de treinamento inicial (2 anos)
        start_date = self.trades_df['timestamp'].min()
        end_date_train = start_date + timedelta(days=2*365)
        
        # Definir o início do período de teste (1 mês após o período de treinamento)
        start_date_test = end_date_train
        end_date_test = start_date_test + timedelta(days=30)
        
        trades = []
        position_open = False
        entry_price = 0
        entry_index = 0
        entry_timestamp = None
        tax = 0.0012
        
        for i in range(12):
            # Filtrar os dados de treinamento e teste
            train_data = self.trades_df[(self.trades_df['timestamp'] >= start_date) & (self.trades_df['timestamp'] < end_date_train)]
            test_data = self.trades_df[(self.trades_df['timestamp'] >= start_date_test) & (self.trades_df['timestamp'] < end_date_test)]
            

            if test_data.empty:
                break
            
            # Executar o treinamento e a predição
            result = self.execute_model_trainer(train_data, test_data)
            y_pred = self.model.predict_classifier(result)
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
                            
            Simulation_real_time_df = pd.DataFrame(trades)
            self.Simulation_real_time_df = pd.concat([self.Simulation_real_time_df, Simulation_real_time_df], ignore_index=True)
            
            # Atualizar as datas para a próxima iteração
            end_date_train += timedelta(days=30)
            start_date_test += timedelta(days=30)
            end_date_test += timedelta(days=30)
        
        
        self.Simulation_real_time_df    
        
        
        return self.Simulation_real_time_df