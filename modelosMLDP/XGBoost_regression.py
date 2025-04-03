import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import boto3
import numpy as np


class XGBoostModelTrainer:
    def __init__(self, ticker, bucket):
        self.regressor_model = XGBRegressor(
            max_depth=5,  # Aumentar a profundidade máxima
            learning_rate=0.05,  # Ajustar a taxa de aprendizado
            n_estimators=500,  # Aumentar o número de estimadores
            objective='reg:squarederror',
            gamma=0,
            random_state=3
        )
        self.ticker = ticker
        self.bucket = bucket
        self.subpasta_modelo = f'models/{ticker}/xgboost'
        self.output_location = f's3://{bucket}/{self.subpasta_modelo}/output'
        self.scaler_key = f'models/{ticker}/scaler/{ticker}_xgboost_scaler.pkl'
        self.target_scaler_key = f'models/{ticker}/scaler/{ticker}_target_scaler.pkl'
        
    def train_model_regressor(self, train_data):
        X_train = train_data.drop(columns=['target'])
        Y_train = train_data[['target']]
        
        # Manter os nomes das colunas
        feature_names = X_train.columns
        
        # Escalonar dados de treino X
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Escalonar o target
        #target_scaler = StandardScaler()
        #Y_train = target_scaler.fit_transform(Y_train)
        
        # Treinar o modelo
        self.regressor_model.fit(X_train, Y_train.values.ravel())
        
        # Obter a importância das features
        feature_importances = self.regressor_model.feature_importances_
        
        # Criar um DataFrame com as importâncias das features
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        
        # Ordenar as features pela importância
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        # Salvar modelo treinado na pasta DataModels
        model_path = os.path.join("modelosMLDP", "dataModels", "xgboost_regressor_model.pkl")
        joblib.dump(self.regressor_model, model_path, protocol=4)
        
        # Configurar o logging

        # Fazer upload do modelo treinado para o S3
        s3_client = boto3.client('s3')
        #s3_client.upload_file(model_path, self.bucket, f'{self.subpasta_modelo}/xgboost_regressor_model.pkl')
        
        # Salvar e fazer upload do scaler para o S3
        scaler_path = os.path.join("modelosMLDP", "dataModels", f'{self.ticker}_xgboost_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        s3_client.upload_file(scaler_path, self.bucket, self.scaler_key)
        
        # Salvar e fazer upload do target scaler para o S3
        #target_scaler_path = os.path.join("modelosMLDP", "dataModels", f'{self.ticker}_target_scaler.pkl')
        #joblib.dump(target_scaler, target_scaler_path)
        #s3_client.upload_file(target_scaler_path, self.bucket, self.target_scaler_key)
        
        return scaler #, target_scaler
    
    def load_model_regressor(self):
        # Carregar o modelo treinado
        model_path = os.path.join("modelosMLDP", "dataModels", "xgboost_regressor_model.pkl")
        self.regressor_model = joblib.load(model_path)
        return self.regressor_model
    
    def load_scalers(self):
        # Carregar os scalers
        scaler_path = os.path.join("modelosMLDP", "dataModels", f'{self.ticker}_xgboost_scaler.pkl')
        target_scaler_path = os.path.join("modelosMLDP", "dataModels", f'{self.ticker}_target_scaler.pkl')
        scaler = joblib.load(scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        return scaler, target_scaler
        
    def predict_regressor(self, test_data, scaler, target_scaler=None):
        regressor_model = self.load_model_regressor()

        # Escalonar dados de teste X
        X_test = scaler.transform(test_data)
        
        # Predizer
        Y_pred = regressor_model.predict(X_test)
        
        # Aplicar a transformação inversa no target
        #Y_pred = target_scaler.inverse_transform(Y_pred.reshape(-1, 1))
        
        return Y_pred
    
    def predict_regressor_as_probability(self, test_data, scaler, alpha=1.0):
        # Carregar o modelo treinado
        regressor_model = self.load_model_regressor()

        # Escalonar dados de teste
        X_test = scaler.transform(test_data)
        
        # Predizer com o modelo regressor (saída contínua)
        Y_pred = regressor_model.predict(X_test)
        
        # Verificar a distribuição das predições
        print("Distribuição das predições:", np.histogram(Y_pred))
        
        # Converter a predição em "probabilidade" usando a função logística:
        # prob = 1 / (1 + exp(-alpha * Y_pred))
        probabilities = 1 / (1 + np.exp(-alpha * Y_pred))
        
        return probabilities