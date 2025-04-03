import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
import joblib
import boto3
import os

class MLPClassificationModel:
    def __init__(self, ticker, bucket, num_classes):
        self.ticker = ticker
        self.bucket = bucket
        self.num_classes = num_classes
        self.subpasta_modelo = f'models/{ticker}/mlp'
        self.output_location = f's3://{bucket}/{self.subpasta_modelo}/output'
        self.scaler_key = f'models/{ticker}/scaler/{ticker}_mlp_scaler.pkl'
        self.model_path = os.path.join("modelosMLDP", "dataModels", "mlp_classification_model.pkl")
        self.scaler_path = os.path.join("modelosMLDP", "dataModels", "mlp_scaler.pkl")

    def train_mlp_classification(self, train_data):
        # Normalizar os dados
        # Separar features e target
        X_train = train_data.drop(columns=['signal'])
        y_train = train_data['signal'].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)

        # Salvar o scaler
        joblib.dump(scaler, self.scaler_path)

        # Calcular pesos das amostras para balancear classes
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        # Criar o modelo MLP ajustado
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),  # Camadas ajustadas
            max_iter=500,  # Aumenta as iterações
            activation='relu',  # Melhor para redes profundas
            solver='adam',
            random_state=3,
            validation_fraction=0.2,
            early_stopping=True,  # Para evitar overfitting
            n_iter_no_change=10
        )

        # Treinar o modelo com os pesos calculados
        model.fit(X_train_scaled, y_train)

        # Salvar o modelo treinado
        joblib.dump(model, self.model_path, protocol=4)
        
        # Fazer upload do modelo treinado e do scaler para o S3
        s3_client = boto3.client('s3')
        #s3_client.upload_file(self.model_path, self.bucket, f'{self.subpasta_modelo}/mlp_classification_model.pkl')
        s3_client.upload_file(self.scaler_path, self.bucket, self.scaler_key)
        
        return scaler

    def load_model_mlp_classification(self):
        # Carregar o modelo treinado
        model = joblib.load(self.model_path)
        return model

    def predict_mlp_classification(self, test_data, scaler):
        # Carregar o modelo treinado
        model = self.load_model_mlp_classification()

        # Normalizar os dados de teste
        scaled_data = scaler.transform(test_data)

        # Fazer predições de probabilidade (probabilidade da classe 1)
        probabilities = model.predict_proba(scaled_data)[:, 1]
        return probabilities

# Diferença entre MinMaxScaler e StandardScaler
# MinMaxScaler: Escalona os dados para um intervalo especificado (por padrão, entre 0 e 1).
# StandardScaler: Escalona os dados para que tenham média 0 e desvio padrão 1 (distribuição normal padrão).