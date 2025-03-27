import numpy as np
import joblib
import boto3
import os
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class StackingEnsembleModel:
    def __init__(self, bucket_name, ticker, model_lstm=None, model_xgb=None, model_mlp=None, scaler_lstm=None, scaler_xgb=None, scaler_mlp=None):
        self.bucket_name = bucket_name
        self.ticker = ticker
        self.model_lstm = model_lstm
        self.model_xgb = model_xgb
        self.model_mlp = model_mlp
        self.scaler_lstm = scaler_lstm
        self.scaler_xgb = scaler_xgb
        self.scaler_mlp = scaler_mlp
        self.subpasta_modelo = f'models/{ticker}/ensemble'
        self.subpasta_scaler = f'models/{ticker}/scaler'
        self.meta_model = self.build_meta_model()

    def build_meta_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def salvar_scaler_ensemble_no_s3(self, key, scaler_emsemble):
        # Salvar e fazer upload do scaler para o S3
        s3_client = boto3.client('s3')
        scaler_path = os.path.join("modelosMLDP", "dataModels", "ensemble_scaler.pkl")
        joblib.dump(scaler_emsemble, scaler_path)
        s3_client.upload_file(scaler_path, self.bucket_name, key)
    
    def salvar_meta_modelo_no_s3(self, key):

        s3 = boto3.client('s3')
        try:
            # Criar o diretório local, se não existir
            local_dir = os.path.join("modelosMLDP", "dataModels")
            os.makedirs(local_dir, exist_ok=True)

            # Caminho local para salvar o modelo
            model_path = os.path.join(local_dir, "ensemble_model.keras")

            # Salvar o modelo localmente no formato .keras
            self.meta_model.save(model_path)

            s3.upload_file(model_path, self.bucket_name, key)
            print(f"Meta-modelo salvo com sucesso no S3 em: {key}")
        except Exception as e:
            print(f"Erro ao salvar o meta-modelo no S3: {e}")

    def gerar_meta_features(self, data_run_lstm, data_run_xgb, data_run_mlp, y_true):
        # Remover a coluna 'signal' dos dados de teste
        data_run_lstm = data_run_lstm.drop(columns=['signal'])
        data_run_mlp = data_run_mlp.drop(columns=['signal'])
        data_run_xgb = data_run_xgb.drop(columns=['target'])
        
        # Previsões individuais:
        pred_lstm_proba = self.model_lstm.predict_lstm_classification(data_run_lstm, self.scaler_lstm)
        pred_xgb_proba = self.model_xgb.predict_regressor_as_probability(data_run_xgb, self.scaler_xgb, alpha=1.0)
        pred_mlp_proba = self.model_mlp.predict_mlp_classification(data_run_mlp, self.scaler_mlp)
        
        # Garantir que todos os arrays tenham o mesmo tamanho (usar o menor tamanho)
        min_length = min(len(pred_lstm_proba), len(pred_xgb_proba), len(pred_mlp_proba))
        pred_lstm_proba = pred_lstm_proba[:min_length]
        pred_xgb_proba = pred_xgb_proba[:min_length]
        pred_mlp_proba = pred_mlp_proba[:min_length]
        
        # Ajustar y_true para o mesmo tamanho
        y_true = y_true[:min_length]
        
        # Imprimir relatórios (convertendo probabilidades para classes com threshold 0.5 para fins de avaliação)
        print("Relatório de Classificação para LSTM:")
        print(classification_report(y_true, (pred_lstm_proba > 0.5).astype(int)))
        
        print("Relatório de Classificação para XGBoost:")
        print(classification_report(y_true, (pred_xgb_proba > 0.5).astype(int)))
        
        print("Relatório de Classificação para MLP:")
        print(classification_report(y_true, (pred_mlp_proba > 0.5).astype(int)))
        
        # Concatenar as previsões como features para o meta-modelo
        meta_features = np.column_stack((pred_lstm_proba, pred_xgb_proba, pred_mlp_proba))
        
        return meta_features, min_length

    def treinar_meta_modelo(self, X_train_lstm, X_train_xgb, X_train_mlp, y_train):
        meta_X_train, min_length = self.gerar_meta_features(X_train_lstm, X_train_xgb, X_train_mlp, y_train)
        scaler_ensemble = StandardScaler()
        meta_features_standardized = scaler_ensemble.fit_transform(meta_X_train)
        self.meta_model.fit(meta_features_standardized, y_train[:min_length], epochs=5, batch_size=32, validation_split=0.2)
        self.salvar_scaler_ensemble_no_s3(f'{self.subpasta_scaler}/{self.ticker}_ensemble_scaler.pkl', scaler_ensemble)
        self.salvar_meta_modelo_no_s3(f'{self.subpasta_modelo}/ensemble_model.keras')
        return meta_X_train, min_length, scaler_ensemble

    def prever_meta_modelo(self, X_test_lstm, X_test_xgb, X_test_mlp, y_test, scaler_ensemble):
        model_path = os.path.join("modelosMLDP", "dataModels", "ensemble_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError("O modelo Ensemble não foi encontrado. Por favor, treine o modelo antes de fazer previsões.")
        self.meta_model = load_model(model_path)
        meta_X_test, min_length = self.gerar_meta_features(X_test_lstm, X_test_xgb, X_test_mlp, y_test)
        meta_X_test_scaled = scaler_ensemble.transform(meta_X_test)
        meta_predictions_proba = self.meta_model.predict(meta_X_test_scaled)
        meta_predictions = (meta_predictions_proba > 0.5).astype(int)
        return meta_X_test, meta_predictions, meta_predictions_proba, min_length

    def avaliar_meta_modelo(self, y_test, meta_predictions, min_length):
        y_test = y_test[:min_length]
        accuracy = accuracy_score(y_test, meta_predictions)
        precision = precision_score(y_test, meta_predictions)
        recall = recall_score(y_test, meta_predictions)
        f1 = f1_score(y_test, meta_predictions)
        cm = confusion_matrix(y_test, meta_predictions)
        print("Relatório de Classificação para Ensemble:")
        print(classification_report(y_test, meta_predictions))
        return accuracy, precision, recall, f1, cm

