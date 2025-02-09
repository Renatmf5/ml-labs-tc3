import joblib
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class XGBoostModelTrainer:
    def __init__(self):
        self.classifier_model = XGBClassifier(
            max_depth=5,
            learning_rate=0.01,
            n_estimators=100,
            objective='multi:softmax',
            num_class=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=1,
            reg_lambda=1,
            random_state=3
        )
        
    def train_model_classifier(self, train_data):


        X_train = train_data.drop(columns=['signal'])
        Y_train = train_data[['signal']]
        
        # Manter os nomes das colunas
        feature_names = X_train.columns
        
        #escalonar dados de treino X
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        """
        # Definir os hiperparâmetros para o Grid Search
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 500],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.01, 0.1, 1]
        }
        
        
         # Grid Search sem validação cruzada
        grid_search = GridSearchCV(estimator=self.classifier_model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=2)
        grid_search.fit(X_train, Y_train.values.ravel())
        
        # Melhor modelo encontrado pelo Grid Search
        self.classifier_model = grid_search.best_estimator_
        
        print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
        
        resultado bom:
        
        Matriz de Confusão:
        [[1466  162  260]
        [ 145 1598  249]
        [ 350  177  874]]
        Acurácia Geral: 74.57%
        Iniciando a predição...
        Predição concluída.
        Acurácia nos Dados de Treinamento: 78.51%
        Relatório de Classificação:
                    precision    recall  f1-score   support

            Classe 0       0.75      0.78      0.76      1888
            Classe 1       0.82      0.80      0.81      1992
            Classe 2       0.63      0.62      0.63      1401

            accuracy                           0.75      5281
        macro avg       0.73      0.73      0.73      5281
        weighted avg       0.75      0.75      0.75      5281
        
        parametros usados :
        
        max_depth=5,
            learning_rate=0.01,
            n_estimators=100,
            objective='multi:softmax',
            num_class=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=1,
            reg_lambda=1,
            random_state=3
        
        """
        
        
        #Treinar o modelo
        #print("Iniciando o treinamento do modelo...")
        self.classifier_model.fit(X_train, Y_train.values.ravel())
        #print("Treinamento concluído.")
        
        # Obter a importância das features
        feature_importances = self.classifier_model.feature_importances_
        
        # Criar um DataFrame com as importâncias das features
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        
         # Ordenar as features pela importância
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        # Salvar modelo treinado na pasta DataModels
        model_path = os.path.join("modelosMLDP","dataModels", "xgboost_classifier_model.pkl")
        joblib.dump(self.classifier_model, model_path)
        
        
        return feature_importance_df
    
    def load_model_classifier(self):
        # Carregar o modelo treinado
        model_path = os.path.join("modelosMLDP","dataModels", "xgboost_classifier_model.pkl")
        self.classifier_model = joblib.load(model_path)
        return self.classifier_model
        
    def predict_classifier(self, test_data):

        classifier_model = self.load_model_classifier()
        
        X_test = test_data.drop(columns=['signal'])
        
        #escalonar dados de teste X
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)
        
        #Predizer
        #print("Iniciando a predição...")
        Y_pred = classifier_model.predict(X_test)
        #print("Predição concluída.")
        
        return Y_pred