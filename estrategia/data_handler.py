import pandas as pd
import api.read_from_s3 as read_from_s3
from sklearn.preprocessing import StandardScaler
import boto3
import joblib

class DataHandler:
    async def cria_data_frame(self, bucket_name,ticker, timeframe, period):
        df = await read_from_s3(bucket_name,ticker, timeframe, period)
        return df
    
    async def prepara_dataset_s3(self, bucket_ml, ticker, df):
        
       features_to_remove = ['timestamp', 'open', 'high', 'low', 'close']
       train_data = df.drop(columns=features_to_remove)
       
       # movimentar a coluna signal para a primeira posição
       signal = train_data['signal']
       train_data.drop(columns=['signal'], inplace=True)
       
       scaler = StandardScaler()
       train_data_scaled = scaler.fit_transform(train_data)
       
       
       # Converter de volta para DataFrame
       train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.columns)
    
       # Inserir a coluna signal na primeira posição
       train_data_scaled.insert(0, 'signal', signal.values)
                     
       # Salvar os parâmetros de escalonamento
       scaler_file_path = f'/tmp/{ticker}_scaler.pkl'
       joblib.dump(scaler, scaler_file_path)
       
       # Salvar o dataset localmente
       key_train = f'{ticker}_train_xgboost.csv'
       local_file_path = f'/tmp/{key_train}'
       train_data_scaled.to_csv(local_file_path, index=False, header=False)
        
       # Upload para o S3
       s3 = boto3.client('s3')
       s3.upload_file(Filename=local_file_path, Bucket=bucket_ml, Key=f'datasets/{ticker}/{key_train}') 
       
       # Upload dos parâmetros de escalonamento para o S3
       s3.upload_file(Filename=scaler_file_path, Bucket=bucket_ml, Key=f'models/{ticker}/scaler/{ticker}_scaler.pkl')     
              
       return True
       
               
        