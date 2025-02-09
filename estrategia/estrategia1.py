from estrategia.data_handler import DataHandler
from api.create_endpoint_sagemaker import SageMakerHandler
from estrategia.backtester import Backtester
from analytics.performance_models import PerformanceAnalyzer
import pandas as pd
import os

class Estrategia1 (DataHandler):
    def __init__(self, bucket_name, bucket_name_model, ticker, timeframe, period):
        self.type = 'classificação_xgboost'
        self.bucket_name = bucket_name
        self.bucket_name_model = bucket_name_model
        self.timeframe = timeframe
        self.period = period
        self.ticker = ticker
        
    async def setup(self):
        #self.df = await self.cria_data_frame(self.bucket_name,self.ticker, self.timeframe, self.period)
        # Caminho para salvar o arquivo Parquet na pasta raiz do projeto
        #parquet_file_path = os.path.join(os.getcwd(), 'data_frame_3A.parquet')
        
        # Salvar o DataFrame no formato Parquet
        #self.df.to_parquet(parquet_file_path, index=False)
        #print(f"DataFrame salvo em {parquet_file_path}")
        
        # Ler o arquivo Parquet e criar um novo DataFrame df2
        self.df = pd.read_parquet('data_frame_3A.parquet')
        print("DataFrame df2 criado a partir do arquivo Parquet")
        
        return self.df

    async def run_backtest(self):
        await self.setup()
        await self.prepara_dataset_s3(self.bucket_name_model, self.ticker, self.df)
        MLHandler = SageMakerHandler(self.bucket_name_model, self.ticker, role_arn='arn:aws:iam::324037302745:role/SageMakerExecutionRole')
        #MLHandler.upload_to_s3(f'{self.ticker}_train_xgboost.parquet')
        MLHandler.configure_estimator()
        MLHandler.train_model()
        MLHandler.deploy_model(endpoint_name=f'{self.ticker}-endpoint')
        """
        backtester = Backtester(self)
        #result = backtester.backtesting_optimized()
        Simulation_real_time_df = backtester.backtesting_real_time()
        Performer = PerformanceAnalyzer(Simulation_real_time_df, self)
        Performer.analyze_performance()
        """
        
        
        return None