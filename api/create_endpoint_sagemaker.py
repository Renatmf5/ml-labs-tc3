import boto3
import sagemaker
from sagemaker import Session
import os

class SageMakerHandler:
    def __init__(self, bucket, ticker, role_arn):
        self.session = sagemaker.Session()
        self.bucket = bucket
        self.ticker = ticker
        self.role = role_arn
        self.subpasta_modelo = f'models/{ticker}/xgboost'
        self.subpasta_dataset = f'datasets/{ticker}'
        self.key_train = f'{ticker}_train_xgboost.csv'
        self.s3_train_data = f's3://{bucket}/{self.subpasta_dataset}/{self.key_train}'
        self.output_location = f's3://{bucket}/{self.subpasta_modelo}/output'
        self.container = sagemaker.image_uris.retrieve(framework='xgboost', region=boto3.Session().region_name, version='latest')
        self.xgboost = None
        self.endpoint_name = None

    def configure_estimator(self):
        self.xgboost = sagemaker.estimator.Estimator(
            image_uri=self.container,
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.2xlarge',
            output_path=self.output_location,
            sagemaker_session=self.session
        )
        self.xgboost.set_hyperparameters(
            max_depth=5,
            learning_rate=0.01,
            num_round=100,
            objective='multi:softmax',
            num_class=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=1,
            reg_lambda=1,
            random_state=3
        )

    def train_model(self):
        train_input = sagemaker.inputs.TrainingInput(s3_data=self.s3_train_data, content_type='text/csv', s3_data_type='S3Prefix')
        data_channels = {'train': train_input}
        self.xgboost.fit(data_channels)

    def deploy_model(self, endpoint_name=None):
        if endpoint_name is None:
            endpoint_name = f'{self.ticker}-endpoint'
            
        # Excluir a configuração do endpoint existente, se houver
        sagemaker_client = boto3.client('sagemaker')
        try:
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except sagemaker_client.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print(f'Endpoint configuration {endpoint_name} does not exist or has already been deleted.')
            else:
                raise
        
        # Implantar o modelo
        self.xgboost_regressor = self.xgboost.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge', endpoint_name=endpoint_name)
        self.endpoint_name = self.xgboost_regressor.endpoint_name

    def predict(self, data):
        runtime = boto3.client('runtime.sagemaker')
        response = runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='text/csv',
            Body=data
        )
        result = response['Body'].read().decode('utf-8')
        return result
    
    
