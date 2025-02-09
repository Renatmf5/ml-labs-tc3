import boto3
import os
import sys
import pandas as pd
import io

s3_client = boto3.client('s3')

async def read_from_s3(bucket_name,ticker, timeframe, period) -> str:
    partition = f'ticker={ticker}/timeframe={timeframe}/period={period}/'
    data_path = f"{timeframe}_{ticker}_{period}.parquet"
    s3_key = f"Refined/{partition}{os.path.basename(data_path)}"
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        body = response['Body'].read()
        return pd.read_parquet(io.BytesIO(body))
    except s3_client.exceptions.NoSuchKey:
        print(f" status_code=500 Key {s3_key} not found in S3 bucket {bucket_name}")
        sys.exit(1)