import boto3
import psycopg2
from psycopg2 import pool
import json

class Database:
    def __init__(self):
        self.secret_name = "RDSPostgresCredentials"
        self.region_name = "ap-south-1"
        self.connection_pool = None
        self.initialize_pool()

    def get_secret(self):
        # Create a Secrets Manager client
        client = boto3.client('secretsmanager', region_name=self.region_name)

        try:
            get_secret_value_response = client.get_secret_value(SecretId=self.secret_name)
        except Exception as e:
            raise e

        # Decrypts secret using the associated KMS key.
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)

    def initialize_pool(self):
        secret = self.get_secret()
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # minconn, maxconn
            user=secret['username'],
            password=secret['password'],
            host=secret['host'],
            port=secret['port'],
            database=secret['dbname']
        )

    def get_connection(self):
        if self.connection_pool:
            return self.connection_pool.getconn()

    def release_connection(self, connection):
        if self.connection_pool:
            self.connection_pool.putconn(connection)

    def close_all_connections(self):
        if self.connection_pool:
            self.connection_pool.closeall()

# Singleton instance of the Database class
db_instance = Database()