import boto3

# Função para obter parâmetros do SSM Parameter Store
def get_ssm_parameter(name: str) -> str:
    ssm_client = boto3.client('ssm', region_name='pa-south-1')
    try:
        response = ssm_client.get_parameter(Name=name, WithDecryption=True)
        return response['Parameter']['Value']
    except ssm_client.exceptions.ParameterNotFound:
        print(" status_code=500 Parameter {name} not found in SSM Parameter Store")