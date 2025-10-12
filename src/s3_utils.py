import boto3

from src.config_loader import load_config


config = load_config()
bucket_name = config['s3']['bucket_name']

def get_s3_client():
    return boto3.client('s3')

def list_s3_objects(bucket_name):
    """
    Возвращает список объектов в бакете S3.
    """
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name)
    return [obj['Key'] for obj in response.get('Contents', [])]

def download_file_from_s3(bucket_name, s3_key, local_path):
    """
    Скачивает файл из S3 в локальную папку.
    """
    s3 = get_s3_client()
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")
