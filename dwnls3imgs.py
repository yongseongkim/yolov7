import json
import subprocess
from pathlib import Path

import boto3


def get_vault_session(aws_profile):
    aws_env_json = subprocess.check_output(['aws-vault', 'export', '--format=json', aws_profile])
    aws_env = json.loads(aws_env_json)
    return boto3.client(
        's3',
        aws_access_key_id=aws_env['AccessKeyId'],
        aws_secret_access_key=aws_env['SecretAccessKey'],
        aws_session_token=aws_env['SessionToken'],
    )


def get_public_url(bucket_name, key):
    return f'https://{bucket_name}.s3.amazonaws.com/{key}'


def get_object_keys(bucket_name):
    s3_client = get_vault_session('scc')
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name)
    obj_urls = []
    for page in page_iterator:
        objects = page['Contents']
        print(f'{len(objects)} objects found.')
        if objects:
            for obj in objects:
                key = obj['Key']
                url = get_public_url(bucket_name, key)
                obj_urls.append(url)
    print(f'Totally {len(obj_urls)} objects found.', end='\n')
    return obj_urls


def download_obj(s3_client, bucket_name, key, output_dir):
    output_dir = Path(output_dir)
    downloaded_path = f'{output_dir}/{key}'
    print(f'Downloading {key} to {downloaded_path}.')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket_name, key, downloaded_path)
    print(f'{key} downloaded.')
    return downloaded_path


if __name__ == '__main__':
    urls = get_object_keys('scc-prod-accessibility-images')
    print(urls)
