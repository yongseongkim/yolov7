import json
import subprocess

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
                # s3_client.download_file(bucket_name, key, './tmp/{}'.format(key))
        break
    print(f'Totally {len(obj_urls)} objects found.', end='\n')
    return obj_urls


if __name__ == '__main__':
    urls = get_object_keys('scc-prod-accessibility-images')
    print(urls)
