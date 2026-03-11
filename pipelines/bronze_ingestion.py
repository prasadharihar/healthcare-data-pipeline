from minio import Minio
import os
import json

# Load Configuration
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_path, "config.json")

with open(config_path, 'r') as f:
    config = json.load(f)

# MinIO connection
client = Minio(
    endpoint=config['minio']['endpoint'],
    access_key=config['minio']['access_key'],
    secret_key=config['minio']['secret_key'],
    secure=config['minio']['secure']
)

bucket_name = config['lakehouse']['bucket_name']
local_folder = config['local']['data_source']

# Path correction for local folder since we are in /pipelines
local_folder_path = os.path.join(base_path, local_folder)

for file in os.listdir(local_folder_path):
    file_path = os.path.join(local_folder_path, file)

    if file.endswith(".parquet"):
        # Uploading to the bronze folder in the bucket
        object_path = f"{config['lakehouse']['bronze_path']}/{file}"

        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_path,
            file_path=file_path
        )

        print(f"Uploaded → {object_path}")

print("All Bronze files uploaded to MinIO Landing Zone")