from minio import Minio

client = Minio(
    endpoint="localhost:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)

bucket_name = ""

if client.bucket_exists(bucket_name=bucket_name):
    client.remove_bucket(bucket_name=bucket_name)
    print("Bucket deleted")