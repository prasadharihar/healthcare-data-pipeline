import json
import os
from pyspark.sql import SparkSession

def get_config():
    # Helper to find config.json in the config folder
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "config", "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def get_spark_session(app_name="HealthcarePipeline"):
    config = get_config()
    minio_cfg = config['minio']
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{minio_cfg['endpoint']}") \
        .config("spark.hadoop.fs.s3a.access.key", minio_cfg['access_key']) \
        .config("spark.hadoop.fs.s3a.secret.key", minio_cfg['secret_key']) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    return spark, config
