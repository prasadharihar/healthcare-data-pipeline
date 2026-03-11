import sys
import os
from pyspark.sql.functions import current_timestamp, col, to_date

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.spark_config import get_spark_session

def run_silver_job(table_name, primary_keys):
    """
    Standard template for Silver layer jobs.
    Everyone in the team should use this function to ensure consistency.
    """
    spark, config = get_spark_session(f"Silver_{table_name}")
    bucket = config['lakehouse']['bucket_name']
    
    bronze_path = f"s3a://{bucket}/{config['lakehouse']['bronze_path']}/{table_name}.parquet"
    silver_path = f"s3a://{bucket}/{config['lakehouse']['silver_path']}/{table_name}"
    
    print(f"--- Starting Silver Job for: {table_name} ---")
    
    # 1. Read
    print(f"Reading from Bronze: {bronze_path}")
    df = spark.read.parquet(bronze_path)
    
    # 2. Transformations (Cleaning)
    print("Applying Cleaning & Deduplication...")
    df = df.dropDuplicates(primary_keys)
    df = df.withColumn("silver_processed_at", current_timestamp())
    
    # 3. Write as Delta
    print(f"Saving as Delta Table to: {silver_path}")
    df.write.format("delta").mode("overwrite").save(silver_path)
    
    print(f"--- Success: {table_name} is now in Silver ---")

if __name__ == "__main__":
    print("This is a template and should be imported by scripts in the /jobs folder.")
