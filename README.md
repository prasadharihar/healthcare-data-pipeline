# Healthcare Data Pipeline 🏥

A modular Data Engineering pipeline built to optimize healthcare workforce management. 

## 🚀 Team Workflow (Getting Started)

When you clone this repository, everyone starts with an empty `jobs/` folder. Follow these steps to set up your local environment and begin your data engineering tasks:

### 1. Setup & Installation
```bash
# Clone and enter the repo
git clone <your-repo-url>
cd DE-use-case

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Your Local MinIO
Open `config/config.json` and update it with your own local MinIO credentials:
- **Endpoint**: Usually `localhost:9000`
- **Access Key / Secret Key**: Your local admin credentials.
- **Bucket Name**: Ensure the bucket exists in your MinIO.

### 3. Generate Local Data
Run this script once to create the raw `.parquet` files in your `generated_data/` folder:
```bash
python generate_data.py
```

### 4. Push to Bronze Layer
Run the ingestion pipeline to upload your raw data to your local MinIO:
```bash
python pipelines/bronze_ingestion.py
```

### 5. Create Your Transformation (Silver Job)
In the `jobs/` folder, create a new file for the table assigned to you (e.g., `leave_records_silver.py`). 

**Copy this template into your file:**
```python
import sys
import os

# Link to template
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates.silver_job_template import run_silver_job

if __name__ == "__main__":
    # Change table name and primary keys as needed
    run_silver_job("YOUR_TABLE_NAME", ["YOUR_PRIMARY_KEY"])
```

### 📁 Repository Structure
- `config/`: Configuration for MinIO and Bucket paths.
- `utils/`: Shared Spark and connection utilities.
- `templates/`: Boilerplate for consistent job development.
- `jobs/`: **Empty** (Create your specific ETL jobs here).
- `pipelines/`: Master pipelines for Ingestion and Cleaning.
- `generated_data/`: Local source of truth for raw data.
