import os
import boto3
import subprocess

access_key = 'PRQNSBSZLPSHMUW1BUEH'
secret_key = 'ot0ALtwR77Ds9vBdSMJTBaQUxuP6YpHNO88roiCb'
bucket_name = 'machinlearning-ai'
region = 'eu-west-0'  # For example, 'us-east-1'
endpoint = 'https://oss.prod-cloud-ocb.orange-business.com'  # Specify your S3 endpoint
download_directory = '/home/cloud/tenant_names'  # Change to your desired download location


# Initialize the S3 client with your specified parameters
s3 = boto3.client(
    's3',
    aws_access_key_id=access_key,  # Replace with your AWS access key
    aws_secret_access_key=secret_key,  # Replace with your AWS secret key
    region_name=region,  # Replace with your AWS region
    endpoint_url=endpoint  # Replace with your S3 endpoint if needed (for example, for custom S3 services)
)

# Define the path to your local model file
local_model_path = '/app/models/cats_vs_dogs_model.keras'  # Replace with your actual model file path
s3_model_path = 'models/cats_vs_dogs_model.keras'  # The path where the model will be saved in S3

# Upload the model file to S3
try:
    print(f"Uploading {local_model_path} to s3://{bucket_name}/{s3_model_path}...")
    s3.upload_file(local_model_path, bucket_name, s3_model_path)
    print(f"Model successfully uploaded to S3: s3://{bucket_name}/{s3_model_path}")
except Exception as e:
    print(f"Error uploading file: {e}")

# Optionally, list objects in the S3 bucket (to verify upload)
response = s3.list_objects(Bucket=bucket_name)
if 'Contents' in response:
    for obj in response['Contents']:
        print(f"Found object: {obj['Key']}")

# Define the path to your Bash script (if needed for further processing)
#bash_script = download_directory + "/cleanfiles.sh"
#print(f"Executing Bash script: {bash_script}")

# Execute the Bash script
#try:
#    subprocess.run(["bash", bash_script], check=True)
#    print("Bash script executed successfully.")
#except subprocess.CalledProcessError as e:
#    print(f"Error executing the Bash script: {e}")
