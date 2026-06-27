# src/step1_download_data.py
# This script downloads the Google ASL Signs dataset using the Kaggle API.
# It extracts the downloaded zip file and verifies the raw data directories.

import os
import zipfile

def download_asl_dataset(path='data/raw'):
    # Import kaggle API inside the function to avoid errors if packages aren't loaded yet
    from kaggle.api.kaggle_api_extended import KaggleApi
    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()
    print("Downloading asl-signs competition dataset to:", path)
    # Download all files for the competition to the specified folder
    api.competition_download_files('asl-signs', path=path)
    print("Download completed successfully!")

def extract_zip(zip_file, extract_to):
    print(f"Extracting {zip_file} to {extract_to}...")
    # Open the downloaded zip archive
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all files
        zip_ref.extractall(extract_to)
    print("Extraction completed!")
    # Delete the zip file to save disk space
    os.remove(zip_file)
    print("Temporary zip file cleaned up.")

def verify_dataset(extract_to):
    # Verify train.csv exists
    csv_path = os.path.join(extract_to, 'train.csv')
    assert os.path.exists(csv_path), "Missing train.csv!"
    # Verify train_landmark_files directory exists
    landmark_path = os.path.join(extract_to, 'train_landmark_files')
    assert os.path.exists(landmark_path), "Missing train_landmark_files directory!"
    # Print the verification summary
    print(f"Verification OK! CSV located at {csv_path}")
    print(f"Landmark files located in {landmark_path}")

if __name__ == '__main__':
    # Define paths
    raw_dir = 'data/raw'
    zip_path = os.path.join(raw_dir, 'asl-signs.zip')
    # Run the workflow
    try:
        download_asl_dataset(raw_dir)
        extract_zip(zip_path, raw_dir)
        verify_dataset(raw_dir)
        print("Kaggle dataset downloaded and verified successfully!")
    except Exception as error:
        print("An error occurred during dataset setup:", error)
        print("Please ensure your kaggle.json token is configured in ~/.kaggle/kaggle.json")
