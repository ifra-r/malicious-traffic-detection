import kaggle
import os

def download_csic2010():
    dataset = 'ispangler/csic-2010-web-application-attacks'
    target_path = 'data/'

    print(f"Starting download of dataset {dataset} into '{target_path}'...")

    # Ensure the target directory exists
    os.makedirs(target_path, exist_ok=True)

    # Download and unzip
    kaggle.api.dataset_download_files(dataset, path=target_path, unzip=True)

    print("Download complete and files extracted.")

if __name__ == '__main__':
    download_csic2010()
