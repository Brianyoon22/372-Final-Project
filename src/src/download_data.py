"""
Downloads the Sports Classification dataset from Kaggle.
Requires a Kaggle API token (~/.kaggle/kaggle.json).
See https://www.kaggle.com/docs/api for setup instructions.
"""
import os
import zipfile
import argparse

def download(dest="data/"):
    os.makedirs(dest, exist_ok=True)
    os.system(f"kaggle datasets download -d gpiosenka/sports-classification -p {dest}")
    zip_path = os.path.join(dest, "sports-classification.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        os.remove(zip_path)
        print(f"Dataset extracted to {dest}")
    else:
        print("Download failed. Check your Kaggle API token.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/", help="Destination directory")
    args = parser.parse_args()
    download(args.dest)
