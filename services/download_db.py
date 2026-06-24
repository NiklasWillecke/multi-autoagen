import os
from pathlib import Path

import boto3

ACCOUNT_ID = "55f0018c4d4676776ce3574929cba20a"
BUCKET_NAME = "movies-chromadb"
PREFIX = "croma-data/"
LOCAL_DIR = "./croma-data"


class DownloadDB:
    def __init__(self):
        self.s3 = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id="DEIN_ACCESS_KEY_ID",
            aws_secret_access_key="DEIN_SECRET_ACCESS_KEY",
            region_name="auto",
        )

    def check_if_exists(self) -> bool:

        ordner = Path(PREFIX)

        if ordner.exists() and ordner.is_dir():
            return True
        else:
            return False

    def download(self):
        if self.check_if_exists:
            return

        paginator = self.s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)

        os.makedirs(LOCAL_DIR, exist_ok=True)
        downloaded = 0

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                local_path = os.path.join(LOCAL_DIR, os.path.relpath(key, PREFIX))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                print(f"Downloading: {key}")
                self.s3.download_file(BUCKET_NAME, key, local_path)
                downloaded += 1

        print(f"Fertig. {downloaded} Dateien nach '{LOCAL_DIR}' heruntergeladen.")
