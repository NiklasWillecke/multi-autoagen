import os
from pathlib import Path

import boto3
from botocore.config import Config

# --- Konfiguration (per Env-Variable ueberschreibbar) ---
ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "55f0018c4d4676776ce3574929cba20a")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "movies-chromadb")
PREFIX = os.getenv("R2_PREFIX", "croma-data/")
LOCAL_DIR = os.getenv("LOCAL_DB_DIR", "./croma-data")


class DownloadDBError(Exception):
    """Wird geworfen, wenn der Download nicht moeglich ist."""


class DownloadDB:
    def __init__(self):
        access_key = os.getenv("Access_Key_ID")
        secret_key = os.getenv("Secret_Access_Key")

        missing = [
            name
            for name, value in (
                ("Access_Key_ID", access_key),
                ("Secret_Access_Key", secret_key),
            )
            if not value
        ]
        if missing:
            raise DownloadDBError(f"Fehlende Env-Variablen: {', '.join(missing)}")

        self.s3 = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(retries={"max_attempts": 5, "mode": "standard"}),
        )

    def check_if_exists(self) -> bool:
        """True nur, wenn der Ordner existiert UND nicht leer ist."""
        ordner = Path(LOCAL_DIR)
        if ordner.exists() and ordner.is_dir() and any(ordner.iterdir()):
            print("DB bereits vorhanden")
            return True
        return False

    def _resolve_prefix(self) -> str:
        """
        Findet den tatsaechlich vorhandenen Prefix.
        Probiert den konfigurierten Prefix und gaengige Schreibvarianten
        (croma/chroma, mit/ohne Slash, Bucket-Root).
        """
        candidates = [
            PREFIX,
            PREFIX.replace("croma", "chroma"),
            PREFIX.replace("chroma", "croma"),
            PREFIX.rstrip("/") + "/",
            "chroma-data/",
            "croma-data/",
            "",  # Bucket-Root
        ]

        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)

            resp = self.s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=cand, MaxKeys=1)
            if resp.get("KeyCount", 0) > 0:
                if cand != PREFIX:
                    print(f"Hinweis: Prefix '{PREFIX}' leer - nutze '{cand}'")
                return cand

        # Nichts gefunden -> zur Diagnose auflisten, was es WIRKLICH gibt
        self._diagnose()
        raise DownloadDBError(
            f"Keine Objekte im Bucket '{BUCKET_NAME}' unter den "
            f"geprueften Prefixen gefunden."
        )

    def _diagnose(self):
        """Listet Buckets und die ersten Keys zur Fehlersuche."""
        try:
            buckets = [b["Name"] for b in self.s3.list_buckets().get("Buckets", [])]
            print("Verfuegbare Buckets:", buckets)
        except Exception as exc:  # noqa: BLE001
            print("Konnte Buckets nicht auflisten:", exc)

        resp = self.s3.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=30)
        print("KeyCount (ohne Prefix):", resp.get("KeyCount", 0))
        for obj in resp.get("Contents", []):
            print("KEY:", obj["Key"])

    def download(self):
        if self.check_if_exists():
            return

        prefix = self._resolve_prefix()

        paginator = self.s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

        os.makedirs(LOCAL_DIR, exist_ok=True)
        downloaded = 0

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # "Ordner"-Marker (enden auf /) ueberspringen
                if key.endswith("/"):
                    continue

                # Zielpfad relativ zum erkannten Prefix aufbauen
                rel_path = os.path.relpath(key, prefix) if prefix else key
                local_path = os.path.join(LOCAL_DIR, rel_path)

                parent = os.path.dirname(local_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

                print(f"Downloading: {key} -> {local_path}")
                self.s3.download_file(BUCKET_NAME, key, local_path)
                downloaded += 1

        if downloaded == 0:
            print(f"WARNUNG: Keine Dateien unter Prefix '{prefix}' heruntergeladen.")
        else:
            print(f"Fertig. {downloaded} Dateien nach '{LOCAL_DIR}' heruntergeladen.")
