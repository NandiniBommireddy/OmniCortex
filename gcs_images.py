"""Abstraction for loading images from local filesystem or Google Cloud Storage.

Usage:
    image_root = ImageRoot.create("gs://mimic-cxr-jpg-2.1.0.physionet.org/files")
    # or
    image_root = ImageRoot.create("physionet.org/mimic-cxr-jpg/2.1.0/files")

    if image_root.exists("p10/p10000032/s50414267/dicom.jpg"):
        img = image_root.open_image("p10/p10000032/s50414267/dicom.jpg")
"""

import io
from pathlib import Path

from PIL import Image


class ImageRoot:
    @staticmethod
    def create(root: str) -> "ImageRoot":
        if root.startswith("gs://"):
            return _GCSImageRoot(root)
        return _LocalImageRoot(root)

    def exists(self, rel_path: str) -> bool:
        raise NotImplementedError

    def open_image(self, rel_path: str) -> Image.Image:
        raise NotImplementedError

    def path_str(self, rel_path: str) -> str:
        raise NotImplementedError

    @property
    def is_gcs(self) -> bool:
        return False


class _LocalImageRoot(ImageRoot):
    def __init__(self, root: str):
        self._root = Path(root)

    def exists(self, rel_path: str) -> bool:
        return (self._root / rel_path).exists()

    def open_image(self, rel_path: str) -> Image.Image:
        return Image.open(self._root / rel_path)

    def path_str(self, rel_path: str) -> str:
        return str(self._root / rel_path)


class _GCSImageRoot(ImageRoot):
    def __init__(self, root: str):
        from google.cloud import storage

        trimmed = root[5:]  # strip "gs://"
        parts = trimmed.split("/", 1)
        self._bucket_name = parts[0]
        self._prefix = (parts[1].rstrip("/") + "/") if len(parts) > 1 else ""
        print(f"[GCS] connecting to gs://{self._bucket_name}/{self._prefix} ...")
        self._client = storage.Client(project="885253748539")
        self._bucket = self._client.bucket(self._bucket_name)
        self._blob_set = None  # lazy-loaded cache

    def _blob_name(self, rel_path: str) -> str:
        return self._prefix + rel_path

    def _ensure_blob_set(self):
        """List all blobs once and cache — avoids one HTTP call per exists()."""
        if self._blob_set is not None:
            return
        print(f"[GCS] listing blobs under gs://{self._bucket_name}/{self._prefix} (one-time) ...")
        self._blob_set = set()
        for blob in self._client.list_blobs(self._bucket, prefix=self._prefix):
            self._blob_set.add(blob.name)
        print(f"[GCS] found {len(self._blob_set)} blobs")

    def exists(self, rel_path: str) -> bool:
        self._ensure_blob_set()
        return self._blob_name(rel_path) in self._blob_set

    def open_image(self, rel_path: str) -> Image.Image:
        blob = self._bucket.blob(self._blob_name(rel_path))
        data = blob.download_as_bytes()
        return Image.open(io.BytesIO(data))

    def path_str(self, rel_path: str) -> str:
        return f"gs://{self._bucket_name}/{self._blob_name(rel_path)}"

    @property
    def is_gcs(self) -> bool:
        return True
