"""
Raw-data lake abstraction.

Production pipelines land raw payloads in MinIO (S3 API). Keys are
deterministic (source/entity/partition), so re-running a task for the same
partition overwrites the same object instead of creating duplicates.
``STORAGE_BACKEND=local`` writes to a directory instead, which is used by
tests and environments without MinIO.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import pandas as pd

from common.config import get_settings


class ObjectStore:
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def list_keys(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def put_parquet(self, key: str, frame: pd.DataFrame) -> str:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, engine="pyarrow", index=False)
        return self.put_bytes(key, buffer.getvalue(), "application/vnd.apache.parquet")

    def get_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.get_bytes(key)), engine="pyarrow")


class S3ObjectStore(ObjectStore):
    def __init__(self) -> None:
        import boto3
        from botocore.config import Config

        settings = get_settings()
        self.bucket = settings.minio_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=10, read_timeout=60),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return f"s3://{self.bucket}/{key}"

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(keys)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or get_settings().local_lake_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"Invalid object key: {key}")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return f"file://{path}"

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        base = self.root
        return sorted(
            str(path.relative_to(base)).replace("\\", "/")
            for path in base.rglob("*")
            if path.is_file() and str(path.relative_to(base)).replace("\\", "/").startswith(prefix)
        )

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    if get_settings().storage_backend == "local":
        return LocalObjectStore()
    return S3ObjectStore()
