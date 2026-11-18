"""Fetch the thesis datasets from Cloudflare R2 into datasets/ when they are missing.

    from project_argus.datasets import fetch_dataset
    path = fetch_dataset("DS_Augmented_v2.csv")

To download everything that's missing:

    uv run python -m project_argus.datasets

Files already in datasets/ are returned with no network call and no credentials.
Settings come from environment variables or a .env file; see .env.example.
"""

import hashlib
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import boto3
from boto3.exceptions import Boto3Error
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
ENV_FILE = ROOT / ".env"
DATASETS = ("DS_Base.csv", "DS_Augmented_v1.csv", "DS_Augmented_v2.csv")
REQUIRED_ENV = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
)

HASH_METADATA_KEY = "sha256"  # Set on each object when it's uploaded.
HASH_CHUNK_BYTES = 2**20
PART_BYTES = 64 * 2**20
MAX_CONCURRENCY = 4
ACCOUNT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
BAR_WIDTH = 30
BYTES_PER_MB = 10**6


class DatasetError(Exception):
    """A dataset could not be fetched or failed verification."""


class _ProgressBar:
    """Draw download progress on one terminal line; boto3 calls this from several threads."""

    def __init__(self, name: str, total: int, stream=sys.stderr) -> None:
        self.name, self.total, self.stream = name, total, stream
        self.done = 0
        self.last_percent = -1
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def __call__(self, count: int) -> None:
        with self.lock:
            # Retries can report negative progress, so keep the total in range.
            self.done = min(max(self.done + count, 0), self.total)
            percent = self.done * 100 // max(self.total, 1)
            if percent == self.last_percent:
                return
            self.last_percent = percent
            self.stream.write(f"\r{self._render(percent)}")
            self.stream.flush()

    def _render(self, percent: int) -> str:
        filled = BAR_WIDTH * percent // 100
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        speed = self.done / max(time.monotonic() - self.started, 1e-6)
        seconds_left = int((self.total - self.done) / speed) if speed > 0 else 0
        line = (
            f"{self.name} [{bar}] {percent:3d}% "
            f"{self.done / BYTES_PER_MB:.0f}/{self.total / BYTES_PER_MB:.0f} MB "
            f"{speed / BYTES_PER_MB:.1f} MB/s, {seconds_left // 60}m {seconds_left % 60:02d}s left"
        )
        return line.ljust(100)  # Clears leftovers from a longer previous line.

    def finish(self) -> None:
        with self.lock:
            if self.last_percent >= 0:
                self.stream.write("\n")
                self.stream.flush()


def _load_env_file(path: Path) -> None:
    """Add KEY=VALUE lines to the environment; variables already set take priority."""
    # shortcut: no inline comments or multi-line values, use python-dotenv if .env ever needs them.
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _settings_from_env() -> dict[str, str]:
    _load_env_file(ENV_FILE)
    settings = {name: os.environ.get(name, "").strip() for name in REQUIRED_ENV}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise DatasetError(
            f"missing settings: {', '.join(missing)}. Copy .env.example to .env and fill them in."
        )
    if not ACCOUNT_ID_PATTERN.fullmatch(settings["R2_ACCOUNT_ID"].lower()):
        raise DatasetError("R2_ACCOUNT_ID must be 32 hexadecimal characters")
    return settings


def _make_client(settings: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings['R2_ACCOUNT_ID'].lower()}.r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=settings["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=settings["R2_SECRET_ACCESS_KEY"],
        config=Config(
            signature_version="s3v4",
            connect_timeout=30,
            read_timeout=300,
            retries={"max_attempts": 5, "mode": "standard"},
            # R2 doesn't support every default checksum newer boto3 versions send.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _download(client, bucket: str, name: str, target: Path) -> None:
    """Download to a temporary file so an interrupted or bad download never gets the final name."""
    head = client.head_object(Bucket=bucket, Key=name)
    size = head["ContentLength"]
    expected_hash = head.get("Metadata", {}).get(HASH_METADATA_KEY)
    if not expected_hash:
        raise DatasetError(
            f"{name} has no {HASH_METADATA_KEY} metadata in {bucket}, so it can't be verified"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free
    if free < size:
        raise DatasetError(f"{name} needs {size} bytes but only {free} are free")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{name}.", suffix=".part"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    progress = _ProgressBar(name, size)
    try:
        transfer = TransferConfig(
            multipart_threshold=PART_BYTES,
            multipart_chunksize=PART_BYTES,
            max_concurrency=MAX_CONCURRENCY,
        )
        try:
            client.download_file(
                bucket, name, str(temporary), Config=transfer, Callback=progress
            )
        finally:
            progress.finish()
        logger.info("Verifying %s", name)
        if temporary.stat().st_size != size:
            raise DatasetError(
                f"{name}: downloaded {temporary.stat().st_size} bytes, expected {size}"
            )
        if _sha256_file(temporary) != expected_hash:
            raise DatasetError(f"{name}: downloaded file failed its checksum")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_dataset(name: str, directory: Path = DATASETS_DIR) -> Path:
    """Return the local path to a dataset, downloading it from R2 only if it's missing."""
    if name not in DATASETS:
        raise DatasetError(
            f"unknown dataset {name!r}; expected one of {', '.join(DATASETS)}"
        )

    target = Path(directory) / name
    if target.is_file() and target.stat().st_size > 0:
        return target

    settings = _settings_from_env()
    try:
        _download(_make_client(settings), settings["R2_BUCKET"], name, target)
    except (ClientError, BotoCoreError, Boto3Error, OSError) as error:
        raise DatasetError(f"could not fetch {name}: {error}") from error
    return target


def fetch_all(directory: Path = DATASETS_DIR) -> list[Path]:
    """Return local paths to every dataset, downloading any that are missing."""
    return [fetch_dataset(name, directory) for name in DATASETS]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        for path in fetch_all():
            logger.info("Ready: %s", path)
    except DatasetError as error:
        logger.error("error: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
