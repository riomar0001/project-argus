import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import boto3
from moto import mock_aws

from project_argus import datasets

BUCKET = "test-bucket"
NAME = "DS_Base.csv"
CONTENT = b"Full_request,class\nGET / HTTP/1.1,Normal\n"
CONTENT_HASH = hashlib.sha256(CONTENT).hexdigest()


class ProgressBarTests(unittest.TestCase):
    def test_draws_progress_and_ends_the_line(self) -> None:
        stream = io.StringIO()
        bar = datasets._ProgressBar("file.csv", 200 * 10**6, stream)
        bar(100 * 10**6)
        bar(100 * 10**6)
        bar.finish()
        output = stream.getvalue()
        self.assertIn(" 50% 100/200 MB", output)
        self.assertIn("100% 200/200 MB", output)
        self.assertIn("#" * datasets.BAR_WIDTH, output)
        self.assertTrue(output.endswith("\n"))

    def test_progress_stays_in_range_and_skips_repeats(self) -> None:
        stream = io.StringIO()
        bar = datasets._ProgressBar("file.csv", 100, stream)
        bar(-50)
        bar(500)
        bar(1)
        self.assertEqual(bar.done, 100)
        self.assertEqual(stream.getvalue().count("\r"), 2)  # 0% and 100% only.

    def test_finish_without_progress_writes_nothing(self) -> None:
        stream = io.StringIO()
        datasets._ProgressBar("file.csv", 100, stream).finish()
        self.assertEqual(stream.getvalue(), "")


class FetchWithoutNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        # No credentials and no .env, so any attempt to reach R2 raises.
        patches = (mock.patch.dict(os.environ, clear=True), mock.patch.object(datasets, "ENV_FILE", self.directory / "none"))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_existing_file_needs_no_credentials(self) -> None:
        target = self.directory / NAME
        target.write_bytes(CONTENT)
        self.assertEqual(datasets.fetch_dataset(NAME, self.directory), target)

    def test_unknown_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(datasets.DatasetError, "unknown dataset"):
            datasets.fetch_dataset("../secret.csv", self.directory)

    def test_missing_file_without_settings_names_them(self) -> None:
        with self.assertRaisesRegex(datasets.DatasetError, "R2_ACCOUNT_ID.*R2_BUCKET"):
            datasets.fetch_dataset(NAME, self.directory)

    def test_env_file_is_loaded_but_real_environment_wins(self) -> None:
        env_file = self.directory / ".env"
        env_file.write_text('# comment\nR2_BUCKET="from-file"\nR2_ACCESS_KEY_ID=from-file\n', encoding="utf-8")
        os.environ["R2_ACCESS_KEY_ID"] = "from-environment"
        datasets._load_env_file(env_file)
        self.assertEqual(os.environ["R2_BUCKET"], "from-file")
        self.assertEqual(os.environ["R2_ACCESS_KEY_ID"], "from-environment")


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        patch = mock.patch.dict(os.environ, AWS_ACCESS_KEY_ID="test", AWS_SECRET_ACCESS_KEY="test", AWS_DEFAULT_REGION="us-east-1")
        patch.start()
        self.addCleanup(patch.stop)
        aws = mock_aws()
        aws.start()
        self.addCleanup(aws.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name) / "new" / "datasets"
        self.target = self.directory / NAME
        self.client = boto3.client("s3")
        self.client.create_bucket(Bucket=BUCKET)

    def put(self, metadata: dict[str, str]) -> None:
        self.client.put_object(Bucket=BUCKET, Key=NAME, Body=CONTENT, Metadata=metadata)

    def assert_nothing_left_behind(self) -> None:
        self.assertEqual(list(self.directory.glob("*")) if self.directory.exists() else [], [])

    def test_downloads_and_creates_the_folder(self) -> None:
        self.put({"sha256": CONTENT_HASH})
        datasets._download(self.client, BUCKET, NAME, self.target)
        self.assertEqual(self.target.read_bytes(), CONTENT)
        self.assertEqual([path.name for path in self.directory.iterdir()], [NAME])

    def test_corrupt_download_is_discarded(self) -> None:
        self.put({"sha256": "0" * 64})
        with self.assertRaisesRegex(datasets.DatasetError, "checksum"):
            datasets._download(self.client, BUCKET, NAME, self.target)
        self.assert_nothing_left_behind()

    def test_object_without_a_hash_is_refused(self) -> None:
        self.put({})
        with self.assertRaisesRegex(datasets.DatasetError, "can't be verified"):
            datasets._download(self.client, BUCKET, NAME, self.target)
        self.assert_nothing_left_behind()


if __name__ == "__main__":
    unittest.main()
