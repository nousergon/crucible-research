"""Unit tests for data_manifest — dated S3 manifest writer."""

import json
from unittest.mock import MagicMock, patch

from data_manifest import write_data_manifest


class TestWriteDataManifest:

    def test_writes_dated_manifest(self):
        s3 = MagicMock()
        with patch("data_manifest.boto3.client", return_value=s3):
            write_data_manifest(
                bucket="b",
                module_name="research",
                run_date="2026-05-12",
                manifest={"n_population": 25, "weekly_run": True},
            )
        call = s3.put_object.call_args
        assert call.kwargs["Key"] == "data_manifest/research/2026-05-12.json"
        payload = json.loads(call.kwargs["Body"])
        assert payload["module"] == "research"
        assert payload["run_date"] == "2026-05-12"
        assert "written_at" in payload
        assert payload["n_population"] == 25
        assert payload["weekly_run"] is True

    def test_put_failure_swallowed(self):
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("AccessDenied")
        with patch("data_manifest.boto3.client", return_value=s3):
            write_data_manifest(
                bucket="b",
                module_name="research",
                run_date="2026-05-12",
                manifest={"n_population": 0},
            )
