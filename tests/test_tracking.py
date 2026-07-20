from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from common.tracking import TrackedRun


class TrackingRunTest(unittest.TestCase):
    @patch("common.tracking.trackio.Artifact")
    @patch("common.tracking.trackio.init")
    def test_logs_typed_config_metrics_and_run_bundle(
        self,
        init: Mock,
        artifact_type: Mock,
    ) -> None:
        remote = Mock()
        remote.id = "trackio-id"
        remote.name = "benchmark-one"
        init.return_value = remote
        artifact_type.return_value = Mock()

        with tempfile.TemporaryDirectory() as directory:
            with TrackedRun.start(
                "serving-benchmark",
                {"backend": "vllm"},
                name="benchmark-one",
                runs_dir=Path(directory),
                auto_log_gpu=False,
                auto_log_cpu=False,
            ) as run:
                run.log({"serve/output_token_throughput": 100.0})

            config = init.call_args.kwargs["config"]
            self.assertEqual(config["run_kind"], "serving-benchmark")
            self.assertEqual(config["schema_version"], 1)
            self.assertEqual(config["resolved_config"]["backend"], "vllm")
            self.assertTrue((Path(directory) / "benchmark-one" / "metadata.json").is_file())
            remote.log_artifact.assert_called_once()
            remote.finish.assert_called_once()

    @patch("common.tracking.trackio.init")
    def test_records_failure_and_does_not_swallow_exception(self, init: Mock) -> None:
        remote = Mock()
        remote.id = "trackio-id"
        remote.name = "failed-run"
        init.return_value = remote

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "broken"):
                with TrackedRun.start(
                    "general-eval",
                    {},
                    name="failed-run",
                    runs_dir=Path(directory),
                    auto_log_gpu=False,
                    auto_log_cpu=False,
                ):
                    raise RuntimeError("broken")

            self.assertTrue(
                (Path(directory) / "failed-run" / "output" / "failure.json").is_file()
            )
            remote.finish.assert_called_once()

    @patch("common.tracking.trackio.Artifact")
    @patch("common.tracking.trackio.init")
    def test_artifact_event_uses_artifact_name(
        self,
        init: Mock,
        artifact_type: Mock,
    ) -> None:
        remote = Mock(id="trackio-id", name="artifact-run")
        init.return_value = remote
        artifact_type.return_value = Mock()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.json"
            result.write_text("{}\n", encoding="utf-8")
            run = TrackedRun.start(
                "serving-benchmark",
                {},
                name="artifact-run",
                runs_dir=root / "runs",
                auto_log_gpu=False,
                auto_log_cpu=False,
            )
            run.log_artifact(
                result,
                name="result",
                artifact_type="serving-benchmark",
            )

            events = (run.context.root / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn('\"artifact_name\": \"result\"', events)


if __name__ == "__main__":
    unittest.main()
