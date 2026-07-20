from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common import ProfileResolver, RunContext


class RunContextTest(unittest.TestCase):
    def test_creates_generic_run_layout_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = RunContext.create(Path(directory) / "run", "sft", {"seed": 42})
            run.event("metrics", step=1, values={"loss": 1.0})
            run.complete()

            self.assertTrue((run.root / "resolved-config.yaml").is_file())
            self.assertTrue((run.root / "metadata.json").is_file())
            self.assertTrue(run.artifacts_dir.is_dir())
            self.assertTrue(run.recovery_dir.is_dir())
            self.assertTrue(run.output_dir.is_dir())
            events = [json.loads(line) for line in (run.root / "events.jsonl").read_text().splitlines()]
            self.assertEqual(
                [event["event"] for event in events],
                ["run_started", "metrics", "run_completed"],
            )

    def test_captures_resolved_profile_and_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "profiles" / "models"
            model_dir.mkdir(parents=True)
            (model_dir / "foundation.yaml").write_text(
                "id: foundation\nmodel:\n  artifact: hf://org/model@revision\n  form: base\n  weights: {format: safetensors}\n  capabilities: {context_window: 4096}\n",
                encoding="utf-8",
            )
            profile = ProfileResolver(root / "profiles").resolve("models", "foundation")

            run = RunContext.create(root / "run", "model-onboarding", {}, profile)

            self.assertTrue((run.root / "resolved-profile.yaml").is_file())
            metadata = json.loads((run.root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["profile"]["reference"], "foundation")
            self.assertEqual(metadata["profile"]["kind"], "models")

    def test_rejects_unknown_run_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported run kind"):
                RunContext.create(Path(directory) / "run", "benchmark", {})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
