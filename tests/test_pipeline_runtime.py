from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kb_pipeline.pipeline import VLLMManager
from kb_pipeline import pipeline as pipeline_module


class ExternalVLLMManagerTest(unittest.TestCase):
    def test_model_paths_match_with_trailing_slash(self) -> None:
        self.assertTrue(
            VLLMManager._models_match(
                "/models/example/",
                "/models/example",
            )
        )

    def test_external_mode_accepts_ready_service_without_managing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = VLLMManager(
                Path(directory),
                start_timeout_sec=1,
                start_poll_sec=1,
                runtime_mode="external",
            )
            with patch.object(
                manager,
                "probe",
                return_value="/models/example",
            ):
                manager.start("/models/example")

            self.assertEqual(manager.current_model, "/models/example")
            self.assertFalse(manager.owned)

    def test_external_stop_never_calls_managed_stop_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = VLLMManager(
                Path(directory),
                runtime_mode="external",
            )
            manager.current_model = "/models/example"
            with patch.object(manager, "_run") as run:
                manager.stop(force=True)

            run.assert_not_called()
            self.assertIsNone(manager.current_model)

    def test_pipeline_uses_global_managed_runtime_directory(self) -> None:
        source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
        self.assertIn(
            'output_dir_path / "runtime" / "vllm"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
