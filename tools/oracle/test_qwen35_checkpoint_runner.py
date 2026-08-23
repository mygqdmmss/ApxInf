import importlib.util
import struct
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("qwen35_checkpoint_runner.py")
SPEC = importlib.util.spec_from_file_location("qwen35_checkpoint_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class RunnerUnitTests(unittest.TestCase):
    def test_greedy_stop_includes_eos_and_reports_eos(self):
        tokens, reason = runner.greedy_token_ids([11, 248046, 12], 8)
        self.assertEqual(tokens, [11, 248046])
        self.assertEqual(reason, "eos")

    def test_greedy_stop_consumes_budget_without_eos(self):
        tokens, reason = runner.greedy_token_ids([11, 12, 13], 3)
        self.assertEqual(tokens, [11, 12, 13])
        self.assertEqual(reason, "budget")

    def test_artifact_path_rejects_escape(self):
        with self.assertRaises(ValueError):
            runner.safe_artifact_path(Path("/tmp/apxinf-runner-test"), "../escape.bin")

    def test_f32_writer_is_little_endian_and_finite(self):
        path = Path("/tmp/apxinf-runner-f32.bin")
        try:
            runner.write_f32(path, [1.0, -2.5])
            self.assertEqual(path.read_bytes(), struct.pack("<ff", 1.0, -2.5))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
