"""Dependency-free checks of trajectory-recording configuration; torch runs are in tests/integration."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jev_dllm.backends.instrumented import checkpoint_blocks


class CheckpointTests(unittest.TestCase):
    def test_quarter_checkpoints_for_both_generation_lengths(self):
        self.assertEqual(checkpoint_blocks(8, [2, 4, 6]), (2, 4, 6))     # 256 tokens, block 32.
        self.assertEqual(checkpoint_blocks(16, [4, 8, 12]), (4, 8, 12))  # 512 tokens, block 32.

    def test_invalid_checkpoints_are_rejected(self):
        for blocks in ([], [0, 2], [2, 8], [4, 2], [2, 2], [2.0]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                checkpoint_blocks(8, blocks)


if __name__ == '__main__':
    unittest.main()
