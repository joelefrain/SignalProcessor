import unittest
from pathlib import Path

from signalprocessor.benchmark import discover_benchmark_pairs, read_seismomatch_txt


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_read_seismomatch_txt(self):
        motion = read_seismomatch_txt(ROOT / "examples" / "data" / "benchmark" / "ATICOEW.txt")
        self.assertEqual(motion.name, "ATICOEW")
        self.assertAlmostEqual(motion.dt, 0.02, places=8)
        self.assertGreater(motion.npts, 9000)

    def test_discover_pairs_maps_double_dot_name(self):
        pairs = discover_benchmark_pairs(ROOT)
        names = {name for name, _, _ in pairs}
        self.assertIn("LIMAEW", names)
        self.assertEqual(len(pairs), 4)


if __name__ == "__main__":
    unittest.main()
