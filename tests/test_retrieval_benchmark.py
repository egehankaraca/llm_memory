import math
import unittest

from scripts.run_retrieval_benchmark import (
    deterministic_vector,
    latency_summary_ms,
    parse_sizes,
    percentile,
)


class RetrievalBenchmarkTest(unittest.TestCase):
    def test_vector_is_deterministic_distinct_and_normalized(self) -> None:
        first = deterministic_vector(12, 99)
        repeated = deterministic_vector(12, 99)
        other = deterministic_vector(13, 99)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 768)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0)

    def test_percentiles_and_latency_summary_are_stable(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        summary = latency_summary_ms([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertEqual(summary["max_ms"], 4.0)
        self.assertEqual(summary["mean_ms"], 2.5)

    def test_sizes_are_sorted_unique_and_bounded(self) -> None:
        self.assertEqual(parse_sizes([10_000, 1_000, 1_000]), [1_000, 10_000])
        with self.assertRaises(ValueError):
            parse_sizes([99])


if __name__ == "__main__":
    unittest.main()
