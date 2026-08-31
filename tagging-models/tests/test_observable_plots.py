"""Tests for post-extraction truth-split observable plots."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from candidates_extraction.plotting import (
    observable_bin_edges,
    plot_observable_pdfs,
)


class ObservablePlotTests(unittest.TestCase):
    def test_integer_observable_uses_centered_bins(self) -> None:
        edges = observable_bin_edges(
            signal_values=np.array([1.0, 2.0, 3.0]),
            background_values=np.array([2.0, 4.0]),
            n_bins=60,
        )
        np.testing.assert_array_equal(edges, np.arange(0.5, 5.5, 1.0))

    def test_writes_one_plot_per_observable(self) -> None:
        candidates = pd.DataFrame(
            {
                "label": [1, 1, 0, 0],
                "continuous": [0.1, 0.2, 0.5, 0.7],
                "with_nan": [1.0, np.nan, 2.0, 3.0],
            }
        )
        with TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "obs_plots"
            summary = plot_observable_pdfs(
                candidates,
                outdir,
                ["continuous", "with_nan"],
                n_bins=10,
            )
            self.assertEqual(summary["n_plots"], 2)
            self.assertTrue((outdir / "continuous.png").is_file())
            self.assertTrue((outdir / "with_nan.png").is_file())
            self.assertTrue((outdir / "observable_plot_summary.json").is_file())
            self.assertEqual(
                summary["plots"]["with_nan"]["n_neutron_captures_finite"],
                1,
            )
            self.assertEqual(summary["truth_definition"]["neutron_captures"], "label == 1")

    def test_per_nucleus_plot_metadata_is_absent(self) -> None:
        candidates = pd.DataFrame(
            {
                "label": [1, 1, 0],
                "Nn": [6, 8, 7],
            }
        )
        with TemporaryDirectory() as tmp:
            summary = plot_observable_pdfs(
                candidates,
                Path(tmp),
                ["Nn"],
                n_bins=10,
            )
            self.assertNotIn(
                "neutron_capture_nuclei_finite",
                summary["plots"]["Nn"],
            )


if __name__ == "__main__":
    unittest.main()
