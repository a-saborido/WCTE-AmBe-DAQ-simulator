"""Fast tests for BONSAI-related mapping and observable calculations."""

from pathlib import Path
import unittest

import numpy as np

from candidates_extraction.config import OBSERVABLE_COLUMNS
from candidates_extraction.geometry import PmtGeometry, WallEstimator
from candidates_extraction.observables.vertex import calculate_bonsai_vertex_observables


REPO_DIR = Path(__file__).resolve().parents[1]


class BonsaiObservableTests(unittest.TestCase):
    def test_readout_ids_map_to_wcsim_tube_ids(self) -> None:
        geometry = PmtGeometry.from_geofile(
            REPO_DIR / "data" / "geofile_NuPRISMBeamTest_16cShort_mPMT.txt"
        )
        cable_ids, found = geometry.lookup_cable_ids(
            np.array([105, 105]),
            np.array([18, 17]),
        )
        np.testing.assert_array_equal(found, [True, True])
        np.testing.assert_array_equal(cable_ids, [1843, 1842])

    def test_bonsai_distances_inside_and_outside_wall(self) -> None:
        wall = WallEstimator(
            axis="y",
            radius_cm=100.0,
            axis_min_cm=-50.0,
            axis_max_cm=50.0,
        )
        observables = calculate_bonsai_vertex_observables(
            xbonsai_cm=np.array([110.0, 60.0, 0.0]),
            xfit_cm=np.array([100.0, 60.0, 0.0]),
            wall=wall,
        )
        self.assertAlmostEqual(observables["Bpdist"], 10.0)
        self.assertAlmostEqual(observables["Bwall"], np.sqrt(200.0))

    def test_failed_fit_produces_nan_observables(self) -> None:
        wall = WallEstimator(
            axis="y",
            radius_cm=100.0,
            axis_min_cm=-50.0,
            axis_max_cm=50.0,
        )
        observables = calculate_bonsai_vertex_observables(
            xbonsai_cm=np.full(3, np.nan),
            xfit_cm=np.zeros(3),
            wall=wall,
        )
        self.assertTrue(np.isnan(observables["Bpdist"]))
        self.assertTrue(np.isnan(observables["Bwall"]))

    def test_bonsai_observables_are_extracted(self) -> None:
        self.assertIn("Bpdist", OBSERVABLE_COLUMNS)
        self.assertIn("Bwall", OBSERVABLE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
