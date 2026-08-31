"""Tests for the local multilateration vertex refit."""

import unittest

import numpy as np

from candidates_extraction.observables.vertex import refit_vertex_by_multilateration_grid


class VertexRefitTests(unittest.TestCase):
    def test_multilateration_grid_refits_seed_and_recomputes_context_nn(self) -> None:
        true_vertex = np.array([10.0, 0.0, 0.0])
        seed_positions = np.array(
            [
                [30.0, 0.0, 0.0],
                [-30.0, 0.0, 0.0],
                [10.0, 25.0, 0.0],
                [10.0, -25.0, 0.0],
                [10.0, 0.0, 25.0],
                [10.0, 0.0, -25.0],
            ],
            dtype=float,
        )
        extra_context_position = np.array([[10.0, 15.0, 0.0]], dtype=float)
        hit_positions = np.vstack([seed_positions, extra_context_position])
        raw_times = np.linalg.norm(hit_positions - true_vertex[None, :], axis=1)
        raw_times[-1] += 1.0

        vertex, refit_nn, best_loc, _ = refit_vertex_by_multilateration_grid(
            times_ns=raw_times,
            pos_cm=hit_positions,
            prompt_vertex_cm=np.zeros(3),
            c_water_cm_per_ns=1.0,
            width_ns=10.0,
            fit_hit_indices=np.arange(6),
            xyz_bounds_cm=20.0,
            coarse_step_cm=10.0,
            fine_step_cm=5.0,
            refine_halfwidth_cm=5.0,
            dt_cut_ns=None,
            earliest_per_channel=False,
        )

        np.testing.assert_allclose(vertex, true_vertex)
        self.assertEqual(refit_nn, 7)
        self.assertIn(6, best_loc.tolist())


if __name__ == "__main__":
    unittest.main()
