"""Tests for applying a saved BDT model bundle."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import joblib
import numpy as np
import pandas as pd

from bdt_model.evaluation import evaluate_predictions
from bdt_model.prediction import score_candidates


class SumProbabilityModel:
    """Small deterministic model used to test the scoring file workflow."""

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        signal = np.sum(values, axis=1) / 10.0
        return np.column_stack([1.0 - signal, signal])


class BdtPredictionTests(unittest.TestCase):
    def test_scoring_filters_nonfinite_rows_and_writes_scores(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            model_path = directory / "model.joblib"
            candidates_path = directory / "candidates.csv"
            output_path = directory / "scored.csv"

            joblib.dump(
                {
                    "model": SumProbabilityModel(),
                    "feature_columns": ["feature_b", "feature_a"],
                },
                model_path,
            )
            pd.DataFrame(
                {
                    "feature_a": [1.0, np.nan, 3.0],
                    "feature_b": [2.0, 2.0, 4.0],
                    "candidate_id": [10, 11, 12],
                }
            ).to_csv(candidates_path, index=False)

            scored = score_candidates(model_path, candidates_path, output_path)

            np.testing.assert_array_equal(scored["candidate_id"], [10, 12])
            np.testing.assert_allclose(scored["bdt_score"], [0.3, 0.7])
            self.assertTrue(output_path.exists())

    def test_labeled_prediction_writes_evaluation_artifacts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            outdir = Path(temporary_directory)
            scored = pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "bdt_score": [0.1, 0.2, 0.8, 0.9],
                }
            )

            metrics = evaluate_predictions(scored, outdir, bdt_cuts=[0.5])

            self.assertEqual(metrics["auc"], 1.0)
            self.assertEqual(metrics["average_precision"], 1.0)
            expected_files = {
                "prediction_metrics.json",
                "prediction_bdt_cut_table.csv",
                "prediction_bdt_score.png",
                "prediction_roc_signal_eff_vs_bkg_acceptance.png",
            }
            self.assertTrue(expected_files.issubset(path.name for path in outdir.iterdir()))

            saved_metrics = json.loads(
                (outdir / "prediction_metrics.json").read_text()
            )
            self.assertEqual(saved_metrics["n_candidates_evaluated"], 4)

            cut_table = pd.read_csv(outdir / "prediction_bdt_cut_table.csv")
            self.assertEqual(cut_table.loc[0, "n_prediction_passing"], 2)
            self.assertEqual(
                cut_table.loc[0, "signal_efficiency_candidate_level"],
                1.0,
            )
            self.assertEqual(
                cut_table.loc[0, "background_acceptance_candidate_level"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
