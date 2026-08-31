"""Tests for source-file and readout-window BDT splitting."""

import unittest

import pandas as pd

from bdt_model.training import split_candidate_windows


class BdtTrainingSplitTests(unittest.TestCase):
    def test_each_file_contributes_whole_windows_to_each_split(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "source_file": source_file,
                    "source_entry": source_entry,
                    "candidate_id": candidate_id,
                }
                for source_file in ["sample_a.root", "sample_b.root"]
                for source_entry in range(8)
                for candidate_id in range(2)
            ]
        )

        idx_train, idx_val, idx_test = split_candidate_windows(candidates, seed=12345)
        split_by_row = pd.Series("", index=candidates.index)
        split_by_row.iloc[idx_train] = "train"
        split_by_row.iloc[idx_val] = "validation"
        split_by_row.iloc[idx_test] = "test"

        self.assertTrue(split_by_row.ne("").all())
        for _, window_rows in candidates.groupby(["source_file", "source_entry"]):
            self.assertEqual(split_by_row.loc[window_rows.index].nunique(), 1)

        candidates_with_split = candidates.assign(split=split_by_row)
        for _, file_rows in candidates_with_split.groupby("source_file"):
            window_counts = file_rows.groupby("split")["source_entry"].nunique()
            self.assertEqual(window_counts.to_dict(), {"test": 2, "train": 4, "validation": 2})


if __name__ == "__main__":
    unittest.main()
