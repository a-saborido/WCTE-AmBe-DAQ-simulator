"""Tests for continuous-window capture metadata assembly."""

import unittest

import numpy as np

from candidates_extraction.root_io import combine_window_payloads


def payload(entry: int, hit_time: float, capture_time: float):
    return {
        "absolute_entry": entry,
        "event_number": entry,
        "w_evt": {
            "time": np.array([hit_time]),
            "charge": np.array([1.0]),
            "slot": np.array([1]),
            "pos": np.array([0]),
            "cable_id": np.array([1]),
            "pmt_pos_cm": np.zeros((1, 3)),
            "pmt_dir": np.zeros((1, 3)),
            "hit_index": np.array([0]),
        },
        "t_evt": {
            "hit_from_capture": np.array([1]),
            "hit_from_prompt": np.array([0]),
            "is_background": np.array([0]),
            "source_event_idx": np.array([entry]),
            "capture_t": np.array([capture_time]),
        },
    }


class RootIoTruthMetadataTests(unittest.TestCase):
    def test_capture_times_follow_continuous_window_offsets(self) -> None:
        _, truth = combine_window_payloads(
            [
                payload(0, 10.0, 12.0),
                payload(1, 20.0, 22.0),
            ],
            first_local_index=0,
            max_following_windows=1,
            window_period_ns=1000.0,
        )
        np.testing.assert_array_equal(truth["capture_t"], [12.0, 1022.0])


if __name__ == "__main__":
    unittest.main()
