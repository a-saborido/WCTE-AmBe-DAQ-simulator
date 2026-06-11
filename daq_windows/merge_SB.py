# ============================================================
# Merge MC-like ROOT + background-like ROOT window-by-window
#
# Case 1:
#   mc_root has no TTrueInfo/is_background:
#       mc hits get is_background = False
#
# Case 2:
#   mc_root already has TTrueInfo/is_background:
#       keep those values exactly
#
# For bkg_root:
#   all bkg hits get is_background = True
#
# If apply_resolution_after_merge = False:
#   fast vectorized merge
#
# If apply_resolution_after_merge = True:
#   merge first, then apply same-PMT time-resolution merging
# ============================================================

from pathlib import Path
import gc

import numpy as np
import awkward as ak
import uproot
from tqdm.auto import tqdm


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

mc_root = Path(
    "/scratch/saborido/WCTE-AmBe-DAQ-simulator/daq_windows/output/"
    "wcte_ambe_mc_digidata_0.root"
)

bkg_root = Path(
    "/scratch/saborido/AmBe_data/clean_bkg/"
    "WCTE_AmBe_clean_bkg_pe.root"
)

output_root = Path("output/wcte_ambe_mc_plus_clean_bkg_pe.root")
output_root.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

chunk_size = 2000
require_same_n_windows = False

apply_resolution_after_merge = False
resolution_ns = 20.0

resolution_time_branch = "hit_pmt_calibrated_times"

pmt_key_branches = [
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
]

gc_every_n_chunks = None

# If bkg_root has TTrueInfo, use its truth branches when available.
# is_background for bkg_root is still forced to True.
use_bkg_truth_if_available = True


# ------------------------------------------------------------
# WCTEReadoutWindows branches
# ------------------------------------------------------------

wcte_branches = [
    "window_time",
    "start_counter",
    "run_id",
    "sub_run_id",
    "spill_counter",
    "event_number",
    "readout_number",

    "trigger_types",
    "trigger_times",

    "led_gains",
    "led_dacsettings",
    "led_ids",
    "led_card_ids",
    "led_slot_numbers",
    "led_event_types",
    "led_types",
    "led_sequence_numbers",
    "led_counters",

    "hit_mpmt_card_ids",
    "hit_pmt_channel_ids",
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
    "hit_pmt_charges",
    "hit_pmt_times",

    "pmt_waveform_mpmt_card_ids",
    "pmt_waveform_pmt_channel_ids",
    "pmt_waveform_mpmt_slot_ids",
    "pmt_waveform_pmt_position_ids",
    "pmt_waveform_times",
    "pmt_waveforms",

    "beamline_pmt_qdc_charges",
    "beamline_pmt_tdc_times",
    "beamline_pmt_qdc_ids",
    "beamline_pmt_tdc_ids",

    "hit_pmt_calibrated_times",
    "hit_pmt_has_time_constant",
]

# User-specified WCTE hit-level branches
wcte_hit_branches = [
    "hit_mpmt_card_ids",
    "hit_pmt_channel_ids",
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
    "hit_pmt_charges",
    "hit_pmt_times",
    "hit_pmt_calibrated_times",
    "hit_pmt_has_time_constant",
]

wcte_dtypes = {
    "window_time": np.int64,
    "start_counter": np.int64,
    "run_id": np.int32,
    "sub_run_id": np.int32,
    "spill_counter": np.int32,
    "event_number": np.int32,
    "readout_number": np.int32,

    "trigger_types": np.int32,
    "trigger_times": np.float64,

    "led_gains": np.float32,
    "led_dacsettings": np.int32,
    "led_ids": np.int32,
    "led_card_ids": np.int32,
    "led_slot_numbers": np.int32,
    "led_event_types": np.int32,
    "led_types": np.int32,
    "led_sequence_numbers": np.int32,
    "led_counters": np.int32,

    "hit_mpmt_card_ids": np.int32,
    "hit_pmt_channel_ids": np.int32,
    "hit_mpmt_slot_ids": np.int32,
    "hit_pmt_position_ids": np.int32,
    "hit_pmt_charges": np.float32,
    "hit_pmt_times": np.float64,
    "hit_pmt_calibrated_times": np.float64,
    "hit_pmt_has_time_constant": np.int8,

    "pmt_waveform_mpmt_card_ids": np.int32,
    "pmt_waveform_pmt_channel_ids": np.int32,
    "pmt_waveform_mpmt_slot_ids": np.int32,
    "pmt_waveform_pmt_position_ids": np.int32,
    "pmt_waveform_times": np.float64,
    "pmt_waveforms": np.float32,

    "beamline_pmt_qdc_charges": np.float32,
    "beamline_pmt_tdc_times": np.float64,
    "beamline_pmt_qdc_ids": np.int32,
    "beamline_pmt_tdc_ids": np.int32,
}


# ------------------------------------------------------------
# TTrueInfo branches
# ------------------------------------------------------------

ttrue_scalar_branches = [
    "event_number",
    "n_overlay",
]

# Hit-level truth branches
ttrue_hit_branches = [
    "x",
    "y",
    "z",

    "dx",
    "dy",
    "dz",

    "hit_from_prompt",
    "hit_from_capture",
    "source_event_idx",
]

# Non-hit-level truth metadata
ttrue_nonhit_branches = [
    "capture_t",
    "relative_capture_t",
    "prompt_time_withcapture",
    "prompt_time",

    "capture_x",
    "capture_y",
    "capture_z",

    "capture_nucleus",
    "capture_ngamma",
    "capture_total_gammaE",
]

ttrue_required_mc_branches = (
    ttrue_scalar_branches
    + ttrue_hit_branches
    + ttrue_nonhit_branches
)

ttrue_dtypes = {
    "event_number": np.int32,
    "n_overlay": np.int32,

    "x": np.float32,
    "y": np.float32,
    "z": np.float32,

    "dx": np.float32,
    "dy": np.float32,
    "dz": np.float32,

    "hit_from_prompt": np.int32,
    "hit_from_capture": np.int32,
    "source_event_idx": np.int32,

    "capture_t": np.float64,
    "relative_capture_t": np.float64,
    "prompt_time_withcapture": np.float64,
    "prompt_time": np.float64,

    "capture_x": np.float32,
    "capture_y": np.float32,
    "capture_z": np.float32,

    "capture_nucleus": np.int32,
    "capture_ngamma": np.int32,
    "capture_total_gammaE": np.float32,

    "is_background": np.bool_,
}

# Used only when bkg_root has no usable TTrueInfo hit-level truth.
ttrue_background_fill_values = {
    "x": np.nan,
    "y": np.nan,
    "z": np.nan,

    "dx": np.nan,
    "dy": np.nan,
    "dz": np.nan,

    "hit_from_prompt": 0,
    "hit_from_capture": 0,
    "source_event_idx": -1,
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def check_required_branches(tree, required, tree_name, file_path):
    available = set(tree.keys())
    missing = [b for b in required if b not in available]

    if len(missing) > 0:
        raise RuntimeError(
            f"Missing branches in {file_path}:{tree_name}:\n"
            + "\n".join(missing)
        )


def cast_branch(arr, dtype):
    """
    Cast scalar or jagged branch to dtype.
    """
    try:
        return ak.values_astype(arr, dtype)
    except Exception:
        return ak.to_numpy(arr).astype(dtype, copy=False)


def make_jagged_from_lengths(lengths, value, dtype):
    """
    Make a jagged array with per-window lengths and constant fill value.
    """
    lengths = np.asarray(lengths, dtype=np.int64)

    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    total = int(offsets[-1])
    content = np.full(total, value, dtype=dtype)

    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets),
            ak.contents.NumpyArray(content),
        )
    )


def list_of_numpy_to_ak(list_of_arrays, dtype):
    return ak.Array([
        np.asarray(x, dtype=dtype)
        for x in list_of_arrays
    ])


def concat_one_window(a, b, dtype):
    return np.concatenate([
        np.asarray(a, dtype=dtype),
        np.asarray(b, dtype=dtype),
    ])


def get_mc_is_background(mc_ttrue, mc_hit_lengths, mc_has_is_background):
    """
    Return hit-level is_background for mc_root.

    If it exists, preserve it.
    If not, create all-False labels.
    """
    if mc_has_is_background:
        return ak.values_astype(mc_ttrue["is_background"], np.bool_)

    return make_jagged_from_lengths(
        mc_hit_lengths,
        False,
        np.bool_,
    )


def get_bkg_truth_part(
    bkg_ttrue,
    branch,
    bkg_hit_lengths,
    use_bkg_hit_truth,
):
    """
    Return bkg-root truth values for one hit-level truth branch.

    If bkg truth is available, use it.
    Otherwise fill with dummy background values.
    """
    dtype = ttrue_dtypes[branch]

    if use_bkg_hit_truth:
        return ak.values_astype(
            bkg_ttrue[branch],
            dtype,
        )

    return make_jagged_from_lengths(
        bkg_hit_lengths,
        ttrue_background_fill_values[branch],
        dtype,
    )


def merge_scalar_branch(mc_ttrue, bkg_ttrue, branch, use_bkg_scalar_truth):
    """
    Merge scalar truth branches.

    event_number:
        copied from mc_root

    n_overlay:
        if bkg n_overlay exists, add them;
        otherwise keep mc n_overlay.
    """
    dtype = ttrue_dtypes[branch]

    if branch == "event_number":
        return cast_branch(mc_ttrue[branch], dtype)

    if branch == "n_overlay" and use_bkg_scalar_truth:
        return (
            ak.to_numpy(mc_ttrue[branch]).astype(dtype, copy=False)
            + ak.to_numpy(bkg_ttrue[branch]).astype(dtype, copy=False)
        )

    return cast_branch(mc_ttrue[branch], dtype)


def merge_nonhit_truth_branch(
    mc_ttrue,
    bkg_ttrue,
    branch,
    use_bkg_nonhit_truth,
):
    """
    Merge non-hit-level truth metadata.

    If bkg has compatible metadata, concatenate window-by-window.
    Otherwise keep mc metadata unchanged.
    """
    dtype = ttrue_dtypes[branch]

    if use_bkg_nonhit_truth:
        return ak.concatenate(
            [
                ak.values_astype(mc_ttrue[branch], dtype),
                ak.values_astype(bkg_ttrue[branch], dtype),
            ],
            axis=1,
        )

    return cast_branch(mc_ttrue[branch], dtype)


# ============================================================
# Fast path: no resolution effect
# ============================================================

def build_wcte_fast_no_resolution(mc_wcte, bkg_wcte):
    """
    Fast vectorized WCTE merge.
    """
    out = {}

    for branch in wcte_branches:
        dtype = wcte_dtypes[branch]

        if branch in wcte_hit_branches:
            out[branch] = ak.concatenate(
                [
                    ak.values_astype(mc_wcte[branch], dtype),
                    ak.values_astype(bkg_wcte[branch], dtype),
                ],
                axis=1,
            )
        else:
            out[branch] = cast_branch(mc_wcte[branch], dtype)

    return out


def build_ttrue_fast_no_resolution(
    mc_ttrue,
    bkg_ttrue,
    mc_hit_lengths,
    bkg_hit_lengths,
    mc_has_is_background,
    use_bkg_hit_truth,
    use_bkg_nonhit_truth,
    use_bkg_scalar_truth,
):
    """
    Fast vectorized TTrueInfo merge.
    """
    out = {}

    # Scalar/window-level truth
    for branch in ttrue_scalar_branches:
        out[branch] = merge_scalar_branch(
            mc_ttrue,
            bkg_ttrue,
            branch,
            use_bkg_scalar_truth,
        )

    # Hit-level truth
    for branch in ttrue_hit_branches:
        dtype = ttrue_dtypes[branch]

        mc_part = ak.values_astype(
            mc_ttrue[branch],
            dtype,
        )

        bkg_part = get_bkg_truth_part(
            bkg_ttrue,
            branch,
            bkg_hit_lengths,
            use_bkg_hit_truth,
        )

        out[branch] = ak.concatenate(
            [mc_part, bkg_part],
            axis=1,
        )

    # is_background
    mc_is_background = get_mc_is_background(
        mc_ttrue,
        mc_hit_lengths,
        mc_has_is_background,
    )

    bkg_is_background = make_jagged_from_lengths(
        bkg_hit_lengths,
        True,
        np.bool_,
    )

    out["is_background"] = ak.concatenate(
        [mc_is_background, bkg_is_background],
        axis=1,
    )

    # Non-hit-level truth metadata
    for branch in ttrue_nonhit_branches:
        out[branch] = merge_nonhit_truth_branch(
            mc_ttrue,
            bkg_ttrue,
            branch,
            use_bkg_nonhit_truth,
        )

    return out


# ============================================================
# Slow path: merge first, then apply resolution
# ============================================================

def build_premerged_window(
    mc_wcte,
    bkg_wcte,
    mc_ttrue,
    bkg_ttrue,
    iw,
    mc_has_is_background,
    use_bkg_hit_truth,
):
    """
    Build one pre-resolution merged window.

    Hit order:
        mc_root hits first, then bkg_root hits.
    """
    wcte_hits = {}

    for branch in wcte_hit_branches:
        dtype = wcte_dtypes[branch]

        wcte_hits[branch] = concat_one_window(
            mc_wcte[branch][iw],
            bkg_wcte[branch][iw],
            dtype,
        )

    mc_n_hits = len(np.asarray(mc_wcte[resolution_time_branch][iw]))
    bkg_n_hits = len(np.asarray(bkg_wcte[resolution_time_branch][iw]))

    truth_hits = {}

    for branch in ttrue_hit_branches:
        dtype = ttrue_dtypes[branch]

        mc_values = np.asarray(
            mc_ttrue[branch][iw],
            dtype=dtype,
        )

        if len(mc_values) != mc_n_hits:
            raise RuntimeError(
                f"TTrueInfo/{branch} length {len(mc_values)} "
                f"does not match MC WCTE hit count {mc_n_hits} "
                f"in local window {iw}."
            )

        if use_bkg_hit_truth:
            bkg_values = np.asarray(
                bkg_ttrue[branch][iw],
                dtype=dtype,
            )

            if len(bkg_values) != bkg_n_hits:
                raise RuntimeError(
                    f"BKG TTrueInfo/{branch} length {len(bkg_values)} "
                    f"does not match BKG WCTE hit count {bkg_n_hits} "
                    f"in local window {iw}."
                )
        else:
            bkg_values = np.full(
                bkg_n_hits,
                ttrue_background_fill_values[branch],
                dtype=dtype,
            )

        truth_hits[branch] = np.concatenate(
            [mc_values, bkg_values]
        )

    # mc_root is_background
    if mc_has_is_background:
        mc_is_background = np.asarray(
            mc_ttrue["is_background"][iw],
            dtype=np.bool_,
        )

        if len(mc_is_background) != mc_n_hits:
            raise RuntimeError(
                f"MC TTrueInfo/is_background length {len(mc_is_background)} "
                f"does not match MC WCTE hit count {mc_n_hits} "
                f"in local window {iw}."
            )
    else:
        mc_is_background = np.zeros(
            mc_n_hits,
            dtype=np.bool_,
        )

    # bkg_root always forced to background
    bkg_is_background = np.ones(
        bkg_n_hits,
        dtype=np.bool_,
    )

    truth_hits["is_background"] = np.concatenate(
        [mc_is_background, bkg_is_background]
    )

    return wcte_hits, truth_hits


def append_resolution_cluster(
    cluster_idx,
    wcte_hits,
    truth_hits,
    out_wcte,
    out_truth,
    stats,
):
    """
    Merge one same-PMT time cluster into one output hit.
    """
    cluster_idx = np.asarray(cluster_idx, dtype=np.int64)
    first_idx = int(cluster_idx[0])

    is_bkg_cluster = np.asarray(
        truth_hits["is_background"][cluster_idx],
        dtype=bool,
    )

    has_nonbackground = np.any(~is_bkg_cluster)
    has_background = np.any(is_bkg_cluster)

    if has_nonbackground and has_background:
        stats["mixed_background_nonbackground_clusters"] += 1

    # -----------------------------
    # WCTE hit-level branches
    # -----------------------------
    for branch in wcte_hit_branches:

        if branch == "hit_pmt_charges":
            value = np.sum(
                np.asarray(wcte_hits[branch][cluster_idx], dtype=np.float64)
            )

        elif branch == "hit_pmt_has_time_constant":
            value = np.max(
                np.asarray(wcte_hits[branch][cluster_idx], dtype=np.int8)
            )

        else:
            value = wcte_hits[branch][first_idx]

        out_wcte[branch].append(value)

    # -----------------------------
    # Truth hit-level branches
    # -----------------------------
    if has_nonbackground:
        truth_ref_indices = cluster_idx[~is_bkg_cluster]
        truth_ref_idx = int(truth_ref_indices[0])
        is_background_value = False
    else:
        truth_ref_indices = cluster_idx
        truth_ref_idx = first_idx
        is_background_value = True

    for branch in ttrue_hit_branches:

        if branch in ["hit_from_prompt", "hit_from_capture"]:
            # Do not allow background hits to create prompt/capture labels
            # for a mixed nonbackground+background merged cluster.
            value = np.max(
                np.asarray(
                    truth_hits[branch][truth_ref_indices],
                    dtype=np.int32,
                )
            )
        else:
            value = truth_hits[branch][truth_ref_idx]

        out_truth[branch].append(value)

    out_truth["is_background"].append(is_background_value)


def apply_resolution_to_premerged_window(
    wcte_hits,
    truth_hits,
    resolution_ns,
):
    """
    Same-PMT time-resolution merging for one premerged window.
    """
    n_hits = len(wcte_hits[resolution_time_branch])

    stats = {
        "hits_before": int(n_hits),
        "hits_after": 0,
        "mixed_background_nonbackground_clusters": 0,
    }

    out_wcte = {
        branch: []
        for branch in wcte_hit_branches
    }

    out_truth = {
        branch: []
        for branch in ttrue_hit_branches + ["is_background"]
    }

    if n_hits == 0:
        resolved_wcte = {
            branch: np.asarray([], dtype=wcte_dtypes[branch])
            for branch in wcte_hit_branches
        }

        resolved_truth = {
            branch: np.asarray([], dtype=ttrue_dtypes[branch])
            for branch in ttrue_hit_branches + ["is_background"]
        }

        return resolved_wcte, resolved_truth, stats

    times = np.asarray(
        wcte_hits[resolution_time_branch],
        dtype=np.float64,
    )

    key_arrays = [
        np.asarray(wcte_hits[branch], dtype=np.int32)
        for branch in pmt_key_branches
    ]

    keys = list(zip(*key_arrays))
    unique_keys = sorted(set(keys))

    for key in unique_keys:

        idx = np.asarray(
            [i for i, this_key in enumerate(keys) if this_key == key],
            dtype=np.int64,
        )

        idx = idx[np.argsort(times[idx], kind="stable")]

        i = 0
        n_this_pmt = len(idx)

        while i < n_this_pmt:
            start = i
            j = i + 1

            while j < n_this_pmt:
                t_prev = times[idx[j - 1]]
                t_now = times[idx[j]]

                if (
                    np.isfinite(t_prev)
                    and np.isfinite(t_now)
                    and (t_now - t_prev) < resolution_ns
                ):
                    j += 1
                else:
                    break

            append_resolution_cluster(
                idx[start:j],
                wcte_hits,
                truth_hits,
                out_wcte,
                out_truth,
                stats,
            )

            i = j

    resolved_wcte = {}
    resolved_truth = {}

    for branch in wcte_hit_branches:
        resolved_wcte[branch] = np.asarray(
            out_wcte[branch],
            dtype=wcte_dtypes[branch],
        )

    for branch in ttrue_hit_branches + ["is_background"]:
        resolved_truth[branch] = np.asarray(
            out_truth[branch],
            dtype=ttrue_dtypes[branch],
        )

    if len(resolved_wcte[resolution_time_branch]) > 0:
        order = np.argsort(
            resolved_wcte[resolution_time_branch],
            kind="stable",
        )

        for branch in wcte_hit_branches:
            resolved_wcte[branch] = resolved_wcte[branch][order]

        for branch in ttrue_hit_branches + ["is_background"]:
            resolved_truth[branch] = resolved_truth[branch][order]

    stats["hits_after"] = int(len(resolved_wcte[resolution_time_branch]))

    return resolved_wcte, resolved_truth, stats


def build_chunks_with_resolution(
    mc_wcte,
    bkg_wcte,
    mc_ttrue,
    bkg_ttrue,
    mc_has_is_background,
    use_bkg_hit_truth,
    use_bkg_nonhit_truth,
    use_bkg_scalar_truth,
):
    """
    Build output chunks with resolution enabled.
    """
    n_chunk = len(mc_wcte["event_number"])

    resolved_wcte_lists = {
        branch: []
        for branch in wcte_hit_branches
    }

    resolved_truth_lists = {
        branch: []
        for branch in ttrue_hit_branches + ["is_background"]
    }

    chunk_stats = {
        "hits_before_resolution": 0,
        "hits_after_resolution": 0,
        "mixed_background_nonbackground_clusters": 0,
    }

    for iw in range(n_chunk):

        wcte_hits, truth_hits = build_premerged_window(
            mc_wcte,
            bkg_wcte,
            mc_ttrue,
            bkg_ttrue,
            iw,
            mc_has_is_background,
            use_bkg_hit_truth,
        )

        resolved_wcte, resolved_truth, stats = apply_resolution_to_premerged_window(
            wcte_hits,
            truth_hits,
            resolution_ns,
        )

        chunk_stats["hits_before_resolution"] += stats["hits_before"]
        chunk_stats["hits_after_resolution"] += stats["hits_after"]
        chunk_stats["mixed_background_nonbackground_clusters"] += (
            stats["mixed_background_nonbackground_clusters"]
        )

        for branch in wcte_hit_branches:
            resolved_wcte_lists[branch].append(
                resolved_wcte[branch]
            )

        for branch in ttrue_hit_branches + ["is_background"]:
            resolved_truth_lists[branch].append(
                resolved_truth[branch]
            )

    # WCTE output
    wcte_out = {}

    for branch in wcte_branches:
        dtype = wcte_dtypes[branch]

        if branch in wcte_hit_branches:
            wcte_out[branch] = list_of_numpy_to_ak(
                resolved_wcte_lists[branch],
                dtype,
            )
        else:
            wcte_out[branch] = cast_branch(
                mc_wcte[branch],
                dtype,
            )

    # TTrueInfo output
    ttrue_out = {}

    for branch in ttrue_scalar_branches:
        ttrue_out[branch] = merge_scalar_branch(
            mc_ttrue,
            bkg_ttrue,
            branch,
            use_bkg_scalar_truth,
        )

    for branch in ttrue_hit_branches + ["is_background"]:
        ttrue_out[branch] = list_of_numpy_to_ak(
            resolved_truth_lists[branch],
            ttrue_dtypes[branch],
        )

    for branch in ttrue_nonhit_branches:
        ttrue_out[branch] = merge_nonhit_truth_branch(
            mc_ttrue,
            bkg_ttrue,
            branch,
            use_bkg_nonhit_truth,
        )

    return wcte_out, ttrue_out, chunk_stats


# ============================================================
# Main merge
# ============================================================

global_stats = {
    "hits_before_resolution": 0,
    "hits_after_resolution": 0,
    "mixed_background_nonbackground_clusters": 0,
}

with uproot.open(mc_root) as f_mc, uproot.open(bkg_root) as f_bkg:

    if "WCTEReadoutWindows" not in f_mc:
        raise RuntimeError(f"{mc_root} does not contain WCTEReadoutWindows")

    if "TTrueInfo" not in f_mc:
        raise RuntimeError(f"{mc_root} does not contain TTrueInfo")

    if "WCTEReadoutWindows" not in f_bkg:
        raise RuntimeError(f"{bkg_root} does not contain WCTEReadoutWindows")

    mc_wcte_tree = f_mc["WCTEReadoutWindows"]
    mc_ttrue_tree = f_mc["TTrueInfo"]
    bkg_wcte_tree = f_bkg["WCTEReadoutWindows"]

    bkg_has_ttrue = "TTrueInfo" in f_bkg
    bkg_ttrue_tree = f_bkg["TTrueInfo"] if bkg_has_ttrue else None

    mc_has_is_background = "is_background" in mc_ttrue_tree.keys()

    use_bkg_hit_truth = (
        use_bkg_truth_if_available
        and bkg_has_ttrue
        and all(branch in bkg_ttrue_tree.keys() for branch in ttrue_hit_branches)
    )

    use_bkg_nonhit_truth = (
        use_bkg_truth_if_available
        and bkg_has_ttrue
        and all(branch in bkg_ttrue_tree.keys() for branch in ttrue_nonhit_branches)
    )

    use_bkg_scalar_truth = (
        use_bkg_truth_if_available
        and bkg_has_ttrue
        and "n_overlay" in bkg_ttrue_tree.keys()
    )

    print("mc_root has TTrueInfo/is_background:", mc_has_is_background)
    print("bkg_root has TTrueInfo:", bkg_has_ttrue)
    print("Using bkg hit-level truth:", use_bkg_hit_truth)
    print("Using bkg non-hit truth:", use_bkg_nonhit_truth)
    print("Using bkg n_overlay:", use_bkg_scalar_truth)

    check_required_branches(
        mc_wcte_tree,
        wcte_branches,
        "WCTEReadoutWindows",
        mc_root,
    )

    check_required_branches(
        bkg_wcte_tree,
        wcte_hit_branches,
        "WCTEReadoutWindows",
        bkg_root,
    )

    check_required_branches(
        mc_ttrue_tree,
        ttrue_required_mc_branches,
        "TTrueInfo",
        mc_root,
    )

    if use_bkg_hit_truth:
        check_required_branches(
            bkg_ttrue_tree,
            ttrue_hit_branches,
            "TTrueInfo",
            bkg_root,
        )

    if use_bkg_nonhit_truth:
        check_required_branches(
            bkg_ttrue_tree,
            ttrue_nonhit_branches,
            "TTrueInfo",
            bkg_root,
        )

    n_mc = mc_wcte_tree.num_entries
    n_bkg = bkg_wcte_tree.num_entries

    print()
    print("MC windows:         ", n_mc)
    print("Background windows: ", n_bkg)
    print("Resolution after merge:", apply_resolution_after_merge)

    if require_same_n_windows and n_mc != n_bkg:
        raise RuntimeError(
            f"Number of windows differs: MC has {n_mc}, background has {n_bkg}"
        )

    n_merge = min(n_mc, n_bkg)

    if n_mc != n_bkg:
        print()
        print("WARNING:")
        print(f"  Number of windows differs: MC={n_mc}, background={n_bkg}")
        print(f"  Merging only the first {n_merge} windows.")

    if n_merge == 0:
        raise RuntimeError("No windows to merge.")

    # Branches to read from mc TTrueInfo
    mc_ttrue_read_branches = list(ttrue_required_mc_branches)

    if mc_has_is_background:
        mc_ttrue_read_branches.append("is_background")

    # Branches to read from bkg TTrueInfo, if used
    bkg_ttrue_read_branches = []

    if use_bkg_hit_truth:
        bkg_ttrue_read_branches += ttrue_hit_branches

    if use_bkg_nonhit_truth:
        bkg_ttrue_read_branches += ttrue_nonhit_branches

    if use_bkg_scalar_truth:
        bkg_ttrue_read_branches += ["n_overlay"]

    # Remove duplicates while preserving order
    bkg_ttrue_read_branches = list(dict.fromkeys(bkg_ttrue_read_branches))

    desc = (
        "Merging + applying resolution"
        if apply_resolution_after_merge
        else "Fast merging ROOT files"
    )

    with uproot.recreate(output_root) as f_out:

        first_chunk = True

        for chunk_idx, start in enumerate(
            tqdm(
                range(0, n_merge, chunk_size),
                desc=desc,
                unit="chunk",
            )
        ):
            stop = min(start + chunk_size, n_merge)

            mc_wcte = mc_wcte_tree.arrays(
                wcte_branches,
                entry_start=start,
                entry_stop=stop,
                library="ak",
            )

            mc_ttrue = mc_ttrue_tree.arrays(
                mc_ttrue_read_branches,
                entry_start=start,
                entry_stop=stop,
                library="ak",
            )

            bkg_wcte = bkg_wcte_tree.arrays(
                wcte_hit_branches,
                entry_start=start,
                entry_stop=stop,
                library="ak",
            )

            if len(bkg_ttrue_read_branches) > 0:
                bkg_ttrue = bkg_ttrue_tree.arrays(
                    bkg_ttrue_read_branches,
                    entry_start=start,
                    entry_stop=stop,
                    library="ak",
                )
            else:
                bkg_ttrue = None

            # ------------------------------------------------
            # Length checks
            # ------------------------------------------------
            mc_hit_lengths = ak.to_numpy(
                ak.num(mc_wcte[resolution_time_branch], axis=1)
            )

            bkg_hit_lengths = ak.to_numpy(
                ak.num(bkg_wcte[resolution_time_branch], axis=1)
            )

            mc_truth_lengths = ak.to_numpy(
                ak.num(mc_ttrue["x"], axis=1)
            )

            if not np.array_equal(mc_hit_lengths, mc_truth_lengths):
                raise RuntimeError(
                    "MC WCTE hit counts and MC TTrueInfo/x counts do not match "
                    f"in chunk {start}:{stop}."
                )

            if mc_has_is_background:
                mc_label_lengths = ak.to_numpy(
                    ak.num(mc_ttrue["is_background"], axis=1)
                )

                if not np.array_equal(mc_hit_lengths, mc_label_lengths):
                    raise RuntimeError(
                        "MC WCTE hit counts and MC is_background counts do not match "
                        f"in chunk {start}:{stop}."
                    )

            if use_bkg_hit_truth:
                bkg_truth_lengths = ak.to_numpy(
                    ak.num(bkg_ttrue["x"], axis=1)
                )

                if not np.array_equal(bkg_hit_lengths, bkg_truth_lengths):
                    raise RuntimeError(
                        "BKG WCTE hit counts and BKG TTrueInfo/x counts do not match "
                        f"in chunk {start}:{stop}."
                    )

            # ------------------------------------------------
            # Build output chunk
            # ------------------------------------------------
            if apply_resolution_after_merge:

                wcte_out, ttrue_out, chunk_stats = build_chunks_with_resolution(
                    mc_wcte,
                    bkg_wcte,
                    mc_ttrue,
                    bkg_ttrue,
                    mc_has_is_background,
                    use_bkg_hit_truth,
                    use_bkg_nonhit_truth,
                    use_bkg_scalar_truth,
                )

            else:

                wcte_out = build_wcte_fast_no_resolution(
                    mc_wcte,
                    bkg_wcte,
                )

                ttrue_out = build_ttrue_fast_no_resolution(
                    mc_ttrue,
                    bkg_ttrue,
                    mc_hit_lengths,
                    bkg_hit_lengths,
                    mc_has_is_background,
                    use_bkg_hit_truth,
                    use_bkg_nonhit_truth,
                    use_bkg_scalar_truth,
                )

                n_hits_chunk = int(
                    np.sum(mc_hit_lengths + bkg_hit_lengths)
                )

                chunk_stats = {
                    "hits_before_resolution": n_hits_chunk,
                    "hits_after_resolution": n_hits_chunk,
                    "mixed_background_nonbackground_clusters": 0,
                }

            # ------------------------------------------------
            # Output consistency checks
            # ------------------------------------------------
            out_hit_lengths = ak.to_numpy(
                ak.num(wcte_out[resolution_time_branch], axis=1)
            )

            out_label_lengths = ak.to_numpy(
                ak.num(ttrue_out["is_background"], axis=1)
            )

            if not np.array_equal(out_hit_lengths, out_label_lengths):
                raise RuntimeError(
                    "Output WCTE hit counts and is_background counts do not match "
                    f"in chunk {start}:{stop}."
                )

            for branch in ttrue_hit_branches:
                out_truth_lengths = ak.to_numpy(
                    ak.num(ttrue_out[branch], axis=1)
                )

                if not np.array_equal(out_hit_lengths, out_truth_lengths):
                    raise RuntimeError(
                        f"Output WCTE hit counts and TTrueInfo/{branch} counts "
                        f"do not match in chunk {start}:{stop}."
                    )

            # ------------------------------------------------
            # Write
            # ------------------------------------------------
            if first_chunk:
                f_out["WCTEReadoutWindows"] = wcte_out
                f_out["TTrueInfo"] = ttrue_out
                first_chunk = False
            else:
                f_out["WCTEReadoutWindows"].extend(wcte_out)
                f_out["TTrueInfo"].extend(ttrue_out)

            for key in global_stats:
                global_stats[key] += chunk_stats[key]

            del mc_wcte, mc_ttrue, bkg_wcte, bkg_ttrue, wcte_out, ttrue_out

            if gc_every_n_chunks is not None:
                if (chunk_idx + 1) % gc_every_n_chunks == 0:
                    gc.collect()


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print(f"Wrote merged file: {output_root.resolve()}")
print(f"Number of merged windows: {n_merge}")

print()
print("Output trees:")
print("  WCTEReadoutWindows")
print("  TTrueInfo")

print()
print("Output TTrueInfo/is_background behavior:")
print("  mc_root: kept if it already existed; otherwise created as False")
print("  bkg_root: always set to True")

print()
print("Resolution summary:")
print("  Resolution applied:", apply_resolution_after_merge)
print("  Hits before resolution:", int(global_stats["hits_before_resolution"]))
print("  Hits after resolution: ", int(global_stats["hits_after_resolution"]))
print(
    "  Hits merged away:      ",
    int(
        global_stats["hits_before_resolution"]
        - global_stats["hits_after_resolution"]
    ),
)

if global_stats["hits_before_resolution"] > 0:
    frac_removed = (
        100.0
        * (
            global_stats["hits_before_resolution"]
            - global_stats["hits_after_resolution"]
        )
        / global_stats["hits_before_resolution"]
    )
    print(f"  Fraction removed:      {frac_removed:.3f}%")

print(
    "  Mixed background/non-background clusters:",
    int(global_stats["mixed_background_nonbackground_clusters"]),
)