#!/usr/bin/env python3
# Usage:
#   python inspect_AmBe_event.py ../simulation/output/wcte_ambe_000.root
#   python inspect_AmBe_event.py ../simulation/output/wcte_ambe_000.root <entry> <trigger>

import sys
import os
import ROOT

def print_file_structure(root_dir, indent=""):
    for key in root_dir.GetListOfKeys():
        name = key.GetName()
        classname = key.GetClassName()
        print(f"{indent}{name}  [{classname}]")

        obj = key.ReadObj()
        if obj.InheritsFrom("TDirectory"):
            print_file_structure(obj, indent + "  ")


def print_tree_branches(tree, max_leaves=10):
    print(f"\nTree: {tree.GetName()}")
    print(f"  Entries: {tree.GetEntries()}")
    print("  Branches:")
    for br in tree.GetListOfBranches():
        print(f"    - {br.GetName()}")
        leaves = br.GetListOfLeaves()
        if leaves and leaves.GetEntries() > 0:
            nshow = min(leaves.GetEntries(), max_leaves)
            for i in range(nshow):
                leaf = leaves.At(i)
                print(f"        leaf: {leaf.GetName()} ({leaf.GetTypeName()})")
            if leaves.GetEntries() > nshow:
                print(f"        ... {leaves.GetEntries() - nshow} more leaves")


def get_track_info(track):
    """Extract common fields from a WCSimRootTrack."""
    def call_first(obj, names, default=None):
        for n in names:
            if hasattr(obj, n):
                try:
                    return getattr(obj, n)()
                except Exception:
                    pass
        return default

    track_id = call_first(track, ["GetId", "GetTrackID"])
    parent_id = call_first(track, ["GetParentId", "GetParentID"])
    pdg = call_first(track, ["GetIpnu", "GetPDG"])
    energy = call_first(track, ["GetE", "GetEnergy"])
    time = call_first(track, ["GetTime"], default=None)

    start_x = track.GetStart(0) if hasattr(track, "GetStart") else None
    start_y = track.GetStart(1) if hasattr(track, "GetStart") else None
    start_z = track.GetStart(2) if hasattr(track, "GetStart") else None

    stop_x = track.GetStop(0) if hasattr(track, "GetStop") else None
    stop_y = track.GetStop(1) if hasattr(track, "GetStop") else None
    stop_z = track.GetStop(2) if hasattr(track, "GetStop") else None

    creator = track.GetCreatorProcessName()

    return {
        "track_id": track_id,
        "parent_id": parent_id,
        "pdg": pdg,
        "energy": energy,
        "time": time,
        "start_x": start_x,
        "start_y": start_y,
        "start_z": start_z,
        "stop_x": stop_x,
        "stop_y": stop_y,
        "stop_z": stop_z,
        "creator": creator,
    }


def is_ambe_prompt_source_track(info):
    """
    Same prompt-source rule as in the extractor:
      trackID = 2, parentID = 0, PDG = 22, E > 0
    """
    try:
        return (
            info["track_id"] == 2 and
            info["parent_id"] == 0 and
            info["pdg"] == 22 and
            info["energy"] is not None and
            abs(float(info["energy"])) > 1e-9
        )
    except Exception:
        return False


def inspect_event_summary(event, subevent=0):
    """Print a compact summary of one trigger before dumping tracks."""
    n_triggers = event.GetNumberOfEvents()
    print(f"\nNumber of subevents/triggers in this entry: {n_triggers}")

    if subevent >= n_triggers:
        print(f"Requested subevent {subevent}, but only {n_triggers} available")
        return None

    trigger = event.GetTrigger(subevent)
    print(f"\nInspecting subevent/trigger {subevent}")

    # -------------------------
    # Track counts
    # -------------------------
    tracks = trigger.GetTracks() if hasattr(trigger, "GetTracks") else None
    if hasattr(trigger, "GetNtrack"):
        n_tracks = trigger.GetNtrack()
    else:
        try:
            n_tracks = tracks.GetEntries()
        except Exception:
            n_tracks = 0

    # -------------------------
    # True-hit counts
    # -------------------------
    hits = trigger.GetCherenkovHits() if hasattr(trigger, "GetCherenkovHits") else None
    hit_times = trigger.GetCherenkovHitTimes() if hasattr(trigger, "GetCherenkovHitTimes") else None

    if hasattr(trigger, "GetNcherenkovhits"):
        n_hit_pmts = trigger.GetNcherenkovhits()
    else:
        try:
            n_hit_pmts = hits.GetEntries()
        except Exception:
            n_hit_pmts = 0

    if hasattr(trigger, "GetNcherenkovhittimes"):
        n_true_hits = trigger.GetNcherenkovhittimes()
    else:
        try:
            n_true_hits = hit_times.GetEntries()
        except Exception:
            n_true_hits = 0

    # -------------------------
    # Digi-hit counts
    # -------------------------
    digis = trigger.GetCherenkovDigiHits() if hasattr(trigger, "GetCherenkovDigiHits") else None

    if hasattr(trigger, "GetNcherenkovdigihits"):
        n_digi = trigger.GetNcherenkovdigihits()
    else:
        try:
            n_digi = digis.GetEntries()
        except Exception:
            n_digi = 0

    # Number of slots in the digi-hit array
    n_digi_slots = trigger.GetNcherenkovdigihits_slots()

    # -------------------------
    # Capture summary
    # -------------------------
    captures = trigger.GetCaptures() if hasattr(trigger, "GetCaptures") else None
    if hasattr(trigger, "GetNcaptures"):
        n_captures = trigger.GetNcaptures()
    else:
        try:
            n_captures = captures.GetEntries()
        except Exception:
            n_captures = 0

    # -------------------------
    # Prompt summary using the same rule as the extractor
    # -------------------------
    has_prompt = False
    prompt_info = None
    if tracks is not None:
        for i in range(n_tracks):
            tr = tracks.At(i)
            if not tr:
                continue
            info = get_track_info(tr)
            if is_ambe_prompt_source_track(info):
                has_prompt = True
                prompt_info = info
                break

    print("\n=== EVENT / SUBEVENT SUMMARY ===")
    print(f"  Number of tracks (GetNtrack): {n_tracks}")
    print(f"  Number of Cherenkov hit PMT groups: {n_hit_pmts}")
    print(f"  Number of true hit-times (matches extracted nhits): {n_true_hits}")
    if n_digi_slots is None:
        print(f"  Number of digi hits: {n_digi}")
    else:
        print(f"  Number of digi hits: {n_digi}   (slots: {n_digi_slots})")
    print(f"  Number of captures: {n_captures}")
    print(f"  Has AmBe prompt source track: {int(has_prompt)}")

    if prompt_info is not None:
        print("  Prompt track summary:")
        print(f"    trackID={prompt_info['track_id']}")
        print(f"    parentTrackID={prompt_info['parent_id']}")
        print(f"    PDG={prompt_info['pdg']}")
        print(f"    E={prompt_info['energy']}")
        print(f"    time={prompt_info['time']}")
        print(f"    creator={prompt_info['creator']}")

    if n_captures > 0 and captures is not None:
        print("\n  Capture list:")
        for ic in range(n_captures):
            cap = captures.At(ic)
            if not cap:
                continue

            try:
                cap_parent = cap.GetCaptureParent()
            except Exception:
                cap_parent = None

            try:
                cap_t = cap.GetCaptureT()
            except Exception:
                cap_t = None

            try:
                cap_x = cap.GetCaptureVtx(0)
                cap_y = cap.GetCaptureVtx(1)
                cap_z = cap.GetCaptureVtx(2)
            except Exception:
                cap_x, cap_y, cap_z = None, None, None

            try:
                cap_nucleus = cap.GetCaptureNucleus()
            except Exception:
                cap_nucleus = None

            try:
                cap_ngamma = cap.GetNGamma()
            except Exception:
                cap_ngamma = None

            try:
                cap_total_gammaE = cap.GetTotalGammaE()
            except Exception:
                cap_total_gammaE = None

            print(
                f"    [{ic}] "
                f"t={cap_t} ns  "
                f"vtx=({cap_x}, {cap_y}, {cap_z})  "
                f"parent={cap_parent}  "
                f"nucleus={cap_nucleus}  "
                f"ngamma={cap_ngamma}  "
                f"total_gammaE={cap_total_gammaE}"
            )

    return trigger


def inspect_tracks(event, subevent=0, max_tracks=200):
    """Inspect tracks in a given trigger/subevent."""
    trigger = inspect_event_summary(event, subevent=subevent)
    if trigger is None:
        return

    if not hasattr(trigger, "GetTracks"):
        print("Could not access tracks with trigger.GetTracks()")
        return

    tracks = trigger.GetTracks()

    if hasattr(trigger, "GetNtrack"):
        n_tracks = trigger.GetNtrack()
    else:
        try:
            n_tracks = tracks.GetEntries()
        except Exception:
            n_tracks = 0

    print(f"\n=== TRACK LIST (showing up to {max_tracks}) ===")
    print(f"Number of valid tracks (GetNtrack): {n_tracks}")

    prompt_candidates = []

    for i in range(min(n_tracks, max_tracks)):
        track = tracks.At(i)
        if not track:
            continue

        info = get_track_info(track)

        print(
            f"[{i:3d}] "
            f"trackID={info['track_id']}  "
            f"parentTrackID={info['parent_id']}  "
            f"PDG={info['pdg']}  "
            f"E={info['energy']}  "
            f"creator={info['creator']}  "
            f"start=({info['start_x']}, {info['start_y']}, {info['start_z']})  "
            f"stop=({info['stop_x']}, {info['stop_y']}, {info['stop_z']})"
        )

        creator_lower = str(info["creator"]).lower()

        if (
            info["track_id"] is not None and info["track_id"] > 0 and
            info["pdg"] is not None and info["pdg"] != 0 and
            info["parent_id"] == 0 and
            creator_lower == "initial"
        ):
            prompt_candidates.append(info)

    print("\nPrimary tracks (prompt candidate and initial neutron):")
    if not prompt_candidates:
        print("  None found")
    else:
        for info in prompt_candidates:
            print(
                f"  trackID={info['track_id']}  "
                f"parentTrackID={info['parent_id']}  "
                f"PDG={info['pdg']}  "
                f"creator={info['creator']}"
            )


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_wcsim_structure.py file.root [entry] [subevent]")
        sys.exit(1)

    filename = sys.argv[1]
    entry = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    subevent = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        sys.exit(1)

    load_status = ROOT.gSystem.Load("libWCSimRoot.so")
    if load_status < 0:
        print("Warning: could not explicitly load libWCSimRoot.so. Continuing anyway.")

    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        print(f"Could not open ROOT file: {filename}")
        sys.exit(1)

    print("\n=== FILE STRUCTURE ===")
    print_file_structure(f)

    print("\n=== TREE SUMMARY ===")
    for tree_name in ["wcsimT", "wcsimGeoT", "wcsimRootOptionsT"]:
        obj = f.Get(tree_name)
        if obj and obj.InheritsFrom("TTree"):
            print_tree_branches(obj)

    tree = f.Get("wcsimT")
    if not tree:
        print("\nCould not find tree 'wcsimT'")
        f.Close()
        sys.exit(1)

    print(f"\n=== INSPECT ENTRY {entry}, SUBEVENT {subevent} ===")

    event = ROOT.WCSimRootEvent()
    branch = tree.GetBranch("wcsimrootevent")
    if not branch:
        print("Could not find branch 'wcsimrootevent'")
        f.Close()
        sys.exit(1)

    branch.SetAddress(ROOT.AddressOf(event))
    tree.GetEntry(entry)

    inspect_tracks(event, subevent=subevent)

    f.Close()


if __name__ == "__main__":
    main()