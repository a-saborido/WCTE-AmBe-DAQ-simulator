// Each row in TTrueHits is one true PMT hit, usually corresponding to one
// optical photon that reached the PMT and produced that hit, except for
// dark-noise rows (turned off).
//
// Each row in TDigiHits is one digitized hit.
//
// Both trees are written to the same output file.

#include <algorithm>
#include <cmath>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "TClonesArray.h"
#include "TFile.h"
#include "TTree.h"

#include "WCSimRootEvent.hh"
#include "WCSimRootGeom.hh"

// -----------------------------------------------------------------------------
// Helper structure for storing a compact copy of the truth-track information.
//
// This is built explicitly because two tasks require it:
//
// 1) identify whether the event contains the AmBe prompt gamma source track
// 2) trace each true PMT hit back through the track-parent chain
//
// The important part for the ancestry logic is:
//   track id  -> parent id
//
// Then for a hit whose parent track is X, the chain can be followed as
//   X -> parent(X) -> parent(parent(X)) -> ...
// and checked against the prompt source track.
// -----------------------------------------------------------------------------
struct TrackInfo {
  int id = -1;
  int parent_id = -1;
  int pdg = 0;
  float time = 0.0f;
  float energy = 0.0f;
  int creator_process = -1;
  std::string creator_process_name;
};

// -----------------------------------------------------------------------------
// One row per true hit before sorting and copying into output vectors
// -----------------------------------------------------------------------------
struct TrueHitRow {
  int tubeid;
  int mPMTid;
  int mPMT_pmtid;

  // PMT geometry
  float x, y, z;
  float dx, dy, dz;

  // True-hit / photon info
  double truetime;
  int parent_track_id;   // kept internally for ancestry tagging
  float photon_end_energy;

  // Creator-process truth
  int creator_process;

  // Per-hit label:
  // 1 if this true hit descends from the prompt AmBe source gamma track,
  // 0 otherwise.
  int hit_from_prompt;

  // Per-hit label:
  // 1 if this true hit has any ancestor track whose creator process is capture,
  // 0 otherwise.
  int hit_from_capture;

  // Per-hit label:
  // 1 if this true hit is not from prompt and not from capture,
  // 0 otherwise.
  int hit_from_otherprocess;
};

// -----------------------------------------------------------------------------
// One row per digi hit before sorting and copying into output vectors
// -----------------------------------------------------------------------------
struct DigiHitRow {
  int tubeid;
  int mPMTid;
  int mPMT_pmtid;
  float q;
  double t;

  // PMT geometry
  float x, y, z;
  float dx, dy, dz;

  // Truth composition of the digit
  int ntruth;

  // Per-digit label:
  // 1 if at least one contributing true photon descends from the prompt
  // AmBe source gamma track, 0 otherwise.
  int hit_from_prompt;

  // Per-digit label:
  // 1 if at least one contributing true photon has any ancestor track whose
  // creator process is capture, 0 otherwise.
  int hit_from_capture;

  // Per-digit label:
  // 1 if this digit is not from prompt and not from capture,
  // 0 otherwise.
  int hit_from_otherprocess;
};

// -----------------------------------------------------------------------------
// In AmBe files the prompt-like source gamma is represented by a very
// specific truth track:
//
//   trackID = 2
//   parentID = 0
//   PDG = 22
//   E > 0
//
// and when there is no prompt, the same "slot" still exists but with E = 0.
//
// This function encodes that rule.
// -----------------------------------------------------------------------------
static bool IsAmBePromptSourceTrack(const TrackInfo& tr)
{
  return (tr.id == 2 &&
          tr.parent_id == 0 &&
          tr.pdg == 22 &&
          std::abs(tr.energy) > 1e-9f);
}

// -----------------------------------------------------------------------------
// Convert a string to lowercase so creator-process name matching is robust
// against capitalisation differences such as "Capture" vs "nCapture".
// -----------------------------------------------------------------------------
static std::string ToLower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// -----------------------------------------------------------------------------
// Check whether a creator-process name corresponds to capture.
//
// Using a substring match allows names such as "capture" or "nCapture" to be
// treated consistently.
// -----------------------------------------------------------------------------
static bool IsCaptureCreatorProcessName(const std::string& process_name)
{
  const std::string lower = ToLower(process_name);
  return lower.find("capture") != std::string::npos;
}

// -----------------------------------------------------------------------------
// Follow the ancestry chain of a given track ID and check whether any track in
// that chain was created by a capture process.
//
// The starting track itself is included in the check because for a PMT hit the
// saved parent track of the photon is already part of the ancestry chain.
//
// A "visited" set protects against malformed ancestry chains so that a
// pathological loop does not produce an infinite loop.
// -----------------------------------------------------------------------------
static bool HasCaptureAncestor(
    int start_track_id,
    const std::unordered_map<int, TrackInfo>& track_map)
{
  if (start_track_id < 0) return false;

  int current = start_track_id;
  std::unordered_set<int> visited;

  while (true) {
    if (current <= 0) {
      return false;
    }

    if (visited.count(current)) {
      // Safety guard against pathological parent loops.
      return false;
    }
    visited.insert(current);

    auto it = track_map.find(current);
    if (it == track_map.end()) {
      // If the current track is not present in the stored truth-track list,
      // the ancestry chain cannot be continued.
      return false;
    }

    if (IsCaptureCreatorProcessName(it->second.creator_process_name)) {
      return true;
    }

    current = it->second.parent_id;
  }
}

// -----------------------------------------------------------------------------
// Follow the parent chain of a given track ID and check whether it eventually
// reaches the selected prompt source track.
//
// Example:
//
//   hit parent track = 57
//   parent(57) = 14
//   parent(14) = 2
//
// If prompt_source_track_id == 2, then this hit came from the prompt chain.
//
// A "visited" set protects against malformed ancestry chains so that a
// pathological loop does not produce an infinite loop.
// -----------------------------------------------------------------------------
static bool DescendsFromPromptTrack(
    int start_track_id,
    int prompt_source_track_id,
    const std::unordered_map<int, TrackInfo>& track_map)
{
  if (prompt_source_track_id < 0) return false;
  if (start_track_id < 0) return false;

  int current = start_track_id;
  std::unordered_set<int> visited;

  while (true) {
    if (current == prompt_source_track_id) {
      return true;
    }

    if (current <= 0) {
      return false;
    }

    if (visited.count(current)) {
      // Safety guard against pathological parent loops.
      return false;
    }
    visited.insert(current);

    auto it = track_map.find(current);
    if (it == track_map.end()) {
      // If the current track is not present in the stored truth-track list,
      // the ancestry chain cannot be continued.
      return false;
    }

    current = it->second.parent_id;
  }
}

void extract_AmBe_hits(
    const char* infile = "../simulation/output/wcte_ambe_merged.root",   // change if needed
    const char* outfile = "../simulation/output/wcte_ambe_merged_hits.root",
    const char* event_branch = "wcsimrootevent")
{
  TFile* fin = TFile::Open(infile, "READ");
  if (!fin || fin->IsZombie()) {
    throw std::runtime_error(std::string("Could not open input file: ") + infile);
  }

  TTree* eventTree = dynamic_cast<TTree*>(fin->Get("wcsimT"));
  TTree* geoTree   = dynamic_cast<TTree*>(fin->Get("wcsimGeoT"));
  if (!eventTree || !geoTree) {
    throw std::runtime_error("Could not find wcsimT and/or wcsimGeoT in input file.");
  }

  WCSimRootEvent* superevent = nullptr;
  WCSimRootGeom* geo = nullptr;
  eventTree->SetBranchAddress(event_branch, &superevent);
  geoTree->SetBranchAddress("wcsimrootgeom", &geo);
  geoTree->GetEntry(0);

  TFile* fout = TFile::Open(outfile, "RECREATE");
  if (!fout || fout->IsZombie()) {
    throw std::runtime_error(std::string("Could not create output file: ") + outfile);
  }

  TTree* out_true = new TTree("TTrueHits", "Ordered per-event true-hit information extracted from WCSim");
  TTree* out_digi = new TTree("TDigiHits", "Ordered per-event digi-hit information extracted from WCSim");

  // -------------------------
  // Shared event-level info
  // -------------------------
  int event = 0;
  int nsubevents = 0;
  int subevent = 0;
  int nhits = 0;
  int ndigi = 0;
  int ncaptures = 0;
  float sumQ = 0.0f;

  // -------------------------
  // Shared event-level prompt tags
  // -------------------------
  //
  // has_prompt:
  //   1 if the AmBe prompt source gamma track exists in this trigger according
  //   to the rule:
  //       trackID = 2, parentID = 0, PDG = 22, E > 0
  //   0 otherwise.
  //
  // The additional prompt_track_* branches are retained as debugging /
  // validation aids for inspecting what was matched.
  // -------------------------
  int has_prompt = 0;
  int prompt_track_id = -1;
  int prompt_track_pdg = 0;
  float prompt_track_energy = 0.0f;
  float prompt_track_time = -9999.0f;
  int prompt_track_creator_process = -1;

  // -------------------------
  // True-hit output vectors
  // -------------------------
  std::vector<int> true_tubeid;
  std::vector<int> true_mPMTid;
  std::vector<int> true_mPMT_pmtid;

  std::vector<float> true_x, true_y, true_z;
  std::vector<float> true_dx, true_dy, true_dz;

  std::vector<double> true_truetime;
  std::vector<float> true_photon_end_energy;
  std::vector<int> true_creator_process;
  std::vector<int> true_hit_from_prompt;
  std::vector<int> true_hit_from_capture;
  std::vector<int> true_hit_from_otherprocess;

  // -------------------------
  // Digi-hit output vectors
  // -------------------------
  std::vector<int> digi_tubeid;
  std::vector<int> digi_mPMTid;
  std::vector<int> digi_mPMT_pmtid;

  std::vector<float> digi_q;
  std::vector<double> digi_t;

  std::vector<float> digi_x, digi_y, digi_z;
  std::vector<float> digi_dx, digi_dy, digi_dz;

  std::vector<int> digi_ntruth;
  std::vector<int> digi_hit_from_prompt;
  std::vector<int> digi_hit_from_capture;
  std::vector<int> digi_hit_from_otherprocess;

  // -------------------------
  // Shared capture output vectors
  // -------------------------
  std::vector<float> capture_t;
  std::vector<float> capture_x, capture_y, capture_z;
  std::vector<int> capture_nucleus;
  std::vector<int> capture_ngamma;
  std::vector<float> capture_total_gammaE;

  // -------------------------
  // TTrueHits branches
  // -------------------------
  out_true->Branch("event", &event, "event/I");
  out_true->Branch("nsubevents", &nsubevents, "nsubevents/I");
  out_true->Branch("subevent", &subevent, "subevent/I");
  out_true->Branch("nhits", &nhits, "nhits/I");
  out_true->Branch("ncaptures", &ncaptures, "ncaptures/I");

  out_true->Branch("has_prompt", &has_prompt, "has_prompt/I");
  out_true->Branch("prompt_track_id", &prompt_track_id, "prompt_track_id/I");
  out_true->Branch("prompt_track_pdg", &prompt_track_pdg, "prompt_track_pdg/I");
  out_true->Branch("prompt_track_energy", &prompt_track_energy, "prompt_track_energy/F");
  out_true->Branch("prompt_track_time", &prompt_track_time, "prompt_track_time/F");
  out_true->Branch("prompt_track_creator_process",
                   &prompt_track_creator_process,
                   "prompt_track_creator_process/I");

  out_true->Branch("tubeid", &true_tubeid);
  out_true->Branch("mPMTid", &true_mPMTid);
  out_true->Branch("mPMT_pmtid", &true_mPMT_pmtid);

  out_true->Branch("x", &true_x);
  out_true->Branch("y", &true_y);
  out_true->Branch("z", &true_z);
  out_true->Branch("dx", &true_dx);
  out_true->Branch("dy", &true_dy);
  out_true->Branch("dz", &true_dz);

  out_true->Branch("truetime", &true_truetime);
  out_true->Branch("photon_end_energy", &true_photon_end_energy);
  out_true->Branch("creator_process", &true_creator_process);
  out_true->Branch("hit_from_prompt", &true_hit_from_prompt);
  out_true->Branch("hit_from_capture", &true_hit_from_capture);
  out_true->Branch("hit_from_otherprocess", &true_hit_from_otherprocess);

  out_true->Branch("capture_t", &capture_t);
  out_true->Branch("capture_x", &capture_x);
  out_true->Branch("capture_y", &capture_y);
  out_true->Branch("capture_z", &capture_z);
  out_true->Branch("capture_nucleus", &capture_nucleus);
  out_true->Branch("capture_ngamma", &capture_ngamma);
  out_true->Branch("capture_total_gammaE", &capture_total_gammaE);

  // -------------------------
  // TDigiHits branches
  // -------------------------
  out_digi->Branch("event", &event, "event/I");
  out_digi->Branch("nsubevents", &nsubevents, "nsubevents/I");
  out_digi->Branch("subevent", &subevent, "subevent/I");
  out_digi->Branch("ndigi", &ndigi, "ndigi/I");
  out_digi->Branch("ncaptures", &ncaptures, "ncaptures/I");
  out_digi->Branch("sumQ", &sumQ, "sumQ/F");

  out_digi->Branch("has_prompt", &has_prompt, "has_prompt/I");
  out_digi->Branch("prompt_track_id", &prompt_track_id, "prompt_track_id/I");
  out_digi->Branch("prompt_track_pdg", &prompt_track_pdg, "prompt_track_pdg/I");
  out_digi->Branch("prompt_track_energy", &prompt_track_energy, "prompt_track_energy/F");
  out_digi->Branch("prompt_track_time", &prompt_track_time, "prompt_track_time/F");
  out_digi->Branch("prompt_track_creator_process",
                   &prompt_track_creator_process,
                   "prompt_track_creator_process/I");

  out_digi->Branch("tubeid", &digi_tubeid);
  out_digi->Branch("mPMTid", &digi_mPMTid);
  out_digi->Branch("mPMT_pmtid", &digi_mPMT_pmtid);
  out_digi->Branch("q", &digi_q);
  out_digi->Branch("t", &digi_t);

  out_digi->Branch("x", &digi_x);
  out_digi->Branch("y", &digi_y);
  out_digi->Branch("z", &digi_z);
  out_digi->Branch("dx", &digi_dx);
  out_digi->Branch("dy", &digi_dy);
  out_digi->Branch("dz", &digi_dz);

  out_digi->Branch("ntruth", &digi_ntruth);
  out_digi->Branch("hit_from_prompt", &digi_hit_from_prompt);
  out_digi->Branch("hit_from_capture", &digi_hit_from_capture);
  out_digi->Branch("hit_from_otherprocess", &digi_hit_from_otherprocess);

  out_digi->Branch("capture_t", &capture_t);
  out_digi->Branch("capture_x", &capture_x);
  out_digi->Branch("capture_y", &capture_y);
  out_digi->Branch("capture_z", &capture_z);
  out_digi->Branch("capture_nucleus", &capture_nucleus);
  out_digi->Branch("capture_ngamma", &capture_ngamma);
  out_digi->Branch("capture_total_gammaE", &capture_total_gammaE);

  const Long64_t nevents = eventTree->GetEntries();
  std::cout << "Reading " << nevents << " entries from " << infile << std::endl;

  for (Long64_t ievt = 0; ievt < nevents; ++ievt) {
    eventTree->GetEntry(ievt);
    if (!superevent) continue;

    event = static_cast<int>(ievt);
    nsubevents = superevent->GetNumberOfEvents();

    for (int isub = 0; isub < nsubevents; ++isub) {
      WCSimRootTrigger* trig = superevent->GetTrigger(isub);
      if (!trig) continue;

      subevent = isub;
      sumQ = trig->GetSumQ();

      // -------------------------
      // Clear output vectors for this subevent
      // -------------------------
      true_tubeid.clear();
      true_mPMTid.clear();
      true_mPMT_pmtid.clear();
      true_x.clear();
      true_y.clear();
      true_z.clear();
      true_dx.clear();
      true_dy.clear();
      true_dz.clear();
      true_truetime.clear();
      true_photon_end_energy.clear();
      true_creator_process.clear();
      true_hit_from_prompt.clear();
      true_hit_from_capture.clear();
      true_hit_from_otherprocess.clear();

      digi_tubeid.clear();
      digi_mPMTid.clear();
      digi_mPMT_pmtid.clear();
      digi_q.clear();
      digi_t.clear();
      digi_x.clear();
      digi_y.clear();
      digi_z.clear();
      digi_dx.clear();
      digi_dy.clear();
      digi_dz.clear();
      digi_ntruth.clear();
      digi_hit_from_prompt.clear();
      digi_hit_from_capture.clear();
      digi_hit_from_otherprocess.clear();

      capture_t.clear();
      capture_x.clear();
      capture_y.clear();
      capture_z.clear();
      capture_nucleus.clear();
      capture_ngamma.clear();
      capture_total_gammaE.clear();

      // -------------------------
      // Reset prompt summary fields for this subevent
      // -------------------------
      has_prompt = 0;
      prompt_track_id = -1;
      prompt_track_pdg = 0;
      prompt_track_energy = 0.0f;
      prompt_track_time = -9999.0f;
      prompt_track_creator_process = -1;

      // -------------------------
      // Save capture information
      // -------------------------
      TClonesArray* captures = trig->GetCaptures();
      const int ncaptures_in = trig->GetNcaptures();

      for (int ic = 0; ic < ncaptures_in; ++ic) {
        auto* cap = captures ? dynamic_cast<WCSimRootCapture*>(captures->At(ic)) : nullptr;
        if (!cap) continue;

        capture_t.push_back(cap->GetCaptureT());
        capture_x.push_back(cap->GetCaptureVtx(0));
        capture_y.push_back(cap->GetCaptureVtx(1));
        capture_z.push_back(cap->GetCaptureVtx(2));

        const int nuc = cap->GetCaptureNucleus();
        capture_nucleus.push_back(nuc);

        capture_ngamma.push_back(cap->GetNGamma());
        capture_total_gammaE.push_back(cap->GetTotalGammaE());
      }

      ncaptures = static_cast<int>(capture_t.size());

      // -------------------------
      // Build a compact truth-track map for this trigger
      // -------------------------
      //
      // All tracks are stored in a map keyed by track ID so that prompt
      // detection and true-hit ancestry tagging can use the same track truth.
      // -------------------------
      std::unordered_map<int, TrackInfo> track_map;
      TClonesArray* tracks = trig->GetTracks();
      const int ntracks = tracks ? tracks->GetEntries() : 0;
      track_map.reserve(static_cast<size_t>(ntracks) + 8);

      for (int it = 0; it < ntracks; ++it) {
        auto* tr = tracks ? dynamic_cast<WCSimRootTrack*>(tracks->At(it)) : nullptr;
        if (!tr) continue;

        TrackInfo info;
        info.id = tr->GetId();

        // For ancestry reconstruction the parent TRACK ID is required, so the
        // correct getter here is GetParentId().
        info.parent_id = tr->GetParentId();

        info.pdg = tr->GetIpnu();
        info.time = tr->GetTime();
        info.energy = tr->GetE();
        info.creator_process = tr->GetCreatorProcess();
        info.creator_process_name = tr->GetCreatorProcessName();

        track_map[info.id] = info;
      }

      // -------------------------
      // Identify the AmBe prompt source track
      // -------------------------
      //
      // The rule is:
      //
      //   trackID = 2
      //   parentID = 0
      //   PDG = 22
      //   E > 0
      //
      // The corresponding "no prompt" case keeps the same source-track slot
      // but with E = 0.
      // -------------------------
      int prompt_source_track_id = -1;

      for (const auto& kv : track_map) {
        const TrackInfo& tr = kv.second;

        if (IsAmBePromptSourceTrack(tr)) {
          has_prompt = 1;
          prompt_source_track_id = tr.id;
          prompt_track_id = tr.id;
          prompt_track_pdg = tr.pdg;
          prompt_track_energy = tr.energy;
          prompt_track_time = tr.time;
          prompt_track_creator_process = tr.creator_process;
          break;
        }
      }

      // -------------------------
      // Gather TRUE hits
      //
      // WCSimRootCherenkovHit groups true hits by PMT.
      // For each PMT entry:
      //   GetTotalPe(0) = start index in CherenkovHitTimes
      //   GetTotalPe(1) = number of true hits on that PMT
      //
      // This is flattened into one row per true hit.
      // -------------------------
      std::vector<TrueHitRow> true_rows;
      true_rows.reserve(trig->GetNcherenkovhittimes());

      TClonesArray* hits = trig->GetCherenkovHits();
      TClonesArray* hitTimes = trig->GetCherenkovHitTimes();

      const int nhit_pmts = hits ? hits->GetEntries() : 0;
      const int nhit_times = hitTimes ? hitTimes->GetEntries() : 0;

      for (int ih = 0; ih < nhit_pmts; ++ih) {
        auto* hit = hits ? dynamic_cast<WCSimRootCherenkovHit*>(hits->At(ih)) : nullptr;
        if (!hit) continue;

        const int tube = hit->GetTubeID();   // 1-based WCSim tube ID
        const WCSimRootPMT* pmt = geo->GetPMTPtr(tube - 1, false);
        if (!pmt) continue;

        const int start = hit->GetTotalPe(0);
        const int nphot = hit->GetTotalPe(1);

        for (int j = 0; j < nphot; ++j) {
          const int idx = start + j;
          if (idx < 0 || idx >= nhit_times) continue;

          auto* htime = dynamic_cast<WCSimRootCherenkovHitTime*>(hitTimes->At(idx));
          if (!htime) continue;

          TrueHitRow row;
          row.tubeid = tube;
          row.mPMTid = pmt->GetmPMTNo();
          row.mPMT_pmtid = pmt->GetmPMT_PMTNo();

          row.x = pmt->GetPosition(0);
          row.y = pmt->GetPosition(1);
          row.z = pmt->GetPosition(2);

          row.dx = pmt->GetOrientation(0);
          row.dy = pmt->GetOrientation(1);
          row.dz = pmt->GetOrientation(2);

          row.truetime = htime->GetTruetime();
          row.parent_track_id = htime->GetParentSavedTrackID();
          row.photon_end_energy = htime->GetPhotonEndEnergy();

          const auto proc = htime->GetPhotonCreatorProcess();
          row.creator_process = static_cast<int>(proc);

          // -------------------------------------------------------------------
          // True-hit ancestry tagging:
          //
          // If the event has a prompt source track, the hit parent track is
          // checked against the prompt ancestry chain.
          //
          // This turns "event has prompt" into
          // "this specific hit belongs to the prompt lineage".
          // -------------------------------------------------------------------
          row.hit_from_prompt = 0;
          if (has_prompt) {
            row.hit_from_prompt =
                DescendsFromPromptTrack(row.parent_track_id,
                                        prompt_source_track_id,
                                        track_map) ? 1 : 0;
          }

          // -------------------------------------------------------------------
          // True-hit ancestry tagging for captures:
          //
          // The parent-track ancestry chain is also checked for any ancestor
          // whose creator process is capture.
          // -------------------------------------------------------------------
          row.hit_from_capture =
              HasCaptureAncestor(row.parent_track_id, track_map) ? 1 : 0;

          // -------------------------------------------------------------------
          // True-hit fallback tagging:
          //
          // If a hit is neither tagged as prompt nor capture, it is assigned to
          // the residual "other process" category.
          // -------------------------------------------------------------------
          row.hit_from_otherprocess =
              (row.hit_from_prompt == 0 && row.hit_from_capture == 0) ? 1 : 0;

          true_rows.push_back(row);
        }
      }

      // Sort by true-hit time, then by tube ID for stable ordering inside the
      // subevent. Event and subevent ordering remain unchanged in the output.
      std::sort(true_rows.begin(), true_rows.end(),
                [](const TrueHitRow& a, const TrueHitRow& b) {
                  if (a.truetime != b.truetime) return a.truetime < b.truetime;
                  return a.tubeid < b.tubeid;
                });

      for (const auto& row : true_rows) {
        true_tubeid.push_back(row.tubeid);
        true_mPMTid.push_back(row.mPMTid);
        true_mPMT_pmtid.push_back(row.mPMT_pmtid);

        true_x.push_back(row.x);
        true_y.push_back(row.y);
        true_z.push_back(row.z);

        true_dx.push_back(row.dx);
        true_dy.push_back(row.dy);
        true_dz.push_back(row.dz);

        true_truetime.push_back(row.truetime);
        true_photon_end_energy.push_back(row.photon_end_energy);
        true_creator_process.push_back(row.creator_process);
        true_hit_from_prompt.push_back(row.hit_from_prompt);
        true_hit_from_capture.push_back(row.hit_from_capture);
        true_hit_from_otherprocess.push_back(row.hit_from_otherprocess);
      }

      nhits = static_cast<int>(true_rows.size());

      // -------------------------
      // Gather digi hits
      //
      // Digi hits are collected independently, but the truth photons that
      // formed each digit are still available through photon IDs.
      // -------------------------
      std::vector<DigiHitRow> digi_rows;
      digi_rows.reserve(trig->GetNcherenkovdigihits());

      TClonesArray* digis = trig->GetCherenkovDigiHits();
      const int ndigi_slots = trig->GetNcherenkovdigihits_slots();

      for (int id = 0; id < ndigi_slots; ++id) {
        TObject* obj = digis ? digis->At(id) : nullptr;
        if (!obj) continue;

        auto* digi = dynamic_cast<WCSimRootCherenkovDigiHit*>(obj);
        if (!digi) continue;

        const int tube = digi->GetTubeId();   // 1-based WCSim tube ID
        const WCSimRootPMT* pmt = geo->GetPMTPtr(tube - 1, false);
        if (!pmt) continue;

        DigiHitRow row;
        row.tubeid = tube;
        row.mPMTid = pmt->GetmPMTNo();
        row.mPMT_pmtid = pmt->GetmPMT_PMTNo();
        row.q = digi->GetQ();
        row.t = digi->GetT();

        row.x = pmt->GetPosition(0);
        row.y = pmt->GetPosition(1);
        row.z = pmt->GetPosition(2);

        row.dx = pmt->GetOrientation(0);
        row.dy = pmt->GetOrientation(1);
        row.dz = pmt->GetOrientation(2);

        // Initialise truth composition counters
        row.ntruth = 0;
        row.hit_from_prompt = 0;
        row.hit_from_capture = 0;
        row.hit_from_otherprocess = 0;

        // Each digit points back to the true photons that formed it.
        // The creator process of each of those photons is inspected here.
        std::vector<int> photon_ids = digi->GetPhotonIds();

        for (int pid : photon_ids) {
          if (pid < 0 || pid >= nhit_times) continue;

          auto* htime = hitTimes ? dynamic_cast<WCSimRootCherenkovHitTime*>(hitTimes->At(pid)) : nullptr;
          if (!htime) continue;

          ++row.ntruth;

          //const auto proc = htime->GetPhotonCreatorProcess();

          // If a digi hit is formed by more than one real photon, the label is
          // set to 1 as soon as at least one contributing photon comes from the
          // prompt ancestry chain.
          if (has_prompt && row.hit_from_prompt == 0) {
            const int parent_track_id = htime->GetParentSavedTrackID();
            if (DescendsFromPromptTrack(parent_track_id,
                                        prompt_source_track_id,
                                        track_map)) {
              row.hit_from_prompt = 1;
            }
          }

          // If a digi hit is formed by more than one real photon, the capture
          // label is set to 1 as soon as at least one contributing photon has
          // a capture ancestor in its parent-track chain.
          if (row.hit_from_capture == 0) {
            const int parent_track_id = htime->GetParentSavedTrackID();
            if (HasCaptureAncestor(parent_track_id, track_map)) {
              row.hit_from_capture = 1;
            }
          }
        }

        // If the digit is neither prompt-tagged nor capture-tagged, assign it
        // to the residual "other process" category.
        row.hit_from_otherprocess =
            (row.hit_from_prompt == 0 && row.hit_from_capture == 0) ? 1 : 0;

        digi_rows.push_back(row);
      }

      // Sort by digit time, then by tube ID for stable ordering inside the
      // subevent. Event and subevent ordering remain unchanged in the output.
      std::sort(digi_rows.begin(), digi_rows.end(),
                [](const DigiHitRow& a, const DigiHitRow& b) {
                  if (a.t != b.t) return a.t < b.t;
                  return a.tubeid < b.tubeid;
                });

      for (const auto& row : digi_rows) {
        digi_tubeid.push_back(row.tubeid);
        digi_mPMTid.push_back(row.mPMTid);
        digi_mPMT_pmtid.push_back(row.mPMT_pmtid);

        digi_q.push_back(row.q);
        digi_t.push_back(row.t);

        digi_x.push_back(row.x);
        digi_y.push_back(row.y);
        digi_z.push_back(row.z);

        digi_dx.push_back(row.dx);
        digi_dy.push_back(row.dy);
        digi_dz.push_back(row.dz);

        digi_ntruth.push_back(row.ntruth);
        digi_hit_from_prompt.push_back(row.hit_from_prompt);
        digi_hit_from_capture.push_back(row.hit_from_capture);
        digi_hit_from_otherprocess.push_back(row.hit_from_otherprocess);
      }

      ndigi = static_cast<int>(digi_rows.size());

      out_true->Fill();
      out_digi->Fill();
    }

    superevent->ReInitialize();
  }

  fout->cd();
  out_true->Write();
  out_digi->Write();
  fout->Close();
  fin->Close();

  std::cout << "Wrote output file: " << outfile << std::endl;
  std::cout << "Output trees: TTrueHits, TDigiHits" << std::endl;
}