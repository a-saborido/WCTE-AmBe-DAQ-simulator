# WCTE AmBe DAQ simulator

This repository contains the workflow used to generate, extract, and process realistic WCSim simulations of WCTE AmBe calibration data, taking into account the DAQ procedure.

The workflow has four main stages:

1. run many individual AmBe decay simulations in parallel using the WCSim AmBe generator,
2. merge the resulting WCSim ROOT files,
3. extract hit-level data and prompt/capture truth into a lighter ROOT format,
4. build synthetic DAQ windows by overlaying individual AmBe events.


## Repository structure

```text
WCTE-AmBe-DAQ-simulator/
├── README.md
├── simulation/
│   ├── submit.sh
│   └── macros/
│       ├── wcte_ambe_template.mac
│       ├── tuning_parameters.mac
│       └── daq.mac
├── extraction/
│   ├── extract_AmBe_hits.C
│   └── inspect_AmBe_event.py
└── daq_windows/
    ├── ambe_windows_truehits.ipynb
    └── ambe_windows_digihits.ipynb
```


## 1. WCSim setup

The simulations were produced with **WCSim v1.12.29**:

- WCSim repository: <https://github.com/WCSim/WCSim>

In the file `macros/jobOptions.mac` of the WCSim installation, the following physics settings must be enabled:

```text
/WCSim/physics/list FTFP_BERT_HP
/WCSim/physics/nCapture GLG4Sim
```

These settings are required for the AmBe production used here, in particular for neutron transport and capture modeling.

The WCSim macro files specific to this AmBe setup are provided in this repository under `simulation/macros/`. These include the main simulation template, DAQ configuration, and detector tuning files used for the production.


## 2. Parallel AmBe production

The simulation is launched through `simulation/submit.sh` as a parallel Slurm job array. Each task creates a job-specific macro from `simulation/macros/wcte_ambe_template.mac`, assigns a different random seed, runs WCSim, and writes one output ROOT file while storing stdout/stderr logs separately.

In practice, a production run generates many files automatically:

- per-job macros in `simulation/macs/`
- per-job logs in `simulation/logs/`
- per-job WCSim outputs in the simulation output directory

For large runs this typically means hundreds of generated macro and log files. In the example `submit.sh` provided here, the array is configured with 100 tasks (`#SBATCH --array=0-99`), so it produces 100 job-specific macro files and 100 output ROOT files.

The generated ROOT files follow the pattern:

```text
wcte_ambe_000.root
wcte_ambe_001.root
...
```


## 3. Merge the simulation outputs

Once all simulation jobs are finished, the individual WCSim files can be merged with `hadd`:

```bash
hadd -f wcte_ambe_merged.root wcte_ambe_*.root
```

This produces a single merged WCSim file that is easier to process downstream.


## 4. Extract hit-level data and prompt/capture truth

The merged WCSim file is then converted into a more analysis-friendly ROOT file with:

```bash
cd extraction
root -l extract_AmBe_hits.C
```

This extraction step does more than just flatten the hits. It also stores AmBe-specific truth information, including:

- hit-level information,
- neutron capture MC truth,
- prompt-track identification,
- per-hit labels indicating whether each contribution comes from the prompt or the capture.

The extractor writes two trees:

- `TTrueHits`: ordered true-hit information
- `TDigiHits`: ordered digitized-hit information

The original event identifiers are preserved and used in order throughout the full workflow. This makes it possible to trace each extracted event back to its corresponding entry in the original WCSim ROOT file. The script `extraction/inspect_AmBe_event.py` is provided for that event-level inspection and cross-checking.


## 5. Build synthetic DAQ windows

After extraction, the individual simulated AmBe events are overlaid into synthetic acquisition windows.

This is implemented in:

- `daq_windows/ambe_windows_truehits.ipynb`
- `daq_windows/ambe_windows_digihits.ipynb`

The idea is to start from single-AmBe events and construct a continuous source stream with a chosen source activity. A dead time between consecutive DAQ windows can also be introduced. For each acquisition window:

1. the number of decays expected in the window is drawn from a Poisson law,
2. events are sampled from the extracted library,
3. each sampled event is shifted in time,
4. delayed activity from earlier decays is allowed to spill into later windows,
5. only hits and captures falling inside the recorded window are kept,
6. the surviving content is sorted and written as one window entry.

To keep the stream realistic, an initial warm-up period is discarded, so the first recorded windows are not affected by the startup of the overlay procedure.

This preserves the prompt-delayed structure of the AmBe events while producing DAQ-like time windows that can be used for later reconstruction or ML studies.

Two versions are provided:

- **true-hit windows** from `TTrueHits`, in `daq_windows/ambe_windows_truehits.ipynb`
- **digitized-hit windows** from `TDigiHits`, in `daq_windows/ambe_windows_digihits.ipynb`

For the digitized-hit case, an additional PMT time-resolution effect can be applied so that nearby hits in the same tube are merged according to the detector time-resolution model.

The final windowed output can also be written back to a ROOT file with a format matching real WCTE data, together with the associated Monte Carlo truth information.


## 6. Summary of the workflow

The generator workflow is:

1. Simulate many AmBe events in parallel with WCSim.
2. Merge the output ROOT files with `hadd`.
3. Extract a compact ROOT representation with hit-level and prompt/capture truth information using `extraction/extract_AmBe_hits.C`.
4. Generate synthetic DAQ windows from the extracted events using the notebooks in `daq_windows/`.

---

## Files in this repository

- `simulation/submit.sh` — Slurm array submission script
- `simulation/macros/wcte_ambe_template.mac` — WCSim template macro used to generate per-job macros
- `simulation/macros/tuning_parameters.mac` — detector / optical tuning parameters
- `simulation/macros/daq.mac` — digitizer and DAQ settings of WCSim
- `extraction/extract_AmBe_hits.C` — ROOT macro to extract hit-level and AmBe-specific truth information
- `extraction/inspect_AmBe_event.py` — utility to inspect WCSim event content
- `daq_windows/ambe_windows_truehits.ipynb` — synthetic window generation starting from true hits
- `daq_windows/ambe_windows_digihits.ipynb` — synthetic window generation starting from digitized hits
