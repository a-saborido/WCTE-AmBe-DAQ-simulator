# WCTE AmBe nTagging

End-to-end tools for producing WCTE AmBe calibration simulations and training
neutron-tagging models from the resulting DAQ-like samples.

## Repository layout

```text
WCTE-AmBe-nTagging/
├── DAQ-simulator/
│   ├── simulation/             WCSim production macros and Slurm submission
│   ├── extraction/             WCSim hit and truth extraction
│   └── daq_windows/            Synthetic DAQ-window generation and inspection
└── tagging-models/
    ├── candidates_extraction/  Prompt and delayed-candidate reconstruction
    ├── bdt_model/              BDT training, evaluation, and prediction
    ├── data/                   Geometry and local input samples
    └── tests/                  Regression tests
```

## Workflow

1. Generate AmBe events with WCSim using `DAQ-simulator/simulation/`.
2. Extract hit-level data and build synthetic DAQ windows with the remaining
   tools under `DAQ-simulator/`.
3. Reconstruct neutron candidates with `tagging-models/candidates_extraction/`.
4. Train, evaluate, or apply the BDT with `tagging-models/bdt_model/`.

Detailed setup and command examples are available in
[`DAQ-simulator/README.md`](DAQ-simulator/README.md) and
[`tagging-models/README.md`](tagging-models/README.md). Run commands from the
corresponding component directory so documented relative paths resolve
correctly.

## Tests

The tagging pipeline includes fast regression tests based on Python's standard
`unittest` framework. Run them from the repository root with:

```bash
cd tagging-models
python -m unittest discover -s tests -v
```

The tests use small synthetic inputs and temporary output directories; they do
not process the large ROOT samples or modify repository data. A successful run
ends with `OK` and returns exit code zero.

Large ROOT files, generated outputs, logs, and caches are excluded through the
component-level `.gitignore` files.
