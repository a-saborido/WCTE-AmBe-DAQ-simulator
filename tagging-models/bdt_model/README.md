# BDT Model

This folder contains the model stage of the neutron-tagging workflow. It reads
candidate tables written by `candidates_extraction/` and does not inspect ROOT readout or
truth trees directly.

## Files

- `training.py`: removes candidates with missing or infinite BDT feature values,
  splits the anchor windows from every source file into train/validation/test,
  and fits and saves the XGBoost model.
- `config.py`: selects which extracted observables are used as BDT features.
- `evaluation.py`: training and labeled-prediction metrics, cut tables, and
  diagnostic plots.
- `prediction.py`: isolated application of a saved model to a new candidate
  table.

Edit `FEATURE_COLUMNS` in `bdt_model/config.py` to train with any subset of the
columns produced by extraction. This list is independent of
`candidates_extraction/config.py`, so changing the BDT feature selection does not change
candidate extraction.

Training stores the selected feature names and their order in the saved model
bundle. Prediction reads that stored list, ensuring that it applies exactly the
features used when the model was trained.

## Workflow

```text
Training
  -> fit and save the model with its feature order
  -> evaluate train/validation/test performance
  -> save metrics, a BDT cut table, and diagnostic plots

Prediction
  -> load the model and feature order
  -> read and score new candidates
  -> save the scored candidate table

Prediction evaluation, only with truth labels
  -> evaluate the new scores
  -> save metrics, a BDT cut table, and diagnostic plots
```

The joblib model bundle contains only `model` and `feature_columns`. Training
evaluation results are stored separately, and prediction does not read the
training table or `bdt_model/config.py`.

## Train And Evaluate

Run training through the repository CLI:

```bash
python wcte_ambe_neutron_bdt.py train \
  --features outputs/ntag_bdt_out/candidates.parquet \
  --outdir outputs/ntag_bdt_out/model
```

Training assigns 50% of each source file's candidate-producing anchor windows
to training, 25% to validation, and 25% to testing. Candidates found from the
same anchor window always remain in the same split.

## Score Candidates

```bash
python -m bdt_model.prediction \
  --model outputs/ntag_bdt_out/model/ntag_xgb_model.joblib \
  --candidates outputs/another_sample/candidates.parquet \
  --output outputs/another_sample/candidates_scored.parquet
```

This always writes the scored candidate table. It does not use training data.

If the new table has binary truth labels, request a separate evaluation:

```bash
python -m bdt_model.prediction \
  --model outputs/ntag_bdt_out/model/ntag_xgb_model.joblib \
  --candidates outputs/labeled_sample/candidates.parquet \
  --output outputs/labeled_sample/candidates_scored.parquet \
  --evaluation-dir outputs/labeled_sample/evaluation
```

Without truth labels, AUC, efficiency, background acceptance, and purity cannot
be calculated; the scored candidate table is still produced normally.
