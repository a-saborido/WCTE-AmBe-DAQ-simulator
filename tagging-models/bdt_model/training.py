"""BDT training orchestration for WCTE AmBe neutron tagging."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .evaluation import evaluate_training


def load_candidates(path: Path) -> pd.DataFrame:
    """Load candidate rows produced by the extract command."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_candidate_windows(
    candidates: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the anchor windows from each source file into train/validation/test."""
    from sklearn.model_selection import train_test_split

    required = ["source_file", "source_entry"]
    missing = [column for column in required if column not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate table is missing split columns: {missing}")

    split_indices = {"train": [], "validation": [], "test": []}
    for file_number, (source_file, file_rows) in enumerate(
        candidates.groupby("source_file", sort=True)
    ):
        windows = np.sort(file_rows["source_entry"].unique())
        if len(windows) < 4:
            raise ValueError(
                f"{source_file} has only {len(windows)} candidate-producing windows; "
                "at least 4 are needed for train/validation/test splitting"
            )

        train_windows, remaining_windows = train_test_split(
            windows,
            test_size=0.50,
            random_state=seed + file_number,
        )
        validation_windows, test_windows = train_test_split(
            remaining_windows,
            test_size=0.50,
            random_state=seed + file_number,
        )

        file_mask = candidates["source_file"].eq(source_file)
        for split_name, selected_windows in [
            ("train", train_windows),
            ("validation", validation_windows),
            ("test", test_windows),
        ]:
            row_mask = file_mask & candidates["source_entry"].isin(selected_windows)
            split_indices[split_name].append(np.flatnonzero(row_mask.to_numpy()))

    return tuple(
        np.sort(np.concatenate(split_indices[name]))
        for name in ["train", "validation", "test"]
    )


def train_and_evaluate(args: argparse.Namespace) -> None:
    """Train the XGBoost BDT, save it, and evaluate its performance."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit("Missing xgboost. Install with: pip install xgboost") from exc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_candidates(Path(args.features))
    if not FEATURE_COLUMNS:
        raise ValueError("bdt_model/config.py must contain at least one feature")
    if len(FEATURE_COLUMNS) != len(set(FEATURE_COLUMNS)):
        raise ValueError("bdt_model/config.py contains duplicate feature names")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Input candidate table must contain a 'label' column")

    train_cols = FEATURE_COLUMNS + ["label"]
    finite_mask = df[train_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    use = df.loc[finite_mask].copy()
    y = use["label"].astype(int).to_numpy()
    X = use[FEATURE_COLUMNS].to_numpy(dtype=float)

    n_sig = int(np.sum(y == 1))
    n_bkg = int(np.sum(y == 0))
    if n_sig < 10 or n_bkg < 10:
        raise ValueError(
            "Not enough candidates after finite-feature selection: "
            f"signal={n_sig}, background={n_bkg}"
        )

    # Every file contributes windows to all three samples. Candidates from the
    # same anchor readout window always stay in the same split.
    idx_train, idx_val, idx_test = split_candidate_windows(use, args.seed)
    split_strategy = "per_file_anchor_window"

    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    for split_name, yy in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        if len(np.unique(yy)) < 2:
            raise ValueError(f"{split_name} split contains only one class; cannot train/evaluate robustly")

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method=args.tree_method,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=args.verbose_eval)

    # Prediction needs only the fitted model and its exact input-column order.
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(FEATURE_COLUMNS),
        },
        outdir / "ntag_xgb_model.joblib",
    )

    evaluate_training(
        args=args,
        outdir=outdir,
        model=model,
        use=use,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        split_strategy=split_strategy,
        n_signal=n_sig,
        n_background=n_bkg,
        feature_columns=FEATURE_COLUMNS,
    )


def add_train_args(
    p: argparse.ArgumentParser,
    require_features: bool = True,
    include_features: bool = True,
    include_outdir: bool = True,
) -> None:
    """Attach model-training CLI options."""
    if include_features:
        p.add_argument(
            "--features",
            required=require_features,
            help="candidates.parquet or candidates.csv from extract",
        )
    if include_outdir:
        p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-jobs", type=int, default=8)

    p.add_argument("--learning-rate", type=float, default=0.025219)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--subsample", type=float, default=0.97)
    p.add_argument("--colsample-bytree", type=float, default=1.0)
    p.add_argument("--tree-method", default="auto")
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--verbose-eval", type=int, default=50)
    p.add_argument("--bdt-cuts", type=float, nargs="+", default=[0.1, 0.5, 0.9, 0.99])
