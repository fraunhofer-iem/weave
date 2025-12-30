#!/usr/bin/env python3
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import sys
from scipy.io import arff as scipy_arff
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score,
    recall_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def load_arff_df(path: str) -> pd.DataFrame:
    """Load ARFF file into a pandas DataFrame and decode byte strings."""
    data, meta = scipy_arff.loadarff(path)
    df = pd.DataFrame(data)
    # Decode bytes to strings for nominal attributes
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
    return df


def guess_target_column(df: pd.DataFrame, explicit: str | None) -> str:
    """Resolve target column name: explicit, 'class' (case-insensitive), or last column."""
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Target column '{explicit}' not found in ARFF.")
        return explicit
    # Common convention: 'class'
    class_candidates = [c for c in df.columns if c.lower() == "class"]
    if class_candidates:
        return class_candidates[0]
    # Fallback: last column
    return df.columns[-1]

def make_preprocess(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    """Create preprocessing transformer: scale numeric, one-hot encode categorical (dense)."""
    X_df = df.drop(columns=[target_col])
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Ensure dense output from OHE (compat across sklearn versions)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", ohe, cat_cols))

    preprocess = ColumnTransformer(transformers, remainder="drop")
    return preprocess


def main():
    parser = argparse.ArgumentParser(description="Cross-validate model on ARFF and output macro P/R/F1 as JSON.")
    parser.add_argument("--arff", required=True, help="Path to the ARFF file.")
    parser.add_argument("--model", required=True, help="Model name: logreg | rf | svm | nb | gb")
    parser.add_argument("--seed", required=True, type=int, help="Random seed.")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds (default: 5).")
    parser.add_argument("--target", default=None, help="Target column name (default: 'class' or last column).")
    args = parser.parse_args()

    # Silence common warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        df = load_arff_df(args.arff)
        target_col = df.columns[-1]
        y = df[target_col]
        x_df = df.drop(columns=[target_col])

        preprocess = make_preprocess(df, target_col)
        model = SVC(kernel="rbf", C=1.0, gamma="scale") #build_model(args.model, args.seed)
        clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

        # Stratified CV for classification
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

        scoring = {
            "avgMacroPrecision": make_scorer(precision_score, average="macro", zero_division=0),
            "precision": make_scorer(precision_score, average="micro", zero_division=0),
            "recall": make_scorer(recall_score, average="micro", zero_division=0),
            "f1": "f1_micro",
        }

        results = cross_validate(
            clf,
            x_df,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=None,
        )

        out = {
            "precision": float(np.mean(results["test_precision"])),
            "recall": float(np.mean(results["test_recall"])),
            "f1": float(np.mean(results["test_f1"])),
        }

        # Single-line JSON for easy Java parsing
        print(json.dumps(out))
    except Exception as e:
        # Print error to stderr and exit non-zero so Java can detect failure
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()