#!/usr/bin/env python
"""
SHAP HTTP explainer for WEKA and MEKA models.

This script:
  1. Reads global and local feature matrices from CSV files.
  2. Calls a Java HTTP /predict endpoint to obtain prediction probabilities.
  3. Uses SHAP (KernelExplainer) to compute:
     - WEKA (binary): global beeswarm + local waterfall plots for the positive class.
     - MEKA (multi-label): global beeswarm + local waterfall plots for each label.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
from shap import Explanation

plt.switch_backend("Agg")

def aggregate_global_top_k(expl: Explanation,
                           feature_names: list[str],
                           k: int = 10) -> Explanation:
    """
    Build a Explanation with top-k individual features (by mean(|SHAP|) over samples)
    """
    vals = np.asarray(expl.values)
    base_values = expl.base_values
    data = expl.data

    n_samples, n_features = vals.shape
    k = min(k, n_features)

    # Rank features by global importance
    mean_abs = np.mean(np.abs(vals), axis=0)
    idx_sorted = np.argsort(-mean_abs)
    top_idx = idx_sorted[:k]
    rest_idx = idx_sorted[k:]

    n_top = len(top_idx)
    new_vals = np.zeros((n_samples, n_top + 1), dtype=float)
    new_vals[:, :n_top] = vals[:, top_idx]
    if rest_idx.size > 0:
        new_vals[:, -1] = vals[:, rest_idx].sum(axis=1)

    # New feature names
    remaining_count = max(0, n_features - n_top)
    sum_name = f"Σ{remaining_count}RemFeat" if remaining_count > 0 else "Σ0RemFeat"
    new_feature_names = [feature_names[i] for i in top_idx] + [sum_name]

    if data is None:
        new_data = None
    else:
        if isinstance(data, pd.DataFrame):
            new_data = pd.DataFrame(
                np.zeros((n_samples, n_top + 1), dtype=float),
                columns=new_feature_names,
                index=getattr(data, "index", None),
            )
            for j, fi in enumerate(top_idx):
                colname = feature_names[fi]
                new_data.iloc[:, j] = data[colname].values
        else:
            data_np = np.asarray(data)
            new_data = np.zeros_like(new_vals)
            new_data[:, :n_top] = data_np[:, top_idx]

    return Explanation(
        values=new_vals,
        base_values=base_values,
        data=new_data,
        feature_names=new_feature_names,
    )


def aggregate_local_top_k(expl: Explanation,
                          df_local: pd.DataFrame,
                          feature_names: list[str],
                          row_index: int,
                          k: int = 10) -> Explanation:
    """
    Build a 1D Explanation for a single instance with top-k features.
    """
    vals = np.asarray(expl.values[row_index])  # (n_features,)
    base = expl.base_values[row_index]
    n_features = vals.shape[0]
    k = min(k, n_features)

    abs_vals = np.abs(vals)
    idx_sorted = np.argsort(-abs_vals)
    top_idx = idx_sorted[:k]
    other_idx = idx_sorted[k:]

    new_vals = np.zeros(k + 1, dtype=float)
    new_vals[:k] = vals[top_idx]
    if len(other_idx) > 0:
        new_vals[-1] = vals[other_idx].sum()

    new_names = [feature_names[i] for i in top_idx] + ["OTHER_FEATURES"]
    row = df_local.iloc[row_index]
    new_data = np.zeros(k + 1, dtype=float)
    for j, fi in enumerate(top_idx):
        fname = feature_names[fi]
        new_data[j] = row[fname]

    return Explanation(
        values=new_vals,
        base_values=base,
        data=new_data,
        feature_names=new_names,
    )

def predict_http(server_url: str, X: np.ndarray) -> np.ndarray:
    """
    Calls the Java HTTP /predict endpoint to obtain prediction probabilities.
    server_url : Base URL of the prediction server
    X : Feature matrix of shape (n_samples, n_features)
    """
    payload = {"instances": X.tolist()}
    resp = requests.post(f"{server_url}/predict", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["probs"], dtype=float)


def explain_weka(server_url: str,
                 global_csv: str,
                 local_csv: str,
                 output_dir: str,
                 positive_class_index: int = 1) -> None:
    """
    Computes SHAP explanations for a WEKA single-label (binary) classifier.

    Global explanations:
        - Beeswarm plot for the positive class (positive_class_index).

    Local explanations:
        - Waterfall plots for each instance in the local CSV, for the positive class.

    """
    df_global = pd.read_csv(global_csv)
    df_local = pd.read_csv(local_csv)

    feature_names = list(df_global.columns)
    Xg = df_global.values
    Xl = df_local.values

    def f_pos(X: np.ndarray) -> np.ndarray:
        probs = predict_http(server_url, X)
        return probs[:, positive_class_index]

    # Todo: Remove sampling
    background = shap.sample(Xg, 100, random_state=0)

    explainer = shap.KernelExplainer(f_pos, background)

    # Todo: Remove sampling
    Xg_explain = shap.sample(Xg, 100, random_state=1)

    # Global explanations
    shap_values_global = explainer(Xg_explain)

    shap_values_global_top = aggregate_global_top_k(
        shap_values_global,
        feature_names=feature_names,
        k=10,
    )

    weka_dir = os.path.join(output_dir, "weka")
    os.makedirs(weka_dir, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values_global_top,
        show=False,
        sort=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(weka_dir, "global_beeswarm_positive.png"))
    plt.close()

    # Local explanations
    shap_values_local = explainer(Xl)

    for i in range(df_local.shape[0]):
        print(f"[MEKA] Computing SHAP values for instance {i} ...")
        expl_single = aggregate_local_top_k(
            shap_values_local,
            df_local,
            feature_names,
            row_index=i,
            k=10,
        )

        plt.figure()
        shap.plots.waterfall(
            expl_single,
            max_display=11,   # 10 + OTHER
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(weka_dir, f"local/local_instance_{i}_positive.png"))
        plt.close()


    print(f"[WEKA] Global and local SHAP plots written to: {weka_dir}")


def explain_meka(server_url: str,
                 global_csv: str,
                 local_csv: str,
                 output_dir: str) -> None:
    """
    Computes SHAP explanations for a MEKA multi-label classifier.
    For each label j:
        Global beeswarm plot showing feature importance for label j.
        Local waterfall plots for each instance in the local CSV for label j.
    """
    df_global = pd.read_csv(global_csv)
    df_local = pd.read_csv(local_csv)

    feature_names = list(df_global.columns)
    Xg = df_global.values
    Xl = df_local.values

    sample_probs = predict_http(server_url, Xg[:1])
    n_labels = sample_probs.shape[1]

    meka_dir = os.path.join(output_dir, "meka")
    os.makedirs(meka_dir, exist_ok=True)

    # For each label, build a separate scalar-output model f_j(X) and run SHAP on it.
    for label_index in range(n_labels):
        print(f"[MEKA] Computing SHAP values for label {label_index} ...")

        def f_label(X: np.ndarray, j: int = label_index) -> np.ndarray:
            probs = predict_http(server_url, X)
            return probs[:, j]

        # Todo: Remove sampling
        background = shap.sample(Xg, 50, random_state=0)

        explainer = shap.KernelExplainer(f_label, background) #Xg

        # Todo: Remove sampling
        Xg_explain = shap.sample(Xg, 50, random_state=1)

        # Global explanations
        shap_values_global = explainer(Xg_explain) #Xg

        shap_values_global_top = aggregate_global_top_k(
            shap_values_global,
            feature_names=feature_names,
            k=10,
        )

        plt.figure()
        shap.summary_plot(
            shap_values_global_top,
            show=False,
            sort=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(meka_dir, f"global_beeswarm_label_{label_index}.png"))
        plt.close()

        # Local explanations
        shap_values_local = explainer(Xl)

        for i in range(df_local.shape[0]):
            expl_single = aggregate_local_top_k(
                shap_values_local,
                df_local,
                feature_names,
                row_index=i,
                k=10,
            )

            plt.figure()
            shap.plots.waterfall(
                expl_single,
                max_display=11,
                show=False
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    meka_dir,
                    f"local/local_instance_{i}_label_{label_index}.png"
                )
            )
            plt.close()

    print(f"[MEKA] Global and local label-specific SHAP plots written to: {meka_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--toolkit",
        choices=["weka", "meka"],
        required=True,
        help="ML toolkit used: 'weka' or 'meka'."
    )
    parser.add_argument(
        "--server_url",
        default="http://localhost:8090",
        help="Base URL of the Java server (default: http://localhost:8090)."
    )
    parser.add_argument(
        "--global_csv",
        default="global_features.csv",
        help="Path to global features CSV."
    )
    parser.add_argument(
        "--local_csv",
        default="local_features.csv",
        help="Path to local features CSV."
    )
    parser.add_argument(
        "--output_dir",
        default="shap_outputs",
        help="Output directory for the plots."
    )

    args = parser.parse_args()

    if args.toolkit == "weka":
        explain_weka(
            server_url=args.server_url,
            global_csv=args.global_csv,
            local_csv=args.local_csv,
            output_dir=args.output_dir
        )
    else:
        explain_meka(
            server_url=args.server_url,
            global_csv=args.global_csv,
            local_csv=args.local_csv,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()