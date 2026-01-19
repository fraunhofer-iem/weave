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

def predict_http(server_url: str, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """
    Calls the Java HTTP /predict endpoint to obtain prediction probabilities.
    server_url : Base URL of the prediction server
    X : Feature matrix of shape (n_samples, n_features)
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    all_probs = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X[start:end]
        payload = {"instances": batch.tolist()}

        resp = requests.post(
            f"{server_url}/predict",
            json=payload,
            timeout=(300, 900)
        )
        resp.raise_for_status()
        data = resp.json()
        batch_probs = np.array(data["probs"], dtype=float)
        all_probs.append(batch_probs)

    return np.vstack(all_probs)


def explain_weka(server_url: str, global_csv: str, local_csv: str, output_dir: str,
                 shap_global_bg_sample_size: int, shap_global_exp_sample_size: int,
                 shap_local_bg_sample_size: int, shap_local_exp_sample_size: int, positive_class_index: int = 1) -> None:
    """
    Computes SHAP explanations for a WEKA single-label (binary) classifier.
    Global explanations:
        - Beeswarm plot for the positive class (positive_class_index).
    Local explanations:
        - Waterfall plots for each instance in the local CSV, for the positive class.
    """
    df_global = pd.read_csv(global_csv)
    df_local_full = pd.read_csv(local_csv)

    feature_names = list(df_global.columns)
    Xg_full = df_global.values
    Xl_full = df_local_full.values

    def f_pos(X: np.ndarray) -> np.ndarray:
        probs = predict_http(server_url, X)
        return probs[:, positive_class_index]

    rng = np.random.default_rng(42)
    n_global = Xg_full.shape[0]
    n_local = Xl_full.shape[0]

    # sample indices from the correct data sources
    global_bg_idx = rng.choice(n_global, size=shap_global_bg_sample_size, replace=False)
    global_exp_idx = rng.choice(n_global, size=shap_global_exp_sample_size, replace=False)
    local_bg_idx = rng.choice(n_local, size=shap_local_bg_sample_size, replace=False)
    local_exp_idx = rng.choice(n_local, size=shap_local_exp_sample_size, replace=False)

    Xg_bg = Xg_full[global_bg_idx]
    Xg_exp = Xg_full[global_exp_idx]

    Xl_bg = Xl_full[local_bg_idx]
    Xl_exp = Xl_full[local_exp_idx]
    df_local = df_local_full.iloc[local_exp_idx].reset_index(drop=True)

    # Global explanations
    explainer_global = shap.KernelExplainer(f_pos, Xg_bg)
    shap_values_global = explainer_global(Xg_exp)

    # Export global shap values
    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    global_shap_array = np.asarray(shap_values_global.values
        if isinstance(shap_values_global, Explanation) else shap_values_global)
    df_global_shap = pd.DataFrame(global_shap_array, columns=feature_names)
    global_mean_abs = df_global_shap.abs().mean(axis=0)
    df_global_agg = pd.DataFrame({
        "Feature": feature_names,
        "GlobalShapValAgg": global_mean_abs.values
    })
    df_global_agg.to_csv(
        os.path.join(global_dir, "global_shap_aggregated.csv"),
        index=False
    )

    shap_values_global_top = aggregate_global_top_k(shap_values_global, feature_names=feature_names, k=10,)

    plt.figure()
    shap.summary_plot(shap_values_global_top, show=False, sort=False)
    plt.tight_layout()
    plt.savefig(os.path.join(global_dir, "global_beeswarm.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # Local explanations (background: local)
    explainer_local = shap.KernelExplainer(f_pos, Xl_bg)
    shap_values_local = explainer_local(Xl_exp)

    local_dir = os.path.join(output_dir, "local")
    os.makedirs(local_dir, exist_ok=True)

    # Export local shap values
    local_array = np.asarray(
        shap_values_local.values
        if isinstance(shap_values_local, Explanation) else shap_values_local
    )

    # Mean absolute SHAP per feature over all local instances
    local_mean_abs = np.mean(np.abs(local_array), axis=0)

    df_local_agg = pd.DataFrame({
        "Feature": feature_names,
        "LocalShapValAgg": local_mean_abs
    })

    df_local_agg.to_csv(
        os.path.join(local_dir, "local_shap_aggregated.csv"),
        index=False
    )

    for i in range(Xl_exp.shape[0]):
        print(f"[WEKA] Computing SHAP values for instance {i} ...")
        expl_single = aggregate_local_top_k(shap_values_local, df_local, feature_names, row_index=i, k=10,)

        plt.figure()
        shap.plots.waterfall(expl_single, max_display=11, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(local_dir, f"local_instance_{i}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"[WEKA] Global and local SHAP plots written to: {output_dir}")


def explain_meka(server_url: str, global_csv: str, local_csv: str, output_dir: str,
                 shap_global_bg_sample_size: int, shap_global_exp_sample_size: int,
                 shap_local_bg_sample_size: int, shap_local_exp_sample_size: int) -> None:
    """
    Computes SHAP explanations for a MEKA multi-label classifier.
    For each label j:
        Global beeswarm plot showing feature importance for label j.
        Local waterfall plots for each instance in the local CSV for label j.
    """
    df_global = pd.read_csv(global_csv)
    df_local_full = pd.read_csv(local_csv)

    feature_names = list(df_global.columns)
    Xg_full = df_global.values
    Xl_full = df_local_full.values

    sample_probs = predict_http(server_url, Xg_full[:1])
    n_labels = sample_probs.shape[1]

    rng = np.random.default_rng(42)
    n_global = Xg_full.shape[0]
    n_local = Xl_full.shape[0]

    # sample indices from the correct data sources
    global_bg_idx = rng.choice(n_global, size=shap_global_bg_sample_size, replace=False)
    global_exp_idx = rng.choice(n_global, size=shap_global_exp_sample_size, replace=False)
    local_bg_idx = rng.choice(n_local, size=shap_local_bg_sample_size, replace=False)
    local_exp_idx = rng.choice(n_local, size=shap_local_exp_sample_size, replace=False)

    Xg_bg = Xg_full[global_bg_idx]
    Xg_exp = Xg_full[global_exp_idx]

    Xl_bg = Xl_full[local_bg_idx]
    Xl_exp = Xl_full[local_exp_idx]
    df_local = df_local_full.iloc[local_exp_idx].reset_index(drop=True)

    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)

    base_local_dir = os.path.join(output_dir, "local")
    os.makedirs(base_local_dir, exist_ok=True)

    global_importance_acc = np.zeros(len(feature_names), dtype=float)
    n_labels_acc_global = 0

    local_importance_acc = np.zeros(len(feature_names), dtype=float)
    n_labels_acc = 0

    # For each label, build separate explainers for global and local
    for label_index in range(n_labels):
        print(f"[MEKA] Computing SHAP values for label {label_index} ...")

        def f_label(X: np.ndarray, j: int = label_index) -> np.ndarray:
            probs = predict_http(server_url, X)
            return probs[:, j]

        # Global explanations (background: global)
        explainer_global = shap.KernelExplainer(f_label, Xg_bg)
        shap_values_global = explainer_global(Xg_exp)

        # Export global shap values
        global_array = np.asarray(
            shap_values_global.values
            if isinstance(shap_values_global, Explanation) else shap_values_global
        )
        global_mean_abs_for_label = np.mean(np.abs(global_array), axis=0)
        global_importance_acc += global_mean_abs_for_label
        n_labels_acc_global += 1

        shap_values_global_top = aggregate_global_top_k(shap_values_global, feature_names=feature_names, k=10,)

        plt.figure()
        shap.summary_plot(shap_values_global_top, show=False, sort=False)
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, f"global_beeswarm_label_{label_index}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

        # Local explanations (background: local)
        explainer_local = shap.KernelExplainer(f_label, Xl_bg)
        shap_values_local = explainer_local(Xl_exp)

        local_dir = os.path.join(base_local_dir, f"{label_index}")
        os.makedirs(local_dir, exist_ok=True)

        # Export local shap values
        local_array = np.asarray(
            shap_values_local.values
            if isinstance(shap_values_local, Explanation) else shap_values_local
        )  # shape: (n_local_instances, n_features)

        # mean |SHAP| per feature for this label
        local_mean_abs_for_label = np.mean(np.abs(local_array), axis=0)
        local_importance_acc += local_mean_abs_for_label
        n_labels_acc += 1

        for i in range(Xl_exp.shape[0]):
            expl_single = aggregate_local_top_k(shap_values_local, df_local, feature_names, row_index=i, k=10,)

            plt.figure()
            shap.plots.waterfall(expl_single, max_display=11, show=False)
            plt.tight_layout()
            plt.savefig( os.path.join(local_dir, f"local_instance_{i}_label_{label_index}.pdf"), dpi=300, bbox_inches='tight')
            plt.close()

    if n_labels_acc_global > 0:
        global_mean_abs_all_labels = global_importance_acc / n_labels_acc_global

        df_global_agg_all = pd.DataFrame({
            "Feature": feature_names,
            "GlobalSHAPValAgg_AllLabels": global_mean_abs_all_labels
        })

        df_global_agg_all.to_csv(
            os.path.join(global_dir, "global_shap_aggregated_all_labels.csv"),
            index=False
        )

    if n_labels_acc > 0:
        local_mean_abs_all_labels = local_importance_acc / n_labels_acc
        df_local_agg_all = pd.DataFrame({
        "Feature": feature_names,
        "LocalSHAPValAgg_AllLabels": local_mean_abs_all_labels
        })
        df_local_agg_all.to_csv(
        os.path.join(base_local_dir, "local_shap_aggregated_all_labels.csv"),
        index=False
        )


    print(f"[MEKA] Global and local label-specific SHAP plots written to: {base_local_dir}")


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

    parser.add_argument(
        "--shap_global_bg_samples",
        default="1000",
        help="No. of Samples for Shap Explainer"
    )

    parser.add_argument(
        "--shap_global_exp_samples",
        default="1000",
        help="No. of Samples to explain for Global Explainability"
    )

    parser.add_argument(
        "--shap_local_bg_samples",
        default="1000",
        help="No. of Samples for Shap Explainer"
    )

    parser.add_argument(
        "--shap_local_exp_samples",
        default="1000",
        help="No. of Samples to explain for Global Explainability"
    )

    args = parser.parse_args()

    if args.toolkit == "weka":
        explain_weka(
            server_url=args.server_url, global_csv=args.global_csv, local_csv=args.local_csv, output_dir=args.output_dir,
            shap_global_bg_sample_size=int(args.shap_global_bg_samples),
            shap_global_exp_sample_size=int(args.shap_global_exp_samples),
            shap_local_bg_sample_size=int(args.shap_local_bg_samples),
            shap_local_exp_sample_size=int(args.shap_local_exp_samples)
        )
    else:
        explain_meka(
            server_url=args.server_url, global_csv=args.global_csv, local_csv=args.local_csv, output_dir=args.output_dir,
            shap_global_bg_sample_size=int(args.shap_global_bg_samples),
            shap_global_exp_sample_size=int(args.shap_global_exp_samples),
            shap_local_bg_sample_size=int(args.shap_local_bg_samples),
            shap_local_exp_sample_size=int(args.shap_local_exp_samples)
        )

if __name__ == "__main__":
    main()