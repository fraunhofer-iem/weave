#!/usr/bin/env python
"""
SHAP HTTP explainer for WEKA and MEKA models.

This script:
  1. Reads global and local feature matrices from CSV files.
  2. Calls a Java HTTP /predict endpoint to obtain prediction probabilities.
  3. Uses SHAP (KernelExplainer) to compute:
     - WEKA: global beeswarm + local waterfall plots. A binary model is explained
       for the positive class; a model with more than two classes is explained for
       every class.
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

def sample_indices(rng, population: int, requested: int, what: str) -> np.ndarray:
    """
    Draw `requested` distinct row indices out of `population`, clamped to the
    population size.

    The local background and explanation sets are drawn from the test ARFF, which
    can hold fewer rows than the configured sample count - the gxa sanitizer set has
    248 rows against a configured 250 - and rng.choice(..., replace=False) raises
    "Cannot take a larger sample than population" instead of clamping.
    """
    if requested > population:
        print(f"[shap] {what}: {requested} samples requested but only {population} "
              f"rows available; using {population}.")
        requested = population
    return rng.choice(population, size=requested, replace=False)


def export_shap_matrix(expl, feature_names: list[str], source_idx, path: str) -> None:
    """
    Write the full SIGNED SHAP matrix: one row per explained instance, one column
    per feature. This is the raw output; the *_aggregated.csv files next to it are
    mean(|SHAP|) collapses of exactly this data and cannot be un-collapsed.

    Columns:
      row          position within the explained set - matches
                   local_instance_<row>.pdf, or local_instance_<row>_class_<k>.pdf
                   for a multi-class WEKA model
      dataset_row  row in the source features CSV. That CSV is written in ARFF order,
                   which is order-locked to the *-methods file, so this is what maps a
                   SHAP row back to a method signature.
      base_value   E[f(x)] over the background set
      prediction   base_value + sum(shap values), i.e. the model output being explained
      <feature>    signed SHAP value, one column per feature
    """
    values = np.asarray(expl.values if isinstance(expl, Explanation) else expl)
    base = (np.asarray(expl.base_values, dtype=float).reshape(-1)
            if isinstance(expl, Explanation) else np.array([np.nan]))
    if base.size == 1:
        base = np.repeat(base, values.shape[0])

    df = pd.DataFrame(values, columns=feature_names)
    df.insert(0, "prediction", base + values.sum(axis=1))
    df.insert(0, "base_value", base)
    df.insert(0, "dataset_row", np.asarray(source_idx))
    df.insert(0, "row", np.arange(values.shape[0]))
    df.to_csv(path, index=False)
    print(f"[shap] wrote {path} "
          f"({values.shape[0]} instances x {len(feature_names)} features, signed)")


def predict_http(server_url: str, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """
    Calls the Java HTTP /predict endpoint to obtain prediction probabilities.
    server_url : Base URL of the prediction server
    X : Feature matrix of shape (n_samples, n_features)

    Batches are posted one at a time, so batch_size sets how many rows the Java
    side can score in parallel. Keep it large: at 128 rows the server had far
    fewer rows in flight than it has cores, and the run was dominated by HTTP
    round trips.
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


def resolve_class_names(class_names: list[str] | None, n_classes: int) -> list[str]:
    """
    Class labels used in plot titles and in class_index.csv.

    WEKA orders the columns of distributionForInstance by the ARFF class
    attribute's declared value order, so element k of class_names names class k.
    Falls back to bare indices when no names are supplied, which keeps this
    script runnable stand-alone against a bare /predict endpoint.
    """
    if not class_names:
        return [str(k) for k in range(n_classes)]
    if len(class_names) != n_classes:
        print(f"[shap] warning: {len(class_names)} class name(s) supplied for "
              f"{n_classes} classes; using indices instead.")
        return [str(k) for k in range(n_classes)]
    return list(class_names)


def build_class_explanations(f, X_bg: np.ndarray, X_exp: np.ndarray,
                             feature_names: list[str], n_outputs: int) -> list[Explanation]:
    """
    Run ONE KernelExplainer pass and split it into one Explanation per output.

    KernelExplainer perturbs each explained instance and calls the model on those
    perturbations regardless of how many outputs f returns; only the weighted
    regression that turns those calls into attributions is repeated per output.
    Measured on shap 0.46.0, a 3-output function costs exactly the same number of
    model evaluations as a 1-output one. That is what makes explaining every class
    affordable here: the model calls are HTTP round trips to the JVM and dominate
    the runtime, so k classes cost about the same as the single class this
    replaced - not k times as much.

    n_outputs == 1 means f returns a 1-D vector and one Explanation comes back.
    """
    explainer = shap.KernelExplainer(f, X_bg)
    raw = explainer.shap_values(X_exp)
    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))

    # shap has returned both a list of (n, features) arrays and a single
    # (n, features, outputs) array across versions, so accept either.
    if isinstance(raw, list):
        per_output = [np.asarray(a, dtype=float) for a in raw]
    else:
        arr = np.asarray(raw, dtype=float)
        per_output = [arr] if arr.ndim == 2 else [arr[:, :, j] for j in range(arr.shape[2])]

    if len(per_output) != n_outputs or base.size != n_outputs:
        raise ValueError(
            f"expected {n_outputs} SHAP output(s) and base value(s), got "
            f"{len(per_output)} and {base.size} - shap "
            f"{getattr(shap, '__version__', '?')} returned an unexpected shape")

    return [
        Explanation(
            values=per_output[j],
            base_values=np.repeat(base[j], per_output[j].shape[0]),
            data=X_exp,
            feature_names=feature_names,
        )
        for j in range(n_outputs)
    ]


def explain_weka(server_url: str, global_csv: str, local_csv: str, output_dir: str,
                 shap_global_bg_sample_size: int, shap_global_exp_sample_size: int,
                 shap_local_bg_sample_size: int, shap_local_exp_sample_size: int,
                 positive_class_index: int = 1,
                 class_names: list[str] | None = None) -> None:
    """
    Computes SHAP explanations for a WEKA classifier.

    Binary models are explained for the positive class (positive_class_index) and
    the output keeps unsuffixed file names. Explaining only one class is complete
    there: the two probabilities sum to 1, so the other class's SHAP values are
    the negation of these and carry no extra information.

    Models with MORE than two classes are explained for every class. Those
    outputs carry a _class_<k> suffix and the local plots are split into
    local/<k>/ subdirectories - the same shape explain_meka uses per label -
    plus a pooled beeswarm and pooled aggregates over all classes. This is what
    the 3-class sscm case study (tag = None/Target/Input) needs; explaining
    class 1 alone left None and Input unexplained.
    """
    df_global = pd.read_csv(global_csv)
    df_local_full = pd.read_csv(local_csv)

    feature_names = list(df_global.columns)
    Xg_full = df_global.values
    Xl_full = df_local_full.values

    # Probe the model for its class count: one column per class value, in the
    # order the ARFF class attribute declares them.
    n_classes = predict_http(server_url, Xg_full[:1]).shape[1]
    if n_classes < 2:
        raise ValueError(f"/predict returned {n_classes} class column(s); a "
                         "classifier needs at least two")
    multiclass = n_classes > 2
    class_indices = list(range(n_classes)) if multiclass else [positive_class_index]
    names = resolve_class_names(class_names, n_classes)

    if multiclass:
        print(f"[WEKA] {n_classes} classes ({', '.join(names)}); explaining every class.")
    else:
        print(f"[WEKA] binary; explaining class {positive_class_index} "
              f"({names[positive_class_index]}).")

    def f_classes(X: np.ndarray) -> np.ndarray:
        probs = predict_http(server_url, X)
        # 1-D for the binary case keeps KernelExplainer single-output, which is
        # byte-for-byte what the earlier binary-only version did.
        return probs[:, class_indices] if multiclass else probs[:, positive_class_index]

    def suffix(class_index: int) -> str:
        return f"_class_{class_index}" if multiclass else ""

    rng = np.random.default_rng(42)
    n_global = Xg_full.shape[0]
    n_local = Xl_full.shape[0]

    # sample indices from the correct data sources
    global_bg_idx = sample_indices(rng, n_global, shap_global_bg_sample_size, "global background")
    global_exp_idx = sample_indices(rng, n_global, shap_global_exp_sample_size, "global explained")
    local_bg_idx = sample_indices(rng, n_local, shap_local_bg_sample_size, "local background")
    local_exp_idx = sample_indices(rng, n_local, shap_local_exp_sample_size, "local explained")

    Xg_bg = Xg_full[global_bg_idx]
    Xg_exp = Xg_full[global_exp_idx]

    Xl_bg = Xl_full[local_bg_idx]
    Xl_exp = Xl_full[local_exp_idx]
    df_local = df_local_full.iloc[local_exp_idx].reset_index(drop=True)

    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    base_local_dir = os.path.join(output_dir, "local")
    os.makedirs(base_local_dir, exist_ok=True)

    if multiclass:
        # Records what each _class_<k> suffix refers to, so the output can be read
        # without going back to the ARFF header.
        index_path = os.path.join(global_dir, "class_index.csv")
        pd.DataFrame({"class_index": class_indices,
                      "class_name": [names[k] for k in class_indices]}
                     ).to_csv(index_path, index=False)
        print(f"[WEKA] wrote {index_path}")

    # Global explanations (background: global)
    global_expls = build_class_explanations(
        f_classes, Xg_bg, Xg_exp, feature_names, len(class_indices))

    pooled_global = []
    global_importance_acc = np.zeros(len(feature_names), dtype=float)

    for class_index, expl in zip(class_indices, global_expls):
        values = np.asarray(expl.values, dtype=float)
        mean_abs = np.abs(values).mean(axis=0)

        pd.DataFrame({
            "Feature": feature_names,
            "GlobalShapValAgg": mean_abs
        }).to_csv(
            os.path.join(global_dir, f"global_shap_aggregated{suffix(class_index)}.csv"),
            index=False
        )

        export_shap_matrix(expl, feature_names, global_exp_idx,
                           os.path.join(global_dir,
                                        f"global_shap_values{suffix(class_index)}.csv"))

        shap_values_global_top = aggregate_global_top_k(expl, feature_names=feature_names, k=10,)

        plt.figure()
        shap.summary_plot(shap_values_global_top, show=False, sort=False)
        if multiclass:
            plt.title(f"class {class_index} ({names[class_index]})")
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, f"global_beeswarm{suffix(class_index)}.pdf"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        global_importance_acc += mean_abs
        pooled_global.append((values, np.asarray(expl.base_values, dtype=float).reshape(-1)))

    # Local explanations (background: local)
    local_expls = build_class_explanations(
        f_classes, Xl_bg, Xl_exp, feature_names, len(class_indices))

    local_importance_acc = np.zeros(len(feature_names), dtype=float)

    for class_index, expl in zip(class_indices, local_expls):
        local_dir = os.path.join(base_local_dir, str(class_index)) if multiclass else base_local_dir
        os.makedirs(local_dir, exist_ok=True)

        values = np.asarray(expl.values, dtype=float)
        mean_abs = np.abs(values).mean(axis=0)

        pd.DataFrame({
            "Feature": feature_names,
            "LocalShapValAgg": mean_abs
        }).to_csv(
            os.path.join(local_dir, f"local_shap_aggregated{suffix(class_index)}.csv"),
            index=False
        )

        export_shap_matrix(expl, feature_names, local_exp_idx,
                           os.path.join(local_dir,
                                        f"local_shap_values{suffix(class_index)}.csv"))

        for i in range(Xl_exp.shape[0]):
            print(f"[WEKA] Plotting instance {i} for class {class_index} ...")
            expl_single = aggregate_local_top_k(expl, df_local, feature_names, row_index=i, k=10,)

            plt.figure()
            shap.plots.waterfall(expl_single, max_display=11, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(local_dir, f"local_instance_{i}{suffix(class_index)}.pdf"),
                        dpi=300, bbox_inches='tight')
            plt.close()

        local_importance_acc += mean_abs

    if multiclass:
        # One beeswarm covering every class: the per-class explanations stacked
        # row-wise, so the plot shows the signed SHAP distribution of the whole
        # model rather than one class at a time. Top-k selection matches the
        # per-class plots, so the two are directly comparable.
        pooled = Explanation(
            values=np.vstack([v for v, _ in pooled_global]),
            base_values=np.concatenate([b for _, b in pooled_global]),
            data=np.vstack([Xg_exp] * len(pooled_global)),
            feature_names=feature_names,
        )
        pooled_top = aggregate_global_top_k(pooled, feature_names=feature_names, k=10)
        plt.figure()
        shap.summary_plot(pooled_top, show=False, sort=False)
        plt.title(f"All {len(pooled_global)} classes pooled "
                  f"({pooled.values.shape[0]} explanations)")
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, "global_beeswarm_all_classes.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[WEKA] wrote global_beeswarm_all_classes.pdf "
              f"({len(pooled_global)} classes x {Xg_exp.shape[0]} instances "
              f"= {pooled.values.shape[0]} rows)")

        pd.DataFrame({
            "Feature": feature_names,
            "GlobalSHAPValAgg_AllClasses": global_importance_acc / len(class_indices)
        }).to_csv(os.path.join(global_dir, "global_shap_aggregated_all_classes.csv"), index=False)

        pd.DataFrame({
            "Feature": feature_names,
            "LocalSHAPValAgg_AllClasses": local_importance_acc / len(class_indices)
        }).to_csv(os.path.join(base_local_dir, "local_shap_aggregated_all_classes.csv"), index=False)

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
    global_bg_idx = sample_indices(rng, n_global, shap_global_bg_sample_size, "global background")
    global_exp_idx = sample_indices(rng, n_global, shap_global_exp_sample_size, "global explained")
    local_bg_idx = sample_indices(rng, n_local, shap_local_bg_sample_size, "local background")
    local_exp_idx = sample_indices(rng, n_local, shap_local_exp_sample_size, "local explained")

    Xg_bg = Xg_full[global_bg_idx]
    Xg_exp = Xg_full[global_exp_idx]

    Xl_bg = Xl_full[local_bg_idx]
    Xl_exp = Xl_full[local_exp_idx]
    df_local = df_local_full.iloc[local_exp_idx].reset_index(drop=True)

    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)

    base_local_dir = os.path.join(output_dir, "local")
    os.makedirs(base_local_dir, exist_ok=True)

    # Per-label global explanations, kept so they can be pooled into one beeswarm.
    pooled_global = []
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

        export_shap_matrix(
            shap_values_global, feature_names, global_exp_idx,
            os.path.join(global_dir, f"global_shap_values_label_{label_index}.csv"))

        base_g = (np.asarray(shap_values_global.base_values, dtype=float).reshape(-1)
                  if isinstance(shap_values_global, Explanation)
                  else np.asarray([explainer_global.expected_value], dtype=float))
        if base_g.size == 1:
            base_g = np.repeat(base_g, global_array.shape[0])
        pooled_global.append((global_array, base_g))

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

        export_shap_matrix(
            shap_values_local, feature_names, local_exp_idx,
            os.path.join(local_dir, f"local_shap_values_label_{label_index}.csv"))

        for i in range(Xl_exp.shape[0]):
            expl_single = aggregate_local_top_k(shap_values_local, df_local, feature_names, row_index=i, k=10,)

            plt.figure()
            shap.plots.waterfall(expl_single, max_display=11, show=False)
            plt.tight_layout()
            plt.savefig( os.path.join(local_dir, f"local_instance_{i}_label_{label_index}.pdf"), dpi=300, bbox_inches='tight')
            plt.close()

    # One beeswarm covering every label: the per-label explanations stacked row-wise,
    # so the plot shows the signed SHAP distribution of the whole multi-label model
    # rather than one label at a time. Top-k selection is the same as the per-label
    # plots, so the two are directly comparable.
    if pooled_global:
        pooled = Explanation(
            values=np.vstack([v for v, _ in pooled_global]),
            base_values=np.concatenate([b for _, b in pooled_global]),
            data=np.vstack([Xg_exp] * len(pooled_global)),
            feature_names=feature_names,
        )
        pooled_top = aggregate_global_top_k(pooled, feature_names=feature_names, k=10)
        plt.figure()
        shap.summary_plot(pooled_top, show=False, sort=False)
        plt.title(f"All {len(pooled_global)} labels pooled "
                  f"({pooled.values.shape[0]} explanations)")
        plt.tight_layout()
        plt.savefig(os.path.join(global_dir, "global_beeswarm_all_labels.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[MEKA] wrote global_beeswarm_all_labels.pdf "
              f"({len(pooled_global)} labels x {Xg_exp.shape[0]} instances "
              f"= {pooled.values.shape[0]} rows)")

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
        default=1000,
        help="No. of Samples for Shap Explainer"
    )

    parser.add_argument(
        "--shap_global_exp_samples",
        default=1000,
        help="No. of Samples to explain for Global Explainability"
    )

    parser.add_argument(
        "--shap_local_bg_samples",
        default=1000,
        help="No. of Samples for Shap Explainer"
    )

    parser.add_argument(
        "--shap_local_exp_samples",
        default=1000,
        help="No. of Samples to explain for Global Explainability"
    )

    parser.add_argument(
        "--class_names",
        default="",
        help="Comma-separated class names in class-index order, used for plot "
             "titles and class_index.csv. Optional; indices are used if omitted."
    )

    args = parser.parse_args()

    class_names = [n for n in (s.strip() for s in args.class_names.split(",")) if n]

    if args.toolkit == "weka":
        explain_weka(
            server_url=args.server_url, global_csv=args.global_csv, local_csv=args.local_csv, output_dir=args.output_dir,
            shap_global_bg_sample_size=int(args.shap_global_bg_samples),
            shap_global_exp_sample_size=int(args.shap_global_exp_samples),
            shap_local_bg_sample_size=int(args.shap_local_bg_samples),
            shap_local_exp_sample_size=int(args.shap_local_exp_samples),
            class_names=class_names
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