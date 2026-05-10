from __future__ import annotations

from pathlib import Path
import json
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from swp.datasets.phonemes import get_phoneme_to_id


def load_run_config(run_dir: Path) -> dict[str, object]:
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_history(run_dir: Path) -> pd.DataFrame:
    history = pd.read_csv(run_dir / "history.csv")
    history["epoch"] = history.index
    return history


def plot_training_history(history: dict[str, list[float]] | pd.DataFrame, save_dir: Path, title: str = "") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    if isinstance(history, dict):
        history_df = pd.DataFrame(history)
        history_df["epoch"] = range(len(history_df))
    else:
        history_df = history.copy()

    loss_cols = [col for col in ["train_loss", "val_loss", "test_loss"] if col in history_df.columns]
    acc_cols = [col for col in ["train_acc", "val_acc", "test_acc"] if col in history_df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

    loss_df = history_df.melt(id_vars="epoch", value_vars=loss_cols, var_name="split", value_name="loss")
    sns.lineplot(data=loss_df, x="epoch", y="loss", hue="split", marker="o", ax=axes[0])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    acc_df = history_df.melt(id_vars="epoch", value_vars=acc_cols, var_name="split", value_name="accuracy")
    sns.lineplot(data=acc_df, x="epoch", y="accuracy", hue="split", marker="o", ax=axes[1])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="Split")

    last_row = history_df.iloc[-1]
    summary_lines = [f"Train: {last_row['train_acc']:.4f}", f"Val: {last_row['val_acc']:.4f}"]
    if "test_acc" in history_df.columns:
        summary_lines.append(f"Test: {last_row['test_acc']:.4f}")
    summary_text = "\n".join(summary_lines)
    axes[1].text(
        0.95,
        0.05,
        summary_text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    if title:
        fig.suptitle(title, y=0.98, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def load_prediction_data(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "predictions.csv")


def load_scale_params(run_dir: Path) -> dict[str, np.ndarray]:
    return dict(np.load(run_dir / "scale_params.npz", allow_pickle=True))


def plot_scale_norms(scales: np.ndarray, save_dir: Path, title_suffix: str = "") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    if scales.ndim > 2:
        scales = scales.reshape(scales.shape[0], -1)
    norms = np.linalg.norm(scales, axis=1)
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=range(len(norms)), y=norms, marker="o")
    plt.xlabel("Position from Start")
    plt.ylabel("Norm of Scale Vector")
    plt.title("Norm of Intervention Scale by Position" + (f" - {title_suffix}" if title_suffix else ""))
    plt.tight_layout()
    plt.savefig(save_dir / "scale_norms.png", dpi=150)
    plt.close()


def plot_scale_stats(scales: np.ndarray, save_dir: Path, title_suffix: str = "") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    if scales.ndim > 2:
        scales = scales.reshape(scales.shape[0], -1)
    median = np.median(scales, axis=1)
    mean = np.mean(scales, axis=1)
    lower = np.percentile(scales, 25, axis=1)
    upper = np.percentile(scales, 75, axis=1)

    x = np.arange(len(median))
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(x, median, marker="o", label="median")
    plt.plot(x, mean, marker="x", label="mean")
    plt.fill_between(x, lower, upper, alpha=0.25, label="IQR (25-75%)")
    plt.xlabel("Position from Start")
    plt.ylabel("Scale value")
    plt.title("Median scale across hidden units by position" + (f" - {title_suffix}" if title_suffix else ""))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "scale_stats.png", dpi=150)
    plt.close()


def pca_reduce(scales: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    if scales.ndim > 2:
        scales = scales.reshape(scales.shape[0], -1)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(scales), pca


def plot_scale_pca_polar(scales: np.ndarray, save_dir: Path, title_suffix: str = "") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    scales_2d, pca = pca_reduce(scales, n_components=2)
    colors = np.arange(scales_2d.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={"projection": None})
    sns.scatterplot(x=scales_2d[:, 0], y=scales_2d[:, 1], hue=colors, palette="viridis", legend=False, s=80, ax=axes[0])
    axes[0].set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[0].set_title("Scale PCA 2D")
    axes[0].grid(alpha=0.3)

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    angles = np.arctan2(scales_2d[:, 1], scales_2d[:, 0])
    r = np.linalg.norm(scales_2d, axis=1)
    scatter = ax.scatter(angles, r, c=colors, cmap="viridis", s=80)
    fig.colorbar(scatter, ax=ax, label="Position from Start", orientation="vertical")
    ax.set_title("Scale PCA Polar")

    if title_suffix:
        fig.suptitle(title_suffix, y=1.03, fontsize=14)

    plt.tight_layout()
    fig.savefig(save_dir / "scale_pca_polar.png", dpi=150)
    plt.close(fig)


def plot_embedding_pca(embedding: np.ndarray, id_to_phoneme: dict[int, str], save_dir: Path, title_suffix: str = "", phoneme_types: dict[str, str] | None = None) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    embedding_2d = PCA(n_components=2).fit_transform(embedding)
    phonemes = [id_to_phoneme.get(i, f"id{i}") for i in range(len(embedding_2d))]
    data = {
        "PC1": embedding_2d[:, 0],
        "PC2": embedding_2d[:, 1],
        "phoneme": phonemes,
    }
    if phoneme_types is not None:
        def _get_type(val):
            if isinstance(val, dict):
                return val.get("Type", "other")
            return val

        data["type"] = [_get_type(phoneme_types.get(p, "other")) for p in phonemes]
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 10))
    if "type" in df.columns:
        ax = sns.scatterplot(data=df, x="PC1", y="PC2", hue="type", palette="Set2", s=100, edgecolor="k")
        plt.legend(title="Phoneme Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax = sns.scatterplot(data=df, x="PC1", y="PC2", s=100, edgecolor="k")

    for _, row in df.iterrows():
        ax.text(row["PC1"] + 0.005, row["PC2"] + 0.005, row["phoneme"], fontsize=7, alpha=0.8)

    plt.title("Embedding PCA" + (f" - {title_suffix}" if title_suffix else ""))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_dir / "embedding_pca.png", dpi=150)
    plt.close()


def create_merged_df(pred_df: pd.DataFrame, wfe_df: pd.DataFrame, phoneme_features: dict[str, dict[str, str]]) -> pd.DataFrame:
    wfe_df = wfe_df.copy()
    wfe_df["No_Stress_str"] = wfe_df["No_Stress"].apply(lambda x: " ".join(x).strip() if isinstance(x, (list, tuple)) else str(x).strip())
    pred_df = pred_df.copy()
    pred_df["input_no_eos"] = pred_df["input"].astype(str).apply(lambda x: x.replace("<EOS>", "").strip())
    if "match" in pred_df.columns:
        pred_df["match"] = (
            pred_df["match"].astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )
    if "token_acc" in pred_df.columns:
        pred_df["token_acc"] = pd.to_numeric(pred_df["token_acc"], errors="coerce")
    if "position" in pred_df.columns:
        pred_df["position"] = pd.to_numeric(pred_df["position"], errors="coerce")

    merged_df = pd.merge(
        pred_df,
        wfe_df.drop(columns=[col for col in ["Phonemes"] if col in wfe_df.columns]),
        left_on=["input_no_eos"],
        right_on=["No_Stress_str"],
        how="left",
    )

    keep_cols = [
        "Word",
        "Condition",
        "input",
        "target",
        "prediction",
        "position",
        "old_phoneme",
        "new_phoneme",
        "seq_len",
        "match",
        "token_acc",
    ]
    keep_cols += [col for col in ["Lexicality", "Size", "Morphology", "Frequency", "Length", "Zipf_Frequency", "Part of Speech"] if col in merged_df.columns]
    merged_df = merged_df[[col for col in keep_cols if col in merged_df.columns]]

    def _phoneme_type(p):
        type_val = phoneme_features.get(p, "other")
        if isinstance(type_val, dict):
            return type_val.get("Type", "other")
        return type_val

    merged_df["old-ph-type"] = merged_df["old_phoneme"].map(_phoneme_type)
    merged_df["new-ph-type"] = merged_df["new_phoneme"].map(_phoneme_type)
    merged_df["type-change"] = merged_df.apply(
        lambda row: f"{row['old-ph-type']}-{row['new-ph-type']}" if pd.notna(row['old-ph-type']) and pd.notna(row['new-ph-type']) else None,
        axis=1,
    )
    return merged_df


def load_wfe_data() -> pd.DataFrame:
    dataset_dir = Path(__file__).resolve().parent / "datasets"
    return pd.read_csv(
        dataset_dir / "wfe_with_repetition.csv",
        converters={"Phonemes": literal_eval, "No_Stress": literal_eval},
    )


def load_phoneme_features() -> dict[str, dict[str, str]]:
    dataset_dir = Path(__file__).resolve().parent / "datasets"
    phonemes_info = pd.read_csv(dataset_dir / "phonemes.csv")
    return phonemes_info.set_index("Phoneme").to_dict("index")


def plot_accuracy_by_feature(predictions: pd.DataFrame, save_dir: Path, feature_col: str, title_suffix: str = "") -> None:
    if feature_col not in predictions.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in predictions")

    if predictions["match"].dtype == object:
        predictions["match"] = pd.to_numeric(predictions["match"].astype(str).str.strip().str.lower().map({"true": "1", "false": "0"}), errors="coerce")

    save_dir.mkdir(parents=True, exist_ok=True)
    summary = predictions.groupby(feature_col)["match"].mean().reset_index().sort_values(by="match", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x=feature_col, y="match", color="steelblue")
    plt.xlabel(feature_col)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by {feature_col}" + (f" - {title_suffix}" if title_suffix else ""))
    plt.tight_layout()
    plt.savefig(save_dir / f"accuracy_by_{feature_col}.png", dpi=150)
    plt.close()


def plot_feature_accuracies_summary(merged_df: pd.DataFrame, save_dir: Path, feature_cols: list[str], title_suffix: str = "") -> None:
    features = [feature for feature in feature_cols if feature in merged_df.columns]
    if not features:
        return

    n_cols = min(3, len(features))
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    if merged_df["match"].dtype == object:
        merged_df["match"] = pd.to_numeric(
            merged_df["match"].astype(str).str.strip().str.lower().map({"true": "1", "false": "0"}),
            errors="coerce",
        )

    for idx, feature in enumerate(features):
        ax = axes[idx // n_cols][idx % n_cols]
        summary = (
            merged_df.groupby(feature)["match"]
            .mean()
            .reset_index()
            .sort_values(by="match", ascending=False)
        )
        sns.barplot(data=summary, x=feature, y="match", color="steelblue", ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(feature)
        if summary[feature].dtype == object:
            ax.tick_params(axis="x", rotation=45)

    for extra_ax in axes.flat[len(features):]:
        fig.delaxes(extra_ax)

    fig.suptitle("Accuracy by Feature" + (f" - {title_suffix}" if title_suffix else ""), y=1.02, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "accuracy_by_features.png", dpi=150)
    plt.close(fig)


def plot_run_summary(run_dir: Path, feature_cols: list[str] | None = None, phoneme_types: dict[str, str] | None = None) -> None:
    if phoneme_types is None:
        phoneme_types = load_phoneme_features()

    config = load_run_config(run_dir)
    title = (
        f"scale_param={config.get('scale_param')} | state_mode={config.get('state_mode')} | "
        f"pretrained={config.get('pretrained')} | freeze_embed={config.get('freeze_embedding', False)}"
    )

    history = load_history(run_dir)
    plot_training_history(history, run_dir, title=title)

    params = load_scale_params(run_dir)
    if "scales" in params:
        plot_scale_norms(params["scales"], run_dir, title_suffix=title)
        plot_scale_pca_polar(params["scales"], run_dir, title_suffix=title)
    if "embedding" in params:
        phoneme_to_id = {v: k for k, v in get_phoneme_to_id().items()}
        try:
            plot_embedding_pca(
                params["embedding"],
                phoneme_to_id,
                run_dir,
                title_suffix=title,
                phoneme_types=phoneme_types,
            )
        except Exception as exc:
            error_text = f"Embedding PCA plot failed: {exc}\n"
            (run_dir / "analysis_plot_errors.txt").write_text(error_text, encoding="utf-8")
            print(error_text)
    else:
        print("No embedding key found in scale_params.npz")

    if feature_cols is not None:
        predictions = load_prediction_data(run_dir)
        wfe_df = load_wfe_data()
        merged_df = create_merged_df(predictions, wfe_df, phoneme_types or load_phoneme_features())
        try:
            plot_feature_accuracies_summary(merged_df, run_dir, feature_cols, title_suffix=title)
        except Exception as exc:
            error_text = f"Feature accuracy plot failed: {exc}\n"
            (run_dir / "analysis_plot_errors.txt").write_text(error_text, encoding="utf-8")
            print(error_text)
        merged_df.to_csv(run_dir / "merged_predictions.csv", index=False)
