from __future__ import annotations

from ast import literal_eval
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from swp.datasets.phonemes import get_phoneme_to_id
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*ChainedAssignmentError.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="seaborn",
)

PLOT_DPI = 150
FIG_RECT = [0, 0, 1, 0.95]
sns.set_style("whitegrid")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, path: Path, rect: list[float] = FIG_RECT) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _flatten_scales(scales: np.ndarray) -> np.ndarray:
    return scales.reshape(scales.shape[0], -1) if scales.ndim > 2 else scales


def _coerce_match(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(
            series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0}),
            errors="coerce",
        )
    return pd.to_numeric(series, errors="coerce")


def load_run_config(run_dir: Path) -> dict[str, object]:
    with open(run_dir / "config.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_history(run_dir: Path) -> pd.DataFrame:
    history = pd.read_csv(run_dir / "history.csv")
    history["epoch"] = history.index
    return history


def load_prediction_data(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "predictions.csv")


def load_scale_params(run_dir: Path) -> dict[str, np.ndarray]:
    return dict(np.load(run_dir / "scale_params.npz", allow_pickle=True))


def plot_training_history(history: dict[str, list[float]] | pd.DataFrame, save_dir: Path, title: str = "") -> None:
    _ensure_dir(save_dir)
    history_df = pd.DataFrame(history) if isinstance(history, dict) else history.copy()
    history_df["epoch"] = range(len(history_df))

    loss_cols = [col for col in ["train_loss", "val_loss", "test_loss"] if col in history_df.columns]
    acc_cols = [col for col in ["train_acc", "val_acc", "test_acc"] if col in history_df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    sns.lineplot(data=history_df.melt(id_vars="epoch", value_vars=loss_cols, var_name="split", value_name="loss"), x="epoch", y="loss", hue="split", marker="o", ax=axes[0])
    axes[0].set(xlabel="Epoch", ylabel="Loss")

    sns.lineplot(data=history_df.melt(id_vars="epoch", value_vars=acc_cols, var_name="split", value_name="accuracy"), x="epoch", y="accuracy", hue="split", marker="o", ax=axes[1])
    axes[1].set(xlabel="Epoch", ylabel="Accuracy")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="Split")

    summary = [f"Train: {history_df.iloc[-1]['train_acc']:.4f}", f"Val: {history_df.iloc[-1]['val_acc']:.4f}"]
    if "test_acc" in history_df.columns:
        summary.append(f"Test: {history_df.iloc[-1]['test_acc']:.4f}")
    axes[1].text(0.95, 0.05, "\n".join(summary), transform=axes[1].transAxes, ha="right", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))

    if title:
        fig.suptitle(title, y=0.98, fontsize=14)

    _save_figure(fig, save_dir / "training_curves.png")


def plot_scale_norms(scales: np.ndarray, save_dir: Path, title_suffix: str = "") -> None:
    _ensure_dir(save_dir)
    scales = _flatten_scales(scales)
    norms = np.linalg.norm(scales, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=np.arange(len(norms)), y=norms, marker="o", ax=ax)
    ax.set(xlabel="Position from Start", ylabel="Norm of Scale Vector", title=f"Norm of Intervention Scale by Position" + (f" - {title_suffix}" if title_suffix else ""))
    _save_figure(fig, save_dir / "scale_norms.png")


def plot_scale_stats(scales: np.ndarray, save_dir: Path, title_suffix: str = "") -> None:
    _ensure_dir(save_dir)
    scales = _flatten_scales(scales)
    median = np.median(scales, axis=1)
    mean = np.mean(scales, axis=1)
    lower = np.percentile(scales, 25, axis=1)
    upper = np.percentile(scales, 75, axis=1)

    x = np.arange(len(median))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, median, marker="o", label="median")
    ax.plot(x, mean, marker="x", label="mean")
    ax.fill_between(x, lower, upper, alpha=0.25, label="IQR (25-75%)")
    ax.set(xlabel="Position from Start", ylabel="Scale value", title=f"Median scale across hidden units by position" + (f" - {title_suffix}" if title_suffix else ""))
    ax.legend()
    _save_figure(fig, save_dir / "scale_stats.png")



def plot_scale_pca_polar(scales: np.ndarray, save_dir: Path, title_suffix: str = "") -> None:
    _ensure_dir(save_dir)
    pca = PCA(n_components=2)
    scales_2d = pca.fit_transform(_flatten_scales(scales))
    colors = np.arange(scales_2d.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=scales_2d[:, 0], y=scales_2d[:, 1], hue=colors, palette="viridis", legend=False, s=80, ax=axes[0])
    axes[0].set(xlabel=f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)", ylabel=f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)", title="Scale PCA 2D")
    axes[0].grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    angles = np.arctan2(scales_2d[:, 1], scales_2d[:, 0])
    r = np.linalg.norm(scales_2d, axis=1)
    scatter = ax2.scatter(angles, r, c=colors, cmap="viridis", s=80)
    fig.colorbar(scatter, ax=ax2, label="Position from Start", orientation="vertical")
    ax2.set_title("Scale PCA Polar")

    if title_suffix:
        fig.suptitle(title_suffix, y=0.98, fontsize=14)

    _save_figure(fig, save_dir / "scale_pca_polar.png")


def plot_embedding_pca(embedding: np.ndarray, id_to_phoneme: dict[int, str], save_dir: Path, title_suffix: str = "", phoneme_types: dict[str, str] | None = None) -> None:
    _ensure_dir(save_dir)
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding)
    explained_var = pca.explained_variance_ratio_ * 100

    phonemes = [id_to_phoneme.get(i, f"id{i}") for i in range(len(embedding_2d))]
    data = {"PC1": embedding_2d[:, 0], "PC2": embedding_2d[:, 1], "phoneme": phonemes}

    if phoneme_types is not None:
        def _get_type(val):
            if isinstance(val, dict):
                return val.get("Type", "other")
            return val

        data["type"] = [_get_type(phoneme_types.get(p, "other")) for p in phonemes]

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter_args = dict(data=df, x="PC1", y="PC2", s=100, edgecolor="k", ax=ax)
    if "type" in df.columns:
        ax = sns.scatterplot(hue="type", palette="Set2", legend="full", **scatter_args)
        ax.legend(title="Phoneme Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax = sns.scatterplot(**scatter_args)

    for _, row in df.iterrows():
        ax.text(row["PC1"] + 0.005, row["PC2"] + 0.005, row["phoneme"], fontsize=7, alpha=0.8)

    fig_title = f"Embedding PCA"
    if title_suffix:
        fig_title += f" - {title_suffix}"

    ax.set(title=fig_title, xlabel=f"PC1 ({explained_var[0]:.1f}% var)", ylabel=f"PC2 ({explained_var[1]:.1f}% var)")
    _save_figure(fig, save_dir / "embedding_pca.png")


def create_merged_df(pred_df: pd.DataFrame, wfe_df: pd.DataFrame, phoneme_features: dict[str, dict[str, str]]) -> pd.DataFrame:
    wfe_df = wfe_df.copy()
    wfe_df["No_Stress_str"] = wfe_df["No_Stress"].apply(lambda x: " ".join(x).strip() if isinstance(x, (list, tuple)) else str(x).strip())

    pred_df = pred_df.copy()
    # =============================================================
    # target for reverese order 
    pred_df["input_no_eos"] = pred_df["input"].astype(str).str.replace("<EOS>", "", regex=False).str.strip()
    if "match" in pred_df.columns:
        pred_df["match"] = _coerce_match(pred_df["match"])
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
        lambda row: f"{row['old-ph-type']}-{row['new-ph-type']}" if pd.notna(row["old-ph-type"]) and pd.notna(row["new-ph-type"]) else None,
        axis=1,
    )
    return merged_df


def load_wfe_data() -> pd.DataFrame:
    dataset_dir = Path(__file__).resolve().parent / "datasets"
    return pd.read_csv(dataset_dir / "wfe_with_repetition.csv", converters={"Phonemes": literal_eval, "No_Stress": literal_eval})


def load_phoneme_features() -> dict[str, dict[str, str]]:
    dataset_dir = Path(__file__).resolve().parent / "datasets"
    phonemes_info = pd.read_csv(dataset_dir / "phonemes.csv")
    return phonemes_info.set_index("Phoneme").to_dict("index")


def plot_accuracy_by_feature(predictions: pd.DataFrame, save_dir: Path, feature_col: str, title_suffix: str = "") -> None:
    if feature_col not in predictions.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in predictions")

    predictions = predictions.copy()
    if "match" in predictions.columns:
        predictions["match"] = _coerce_match(predictions["match"])

    _ensure_dir(save_dir)
    summary = predictions.groupby(feature_col)["match"].mean().reset_index().sort_values(by="match", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=summary, x=feature_col, y="match", color="steelblue", ax=ax)
    ax.set(xlabel=feature_col, ylabel="Accuracy", title=f"Accuracy by {feature_col}" + (f" - {title_suffix}" if title_suffix else ""))
    ax.title.set_pad(12)
    _save_figure(fig, save_dir / f"accuracy_by_{feature_col}.png")


def plot_feature_accuracies_summary(merged_df: pd.DataFrame, save_dir: Path, feature_cols: list[str], title_suffix: str = "") -> None:
    features = [feature for feature in feature_cols if feature in merged_df.columns]
    if not features:
        return

    merged_df = merged_df.copy()
    if "match" in merged_df.columns:
        merged_df["match"] = _coerce_match(merged_df["match"])

    n_cols = min(3, len(features))
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for idx, feature in enumerate(features):
        ax = axes[idx // n_cols][idx % n_cols]
        summary = merged_df.groupby(feature)["match"].mean().reset_index().sort_values(by="match", ascending=False)
        sns.barplot(data=summary, x=feature, y="match", color="steelblue", ax=ax)
        ax.set(xlabel=feature, ylabel="Accuracy", title=feature, ylim=(0, 1))
        if summary[feature].dtype == object:
            ax.tick_params(axis="x", rotation=45)

    for extra_ax in axes.flat[len(features):]:
        fig.delaxes(extra_ax)

    fig.suptitle("Accuracy by Feature" + (f" - {title_suffix}" if title_suffix else ""), y=0.98, fontsize=14)
    _save_figure(fig, save_dir / "accuracy_by_features.png")


def plot_run_summary(run_dir: Path, feature_cols: list[str] | None = None, phoneme_types: dict[str, str] | None = None) -> None:
    if phoneme_types is None:
        phoneme_types = load_phoneme_features()

    config = load_run_config(run_dir)
    embedding_init = config.get('embedding_init', 'none')
    title = (
        f"scale_param={config.get('scale_param')} | state_mode={config.get('state_mode')} | "
        f"embed_init={embedding_init} | train_embed={ config.get('train_embedding', False)}"
    )

    plot_training_history(load_history(run_dir), run_dir, title=title)

    params = load_scale_params(run_dir)
    if "scales" in params:
        plot_scale_norms(params["scales"], run_dir, title_suffix=title)
        plot_scale_pca_polar(params["scales"], run_dir, title_suffix=title)

    if "embedding" in params:
        phoneme_to_id = {v: k for k, v in get_phoneme_to_id().items()}
        try:
            plot_embedding_pca(params["embedding"], phoneme_to_id, run_dir, title_suffix=title, phoneme_types=phoneme_types)
        except Exception as exc:
            error_text = f"Embedding PCA plot failed: {exc}\n"
            (run_dir / "analysis_plot_errors.txt").write_text(error_text, encoding="utf-8")
            print(error_text)
    else:
        print("No embedding key found in scale_params.npz")

    if feature_cols is not None:
        predictions = load_prediction_data(run_dir)
        merged_df = create_merged_df(predictions, load_wfe_data(), phoneme_types)
        try:
            plot_feature_accuracies_summary(merged_df, run_dir, feature_cols, title_suffix=title)
        except Exception as exc:
            error_text = f"Feature accuracy plot failed: {exc}\n"
            (run_dir / "analysis_plot_errors.txt").write_text(error_text, encoding="utf-8")
            print(error_text)
        merged_df.to_csv(run_dir / "merged_predictions.csv", index=False)
