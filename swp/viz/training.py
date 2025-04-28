import pathlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import pchip_interpolate
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils.paths import get_figures_dir

sns.set_palette("colorblind")


# Plot the training and validation loss curves
def training_curves(train_losses: list, valid_losses: list, model: str, n_epochs: int):
    # Extract parameters from the model name
    h, r, d, t, l, m, f = [p[1:] for p in model.split("_")]
    m = "RNN" if m[0] == "n" else "LSTM"

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(1, n_epochs + 1), y=train_losses, label="Training")
    sns.lineplot(x=range(1, n_epochs + 1), y=valid_losses, label="Validation")
    plt.title(f"{m} (fold {f}): H={h}, LR={r}, L={l}, D={d}, TF={t}, LR={r}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()

    MODEL_FIGURES_DIR = get_figures_dir() / model
    MODEL_FIGURES_DIR.mkdir(exist_ok=True)
    filename = MODEL_FIGURES_DIR / "training.png"
    plt.savefig(filename, dpi=300)  # , bbox_inches="tight")
    plt.close()
