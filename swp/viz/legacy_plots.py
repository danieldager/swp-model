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


# Plot the confusion matrix for the test data
# def confusion_matrix(confusions: dict, model_name: str, epoch: str) -> None:
#     # TODO Daniel docstring

#     # Initialize the confusion matrix
#     # confusions = {}
#     # for t in phoneme_stats.keys():
#     #     confusions[t] = {p: 0 for p in phoneme_stats.keys()}

#     # Tabulate confusion between prediction and target
#     # if len(target) == len(prediction):
#     #     for t, p in zip(target, prediction):
#     #         confusions[t][p] += 1
#     df = pd.DataFrame.from_dict(confusions, orient="index")

#     # TODO get_phoneme_to_id
#     phonemes = None

#     # Normalize the confusion matrix
#     # df = np.log1p(df) # log scale
#     # df = df.div(df.sum(axis=1), axis=0) # row normal
#     # df = (df - df.min().min()) / (df.max().max() - df.min().min()) # min max
#     df = (df - df.mean()) / df.std()  # z score

#     # Create a confusion matrix
#     plt.figure(figsize=(8, 7))

#     # Plot the heatmap with enhanced aesthetics
#     heatmap = sns.heatmap(
#         df,
#         annot=False,
#         cmap="Blues",
#         square=True,
#         cbar_kws={"label": "Counts"},
#         xticklabels=True,
#         yticklabels=True,
#     )

#     # Adjust the colorbar
#     cbar = heatmap.collections[0].colorbar
#     cbar.ax.set_aspect(10)  # Larger values make the colorbar narrower

#     # Display X-ticks on the top
#     plt.gca().xaxis.tick_top()
#     plt.gca().xaxis.set_label_position("top")

#     # Set axis labels and title
#     plt.title("Confusion Matrix", fontsize=14, pad=20)
#     plt.xlabel("Prediction", fontsize=10)
#     plt.ylabel("Ground Truth", fontsize=10)
#     plt.xticks(fontsize=5, rotation=90)
#     plt.yticks(fontsize=5, rotation=0)
#     plt.tight_layout()

#     MODEL_FIGURES_DIR = get_figures_dir() / model_name
#     MODEL_FIGURES_DIR.mkdir(exist_ok=True)

#     filename = f"confusion{epoch}.png"
#     plt.savefig(MODEL_FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
#     plt.close()
