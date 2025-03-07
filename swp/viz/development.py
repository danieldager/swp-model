import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Define functions to assign properties based on the condition code
def get_color(condition):
    # Conditions starting with P are red.
    if condition[0] == "P":
        return "blue"
    # Conditions starting with R: if ending with H, dark blue; if ending with L, light blue.
    elif condition[0] == "R":
        if condition[-1] == "H":
            return "darkred"
        elif condition[-1] == "L":
            return "red"
    return "black"  # fallback


def get_linewidth(condition):
    # Second character: L means thicker, S means thinner.
    if condition[1] == "L":
        return 3  # thicker
    elif condition[1] == "S":
        return 1  # thinner
    return 2  # default


def get_linestyle(condition):
    # Third character: S means solid, C means dashed.
    if condition[2] == "S":
        return "-"  # solid/unbroken
    elif condition[2] == "C":
        return "--"  # dashed
    return "-"  # default


def development_plots(df: pd.DataFrame, figures_dir: pathlib.Path, mode: str) -> None:
    conditions = df.columns.tolist()[1:]

    # drop epochs with underscore
    df = df[~df.epoch.str.contains("_")]
    df.epoch = df.epoch.astype(int)
    df = df.sort_values("epoch")

    # separate epochs 1-20 and 21 onwards
    df20 = df[df.epoch <= 20]
    df150 = df[df.epoch > 20]

    df20_long = pd.melt(
        df20,
        id_vars="epoch",  # type: ignore
        value_vars=conditions,
        var_name="Condition",
        value_name="Value",
    )

    # Plotting each condition individually with custom properties
    plt.figure(figsize=(11, 6))
    for condition, group in df20_long.groupby("Condition"):
        plt.plot(
            group["epoch"],
            group["Value"],
            label=condition,
            color=get_color(condition),
            linestyle=get_linestyle(condition),
            linewidth=get_linewidth(condition),
        )

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Error Count", fontsize=18)
    plt.legend(title="Condition", fontsize=16, title_fontsize=18, ncol=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(figures_dir / f"{mode}_20.png", dpi=300)
    plt.close()

    df150_long = pd.melt(
        df150,
        id_vars="epoch",  # type: ignore
        value_vars=conditions,
        var_name="Condition",
        value_name="Value",
    )

    plt.figure(figsize=(11, 6))
    for condition, group in df150_long.groupby("Condition"):
        plt.plot(
            group["epoch"],
            group["Value"],
            label=condition,
            color=get_color(condition),
            linestyle=get_linestyle(condition),
            linewidth=get_linewidth(condition),
        )

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Error Count", fontsize=18)
    plt.legend(title="Condition", fontsize=16, title_fontsize=18, ncol=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    max_epoch = df150.epoch.max()
    plt.savefig(figures_dir / f"{mode}_{max_epoch}.png", dpi=300)
    plt.close()
