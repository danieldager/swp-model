"""Shared look for the report figures: one matplotlib style, one palette.

Imported by ``state_analysis.surprisal`` and by the plotting notebooks, so every panel
agrees on colour, line weight and spacing instead of each module inventing its own.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

INK = "#2B2B2B"
COLORS = {"All": "#334155", "Consonant": "#1565C0", "Vowel": "#E07B39", "EOS": "#9E9E9E"}
SURPRISAL_COLOR = "#7B5AA6"
STRATA_COLORS = ["#9CC7E8", "#3E80D1", "#123B66"]  # low -> high surprisal


def paper_style():
    """Figure defaults: thin dark spines, muted grid, labels with room to breathe."""
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
