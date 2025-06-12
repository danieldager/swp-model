import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*set_ticklabels\(\) should only be used with a fixed number of ticks.*",
)

import matplotlib.pyplot as plt
import numpy as np


def set_edge_ticks(
    ax, tick_fontsize=22, x_decimal_places=2, y_decimal_places=2, bins: bool = False
):
    # Force the figure to render so we get the final limits.
    plt.draw()
    # Retrieve current x- and y-axis limits.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # round x limits
    x_min = int(round(x_min))
    x_max = int(round(x_max))

    # Set x-axis ticks to exactly the minimum and maximum values.
    if bins:
        ax.set_xticks(list(range(x_min, x_max + 1)))
        ax.set_xticklabels([" " for _ in range(x_min, x_max + 1)])
    else:
        ax.set_xticks([x_min, x_max])
        ax.set_xticklabels(
            [f"{x_min:.{x_decimal_places}f}", f"{x_max:.{x_decimal_places}f}"],
            fontsize=tick_fontsize,
        )

    # For the y-axis, lower limit will be set a bit below zero for clarity.
    n_digits = y_decimal_places - int(np.floor(np.log10(abs(y_max)))) - 1
    rounded = 5 * np.ceil(y_max * 10**n_digits / 5) / 10**n_digits
    # rounded = np.ceil(y_max, n_digits)
    tick = f"{rounded:.{n_digits}f}"
    if tick[-1] == "0":
        tick = tick[:-1]
    ax.set_yticks([0, rounded])
    ax.set_yticklabels(["0", tick], fontsize=tick_fontsize)
    new_y_min = -y_max / 21
    ax.set_ylim(new_y_min, rounded)

    # use linspace to get a range of values between the min and max
    x_lines = np.linspace(x_min, x_max, 6)
    y_lines = np.linspace(0, rounded, 6)

    # Draw grid lines
    for x in x_lines:
        ax.axvline(x, color="gray", linewidth=0.5, linestyle="--")
    for y in y_lines:
        ax.axhline(y, color="gray", linewidth=0.5, linestyle="--")
