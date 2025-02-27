import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_trajectories(trajectory_df: pd.DataFrame, dir: pathlib.Path, filename: str):
    color_dict = {
        lexicality: sns.color_palette("colorblind")[i]  # type: ignore
        for i, lexicality in enumerate(trajectory_df["Lexicality"].unique())
    }
    style_dict = {
        "complex": "-",
        "simple": "--",
    }
    for i, row in trajectory_df.iterrows():
        x_coords, y_coords = row["Trajectory"]

        plt.plot(
            x_coords,
            y_coords,
            color=color_dict[row["Lexicality"]],
            linestyle=style_dict[row["Morphology"]],
            alpha=0.2,
        )
        # add scatter plot
    plt.savefig(dir / filename, dpi=300)
    plt.close()


def sns_plot_trajectories(
    trajectory_df: pd.DataFrame, dir: pathlib.Path, filename: str
):
    trajectory_data = []
    for i, row in trajectory_df.iterrows():
        traj = row["Trajectory"]
        for j in range(len(traj[0])):
            trajectory_data.append(
                {
                    "Trajectory ID": i,
                    "Step": j,
                    "X": traj[0][j],
                    "Y": traj[1][j],
                    "Lexicality": row["Lexicality"],
                    "Morphology": row["Morphology"],
                    "Length": row["Length"],
                }
            )

    # trajectory_data = [
    #     {
    #         "Trajectory ID": i,
    #         "Step": j,
    #         "X": x,
    #         "Y": y,
    #         "Lexicality": row["Lexicality"],
    #         "Morphology": row["Morphology"],
    #     }
    #     for i, row in trajectory_df.iterrows()
    #     for j, (x, y) in enumerate(zip(row["Trajectory"][0], row["Trajectory"][1]))
    # ]
    data_df = pd.DataFrame(trajectory_data)
    sns.lineplot(
        data=data_df,
        x="X",
        y="Y",
        hue="Length",
        # hue="Lexicality",
        style="Morphology",
        markers=True,
        sort=False,
        units="Trajectory ID",
        estimator=None,  # type: ignore
    )
    plt.savefig(dir / filename, dpi=300)
    plt.close()


# mark/color the last (!) timepoint of the stimuli based on the factorial design
# (lexicality (blue vs. red), freq (hue of blue), morph complex (square vs. circles, or continuous vs dashed line)
# and length (will be seen by the length of the trajectory, or you could use the size of the markers to highlight the total length).
# keep the rest of the trajectory with alpha=0.2, or grey, we need to try.


def better_plot_trajectories(
    trajectory_df: pd.DataFrame, dir: pathlib.Path, filename: str
):
    color_dict = {
        lexicality: sns.color_palette("colorblind")[i]  # type: ignore
        for i, lexicality in enumerate(trajectory_df["Lexicality"].unique())
    }
    style_dict = {
        "complex": "D",
        "simple": "o",
    }
    middles = {}
    types = {}
    lasts = {}
    for i, row in trajectory_df.iterrows():
        x_coords, y_coords = row["Trajectory"]

        plt.plot(x_coords, y_coords, color="grey", alpha=0.1, zorder=0)
        last = len(x_coords) - 1
        for j in range(len(x_coords)):
            if j == last:
                curr = lasts.setdefault(j, [])
                type_curr = types.setdefault((row["Morphology"], row["Lexicality"]), [])
                type_curr.append((x_coords[j], y_coords[j]))
            else:
                curr = middles.setdefault(j, [])
            curr.append((x_coords[j], y_coords[j]))
    for index, coords in middles.items():
        x_coords, y_coords = zip(*coords)
        plt.scatter(
            x=x_coords,
            y=y_coords,
            marker=f"${index}$",  # type: ignore
            zorder=1,
            color="black",
            alpha=0.2,
        )
    for (morph, lex), coords in types.items():
        x_coords, y_coords = zip(*coords)
        plt.scatter(
            x=x_coords,
            y=y_coords,
            s=100,
            marker=style_dict[morph],  # type: ignore
            color=color_dict[lex],
            zorder=2,
        )
    for index, coords in lasts.items():
        x_coords, y_coords = zip(*coords)
        plt.scatter(
            x=x_coords,
            y=y_coords,
            marker=f"${index}$",  # type: ignore
            color="black",
            zorder=3,
            alpha=0.5,
        )
    plt.savefig(dir / filename, dpi=300)
    plt.close()
