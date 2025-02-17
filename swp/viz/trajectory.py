import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_trajectories(
    trajectory_df: pd.DataFrame, checkpoint: str, dir: pathlib.Path, mode: str
):
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
    plt.savefig(dir / f"{mode}_{checkpoint}_traj.png", dpi=300)
    plt.close()


def sns_plot_trajectories(
    trajectory_df: pd.DataFrame, checkpoint: str, dir: pathlib.Path, mode: str
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
    plt.savefig(dir / f"sns_{mode}_{checkpoint}_traj.png", dpi=300)
    plt.close()


# mark/color the last (!) timepoint of the stimuli based on the factorial design
# (lexicality (blue vs. red), freq (hue of blue), morph complex (square vs. circles, or continuous vs dashed line)
# and length (will be seen by the length of the trajectory, or you could use the size of the markers to highlight the total length).
# keep the rest of the trajectory with alpha=0.2, or grey, we need to try.
