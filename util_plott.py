"""Plot CoP data from mocap CSV logs (e.g. output.csv)."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

MOCAP_COLUMNS = ["time_sent", "time_recv", "copR", "copL", "Frz", "Flz"]


def load_mocap_csv(csv_path: str) -> pd.DataFrame:
    """Load a mocap CSV and add a normalized time column starting at 0 s."""
    df = pd.read_csv(csv_path, header=None, names=MOCAP_COLUMNS)
    df["time"] = df["time_recv"] - df["time_recv"].iloc[0]
    return df


def plot_cop(
    df: pd.DataFrame,
    time_min: float | None = None,
    time_max: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot copL and copR vs time in two stacked subplots."""
    df_plot = df.copy()
    if time_min is not None:
        df_plot = df_plot[df_plot["time"] >= time_min]
    if time_max is not None:
        df_plot = df_plot[df_plot["time"] <= time_max]

    fig, (ax_l, ax_r) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax_l.plot(df_plot["time"], df_plot["copL"], color="tab:blue")
    ax_l.set_ylabel("copL")
    ax_l.grid(True)

    ax_r.plot(df_plot["time"], df_plot["copR"], color="tab:orange")
    ax_r.set_ylabel("copR")
    ax_r.set_xlabel("Time (s)")
    ax_r.grid(True)

    fig.suptitle("Center of Pressure")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot copL and copR from a mocap CSV.")
    parser.add_argument("csv_path", nargs="?", default="output.csv", help="Path to CSV file")
    parser.add_argument("--time-min", type=float, default=None, help="Start time (s)")
    parser.add_argument("--time-max", type=float, default=None, help="End time (s)")
    parser.add_argument("--save", default=None, help="Save figure to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    args = parser.parse_args()

    df = load_mocap_csv(args.csv_path)
    plot_cop(
        df,
        time_min=args.time_min,
        time_max=args.time_max,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
