from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from t2_spline import DEFAULT_SPLINE_PARAMS, spline_generator


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "test_run" / "AB01_1_scale_0.35_output_torque.csv"

OUTPUT_PATH_OVERVIEW = BASE_DIR / "test_run" / "AB01_1_scale_0.35_output_torque_mtr_cmd_plot.png"
OUTPUT_PATH_PROFILE = BASE_DIR / "test_run" / "AB01_1_scale_0.35_output_torque_gait_profile_plot.png"

START_ROW, END_ROW = 500, 4000
STRIDE_TOL = 0.3  # keep strides within +/-30% of the median stride time

# The gait-cycle % is not logged in the CSV, so it is reconstructed from the
# command. The command is spline[gc%], whose extension (negative) peak always
# occurs at the same gait-cycle %. Each extension peak is therefore an anchor
# at gc = PEAK_GC, and gc is linearly interpolated between consecutive peaks.
PEAK_GC = float(np.argmin(spline_generator(**DEFAULT_SPLINE_PARAMS)))


def reconstruct_gait(time, cmd):
	"""Return (heel_strike_times, gc_percent_per_sample, strides) for one leg."""
	peaks, _ = find_peaks(-cmd, height=0.5 * np.max(-cmd), distance=30)
	t_pk = time[peaks]
	periods = np.diff(t_pk)
	median_T = np.median(periods)

	# Heel strike = PEAK_GC percent of a stride before each peak.
	heel_strikes = t_pk[1:] - PEAK_GC / 100.0 * periods

	gc = np.full(len(time), np.nan)
	strides = []
	for k, T in enumerate(periods):
		if abs(T - median_T) > STRIDE_TOL * median_T:
			continue  # pause / dropped stride
		t0 = heel_strikes[k]
		t1 = t0 + T
		mask = (time >= t0) & (time < t1)
		gc[mask] = 100.0 * (time[mask] - t0) / T
		if mask.sum() > 5:
			strides.append(np.interp(np.linspace(0, 100, 101), gc[mask], cmd[mask]))
	return heel_strikes, gc, np.array(strides)


def label_directions(ax) -> None:
	"""Mark heel strike and the flexion (+) / extension (-) torque directions."""
	ax.axhline(0, color="black", linewidth=0.8)
	ax.axvline(0, color="gray", linestyle="--", linewidth=1)
	ax.text(0.5, 0.02, "heel strike", transform=ax.get_xaxis_transform(),
		rotation=90, va="bottom", ha="left", color="gray")
	ax.text(1.01, 0.75, "Flexion (+) ↑", transform=ax.transAxes, rotation=90,
		va="center", ha="left", color="tab:green", fontweight="bold")
	ax.text(1.01, 0.25, "↓ Extension (−)", transform=ax.transAxes, rotation=90,
		va="center", ha="left", color="tab:red", fontweight="bold")


def main() -> None:
	df = pd.read_csv(CSV_PATH).iloc[START_ROW:END_ROW]
	time = df["time"].to_numpy()
	legs = {"L": ("mtr_cmd_L", "tab:blue"), "R": ("mtr_cmd_R", "tab:orange")}

	results = {
		leg: reconstruct_gait(time, df[col].to_numpy()) for leg, (col, _) in legs.items()
	}

	# Figure 1: overall command (top) and gait cycle % (bottom)
	fig1, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
	for leg, (col, color) in legs.items():
		ax_top.plot(time, df[col], label=col, color=color, linewidth=1.5)
		hs, gc, _ = results[leg]
		ax_bot.plot(time, gc, label=f"gait cycle {leg}", color=color, linewidth=1.2)
		ax_bot.plot(hs, np.zeros_like(hs), "v", color=color, markersize=5)
	ax_top.set_ylabel("motor command")
	ax_top.set_title("mtr_cmd_L and mtr_cmd_R")
	ax_top.legend(loc="lower left")
	ax_bot.set_xlabel("time (s)")
	ax_bot.set_ylabel("gait cycle (%) from heel strike")
	ax_bot.set_ylim(0, 100)
	ax_bot.legend(loc="upper right")
	fig1.tight_layout()

	# Figure 2: torque profile vs gait cycle, mean +/- std
	fig2, ax = plt.subplots(figsize=(9, 5))
	x = np.linspace(0, 100, 101)
	for leg, (col, color) in legs.items():
		strides = results[leg][2]
		mean, std = strides.mean(axis=0), strides.std(axis=0)
		ax.plot(x, mean, color=color, linewidth=2, label=f"{col} mean (n={len(strides)})")
		ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25)
	label_directions(ax)
	ax.set_xlabel("gait cycle (%), 0 = heel strike")
	ax.set_ylabel("motor command (Nm)")
	ax.set_title("Torque profile vs gait cycle (mean ± SD)")
	ax.set_xlim(0, 100)
	ax.grid(True, alpha=0.3)
	ax.legend(loc="upper left")
	fig2.tight_layout()

	fig1.savefig(OUTPUT_PATH_OVERVIEW, dpi=160)
	fig2.savefig(OUTPUT_PATH_PROFILE, dpi=160)

	plt.show()


if __name__ == "__main__":
	main()
