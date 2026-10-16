from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from t2_spline import DEFAULT_SPLINE_PARAMS, TNSRE_LG_SPLINE_PARAMS, spline_generator


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "test_run" / "AB01_1_scale_0.2_output_torque.csv"

OUTPUT_PATH_OVERVIEW = BASE_DIR / "test_run" / "AB01_1_scale_0.2_output_torque_mtr_cmd_plot.png"
OUTPUT_PATH_PROFILE = BASE_DIR / "test_run" / "AB01_1_scale_0.2_output_torque_gait_profile_plot.png"

START_ROW, END_ROW = 500, 4000
STRIDE_TOL = 0.3  # keep strides within +/-30% of the median stride time

# The gait-cycle % is not logged in the CSV, so it is reconstructed from the
# command. The command is spline[gc%], whose extension (negative) peak always
# occurs at the same gait-cycle %. Each extension peak is therefore an anchor
# at gc = PEAK_GC, and gc is linearly interpolated between consecutive peaks.
# Must be the profile the CSV was recorded with, or heel strike lands in the
# wrong place: DEFAULT_SPLINE_PARAMS (Franks P1, ext peak at 10%) for trials
# from commit ce83cba on, TNSRE_LG_SPLINE_PARAMS (ext peak at 18%) before that.
PROFILE_PARAMS = DEFAULT_SPLINE_PARAMS
PEAK_GC = float(np.argmin(spline_generator(**PROFILE_PARAMS)))


def reconstruct_gait(time, cmd):
	"""Return (heel_strike_times, gc_percent_per_sample, strides) for one leg.

	Phase is anchored at each extension peak (gc = PEAK_GC) and is linear in
	time between consecutive peaks, so it is continuous and every heel strike
	is exactly where the phase crosses 0/100%.
	"""
	peaks, _ = find_peaks(-cmd, height=0.5 * np.max(-cmd), distance=30)
	t_pk = time[peaks]
	periods = np.diff(t_pk)
	median_T = np.median(periods)
	good = np.abs(periods - median_T) <= STRIDE_TOL * median_T  # False = pause / missed peak

	# Unwrapped phase in percent: PEAK_GC at peak 0, +100 per peak.
	phase_pk = PEAK_GC + 100.0 * np.arange(len(t_pk))
	phase = np.interp(time, t_pk, phase_pk, left=np.nan, right=np.nan)
	seg = np.clip(np.searchsorted(t_pk, time, side="right") - 1, 0, len(good) - 1)
	phase[~good[seg]] = np.nan  # no gait info inside bad intervals
	gc = phase % 100.0

	# Heel strikes: where the phase crosses a multiple of 100.
	m_all = np.arange(1, len(t_pk))  # crossing between peak m-1 and m
	heel_strikes = np.interp(100.0 * m_all, phase_pk, t_pk)
	heel_strikes = heel_strikes[(heel_strikes > t_pk[0]) & (heel_strikes < t_pk[-1])]

	# One stride = heel strike to next heel strike, only if all of it is valid.
	strides = []
	x = np.linspace(0, 100, 101)
	for t0, t1 in zip(heel_strikes[:-1], heel_strikes[1:]):
		mask = (time >= t0) & (time < t1)
		if mask.sum() < 20 or np.isnan(phase[mask]).any():
			continue
		if abs((t1 - t0) - median_T) > STRIDE_TOL * median_T:
			continue
		strides.append(np.interp(x, 100.0 * (time[mask] - t0) / (t1 - t0), cmd[mask]))
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
