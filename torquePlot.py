from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "test_run" / "AB01_1_scale_0.7_output_torque.csv"
OUTPUT_PATH = BASE_DIR / "test_run" / "AB01_1_scale_0.7_output_torque_mtr_cmd_plot.png"


def main() -> None:
	df = pd.read_csv(CSV_PATH)
	slice_df = df.iloc[500:4000]

	plt.figure(figsize=(12, 5))
	plt.plot(slice_df["time"], slice_df["mtr_cmd_L"], label="mtr_cmd_L", linewidth=1.5)
	plt.plot(slice_df["time"], slice_df["mtr_cmd_R"], label="mtr_cmd_R", linewidth=1.5)
	plt.xlabel("time (s)")
	plt.ylabel("motor command")
	plt.title("mtr_cmd_L and mtr_cmd_R, rows 100-3000")
	plt.legend()
	plt.tight_layout()
	plt.savefig(OUTPUT_PATH, dpi=160)
	plt.show()


if __name__ == "__main__":
	main()
