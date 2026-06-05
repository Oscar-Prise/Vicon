import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("output.csv", header=None)
df.columns = ["time_sent", "time_recv", "copR", "copL", "Frz", "Flz"]
# Normalize time so the plot starts at 0 seconds.
df["time"] = df["time_sent"] - df["time_sent"].iloc[0]
# Keep only data between 5 and 10 seconds (inclusive).
df_plot = df[(df["time"] >= 5) & (df["time"] <= 10)]

plt.plot(df_plot["time"].to_numpy(), df_plot["Frz"].to_numpy(), label="copR")
plt.plot(df_plot["time"].to_numpy(), df_plot["Flz"].to_numpy(), label="copL")
plt.legend()
plt.grid(True)
plt.savefig("cop_plot.png")