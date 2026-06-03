import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("output.csv", header=None)
df.columns = ["copR", "copL", "time"]  # rename to whatever fits your data

# Normalize time so the plot starts at 0 seconds.
df["time"] = df["time"] - df["time"].iloc[0]

plt.plot(df["time"].to_numpy(), df["copR"].to_numpy(), label="copR")
plt.plot(df["time"].to_numpy(), df["copL"].to_numpy(), label="copL")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("cop_plot.png")