# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "seaborn",
# ]
# ///

# chart.py
# Author: 23f2005559@ds.study.iitm.ac.in

# chart.py
# Author: 23f2005559@ds.study.iitm.ac.in

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set professional style
sns.set_style("whitegrid")
sns.set_context("talk")

# Generate synthetic seasonal revenue data
np.random.seed(42)
months = pd.date_range(start="2022-01", periods=24, freq="M")
segments = ["Premium", "Standard", "Budget"]

data = []
for segment in segments:
    base = np.linspace(20000, 40000, len(months))
    seasonal = 5000 * np.sin(np.linspace(0, 4*np.pi, len(months)))
    noise = np.random.normal(0, 2000, len(months))
    multiplier = {"Premium": 1.4, "Standard": 1.0, "Budget": 0.7}[segment]
    revenue = (base + seasonal + noise) * multiplier
    for m, r in zip(months, revenue):
        data.append([m, segment, r])

df = pd.DataFrame(data, columns=["Month", "Segment", "Revenue"])

# Plot lineplot
plt.figure(figsize=(8, 8))  # (8 inches * 64 dpi = 512 pixels)
sns.lineplot(data=df, x="Month", y="Revenue", hue="Segment", palette="deep", linewidth=2.5)

# Titles and labels
plt.title("Monthly Revenue Trends by Customer Segment", fontsize=16, weight="bold")
plt.xlabel("Month", fontsize=12)
plt.ylabel("Revenue (USD)", fontsize=12)

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Save chart with exact dimensions
plt.savefig("chart.png", dpi=64)  # no bbox_inches
plt.close()
