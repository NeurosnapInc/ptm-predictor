"""
This script calculates and visualizes basic statistics for the processed PTM (Post-Translational Modification)
datasets located in the `ptms_clean/` directory.

It performs the following tasks:
1. **Reads each CSV file** in `ptms_clean/` and counts the number of entries (rows) per file.
2. **Visualizes dataset size distribution** using:
   - A bar chart showing the number of entries per PTM file.
   - A histogram showing how PTM files are distributed based on entry count.
3. **Prints summary statistics**, including:
   - Mean and standard deviation of PTM entries across files.
   - Number of files with fewer than 100, fewer than 1000, and more than 1000 entries.

This script is useful for identifying class imbalance issues, spotting outlier files, and determining thresholds
for filtering PTMs based on data volume.

To run:
Simply execute the script after generating cleaned CSVs using the PTM preprocessing pipeline.
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statistics import fmean, stdev

x = []
y = []
for fname in os.listdir("ptms_clean"):
  if fname.endswith(".csv"):  # Optional: only process .csv files
    df = pd.read_csv(f"ptms_clean/{fname}")
    x.append(fname)
    y.append(len(df))

# Bar Chart: File vs Row Count
bar_fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color="royalblue")])
bar_fig.update_layout(title="Number of Entries per PTM File", xaxis_title="File Name", yaxis_title="Number of Entries", xaxis_tickangle=-45)
bar_fig.show()

# Histogram: Distribution of Row Counts Across Files
hist_fig = px.histogram(y, nbins=20, title="Distribution of Entry Counts Across PTM Files")
hist_fig.update_layout(xaxis_title="Number of Entries", yaxis_title="Frequency")
hist_fig.show()

print("=" * 25, "DATASET STATISTICS", "=" * 25)
print(f"PTMs mean.: {fmean(y):,.0f}")
print(f"PTMs stdev: {stdev(y):,.0f}")
print(f"Below 100.: {sum(1 for e in y if e < 100):,.0f}")
print(f"Below 1000: {sum(1 for e in y if e < 1000):,.0f}")
print(f"Above 1000: {sum(1 for e in y if e > 1000):,.0f}")