import csv
import pandas as pd

import pandas as pd

file_slug = "./emotibit_run2/2024-09-18_22-46-06-359329"
data_pts = ["EA", "EL", "PI", "PR", "PG", "T1", "HR"]
dfs = []

for pt in data_pts:
    df = pd.read_csv(f"{file_slug}_{pt}.csv")
    dfs.append({'type': pt, 'df': df})

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_emotibit_data(dfs, file_slug, difficulty_csv_path):
    difficulty_df = pd.read_csv(difficulty_csv_path)

    n_plots = len(dfs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots), sharex=True)
    fig.suptitle(f"EmotiBit Data: {file_slug}", fontsize=16)

    for i, data in enumerate(dfs):
        df = data['df']
        data_type = data['type']
        

        # Plot the data
        axes[i].plot(df['LocalTimestamp'], df[data_type])
        axes[i].set_ylabel(data_type)
        axes[i].set_title(f"{data_type} over time")
        axes[i].grid(True)

        # Add difficulty overlay
        for _, row in difficulty_df.iterrows():
            start = row['Start Time (Unix)']
            end = row['End Time (Unix)']
            color = 'green' if row['Difficulty'] == 'Easy' else 'red'
            axes[i].axvspan(start, end, alpha=0.3, color=color)


 

    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Use tight layout to prevent overlapping
    plt.tight_layout()
    plt.show()
    

plot_emotibit_data(dfs, file_slug, difficulty_csv_path="./paragraph_timestamps.csv")

    
    
    
    
    