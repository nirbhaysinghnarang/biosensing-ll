import csv
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt

def read_emotibit(folder_location, slug):
    data_pts = ["EA", "EL", "PI", "PR", "PG", "T1", "SF", "HR"]
    dfs = []

    for pt in data_pts:
        try:
            df = pd.read_csv(f"{folder_location}/{slug}_{pt}.csv")
            dfs.append({'type': pt, 'df': df})
        except FileNotFoundError:
            print(f"Could not locate file {pt}")
            
        
    return dfs

    
def plot_emotibit_data(dfs, difficulty_csv_path):
    difficulty_df = pd.read_csv(difficulty_csv_path)

    n_plots = len(dfs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots), sharex=True)

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