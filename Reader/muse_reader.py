import csv 
import pandas as pd
import matplotlib.pyplot as plt
def read_muse(muse_fpath):
    data = pd.read_csv(muse_fpath)    
    data['timestamp'] = data['timestamp'] / 1000    
    grouped_data = data.groupby(['index', 'electrode', 'timestamp'])['value'].mean().reset_index()    
    grouped_data_sorted = grouped_data.sort_values(by=['timestamp', 'electrode'])
    print(grouped_data_sorted)
    return grouped_data_sorted

def plot_muse_data(data, difficulty_df):
    max_eeg_time = data['timestamp'].max()
    min_difficulty_time = difficulty_df['Start Time (Unix)'].min()
    offset = min_difficulty_time - max_eeg_time
    print(f"Offset: {offset} seconds")

    electrodes = data['electrode'].unique()
    fig, axes = plt.subplots(nrows=len(electrodes), ncols=1, figsize=(10, 6*len(electrodes)), sharex=True)
    fig.suptitle('Electrode Readings Over Time with Difficulty Bands')

    # Ensure axes is always a list, even with a single subplot
    if len(electrodes) == 1:
        axes = [axes]

    for i, electrode in enumerate(electrodes):
        electrode_data = data[data['electrode'] == electrode]
        axes[i].plot(electrode_data['timestamp'], electrode_data['value'])
        axes[i].set_title(f'Electrode: {electrode}')
        axes[i].set_ylabel('Value')
        
        # Only set xlabel for the bottom subplot
        if i == len(electrodes) - 1:
            axes[i].set_xlabel('Timestamp (seconds)')
        
        axes[i].tick_params(axis='x', rotation=45)

        for _, row in difficulty_df.iterrows():
            start = row['Start Time (Unix)']
            end = row['End Time (Unix)']
            color = 'green' if row['Difficulty'] == 'Easy' else 'red'
            axes[i].axvspan(start, end, alpha=0.3, color=color)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()