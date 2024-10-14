import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

from muse_reader import read_muse
from pupil_reader import read_pupils, read_fixations, plot_pupil_diameters_speed_and_fixations
from emotibit_reader import read_emotibit, plot_emotibit_data

from sklearn.impute import SimpleImputer

from muse_reader import read_muse, plot_muse_data
from pupil_reader import read_pupils
from emotibit_reader import read_emotibit

from evaluator import Evaluator

def read_all_data(folder_loc, muse_file, pupil_folder_path, emotibit_folder_path, emoti_slug):
    muse_data = read_muse(
        os.path.join(folder_loc, muse_file)
    )
    pupil_data = read_pupils(
        os.path.join(folder_loc, pupil_folder_path)
    )
    emotibit_data = read_emotibit(
        os.path.join(folder_loc, emotibit_folder_path),
        emoti_slug
    )
    
    
    
    return muse_data, pupil_data, emotibit_data

def calculate_psd(data, fs, freq_bands):
    f, Pxx = signal.welch(data, fs, nperseg=fs)
    psd_bands = {}
    for band, (low, high) in freq_bands.items():
        idx = np.logical_and(f >= low, f <= high)
        psd_bands[band] = np.mean(Pxx[idx])
    return psd_bands

def process_emotibit(emotibit_data, time_windows):
    features = []
    for start, end in time_windows:
        window_features = []
        for data_dict in emotibit_data:
            data_type = data_dict['type']
            df = data_dict['df']
            window_data = df[(df['LocalTimestamp'] >= start) & (df['LocalTimestamp'] <= end)]
            
            if data_type in ['EA', 'EL', 'ER', 'EDR', 'SA', 'SR', 'SF']:
                col_data = window_data[data_type]
                if len(col_data) > 0:
                    window_features.extend([
                        np.mean(col_data),
                        np.std(col_data),
                        np.max(col_data),
                        np.min(col_data)
                    ])
                    if data_type in ['ER', 'EDR']:
                        window_features.append(np.max(np.diff(col_data)))
                else:
                    window_features.extend([np.nan] * (5 if data_type in ['ER', 'EDR'] else 4))
            elif data_type == 'HR':
                hr_data = window_data[data_type]
                window_features.append(np.mean(hr_data) if len(hr_data) > 0 else np.nan)
        
        features.append(window_features)
    return np.array(features)

def process_eeg(eeg_data, time_windows, fs=256):
    freq_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100)
    }
    electrodes = ['AF7', 'AF8', 'AUX', 'TP10', 'TP9']
    
    features = []
    for start, end in time_windows:
        window_data = eeg_data[(eeg_data['timestamp'] >= start) & (eeg_data['timestamp'] <= end)]
        window_features = []
        for electrode in electrodes:
            electrode_data = window_data[window_data['electrode'] == electrode]['value']
            if len(electrode_data) > 0:
                psd_bands = calculate_psd(electrode_data, fs, freq_bands)
                window_features.extend([psd_bands[band] for band in freq_bands])
            else:
                window_features.extend([np.nan] * len(freq_bands))
        features.append(window_features)
    return np.array(features)

def process_emotibit(emotibit_data, time_windows):
    features = []
    for start, end in time_windows:
        window_features = []
        for data_dict in emotibit_data:
            data_type = data_dict['type']
            df = data_dict['df']
            window_data = df[(df['LocalTimestamp'] >= start) & (df['LocalTimestamp'] <= end)]
            
            if data_type in ['EA', 'EL', 'ER', 'EDR', 'SF']:
                col_data = window_data[data_type]
                if len(col_data) > 0:
                    window_features.extend([
                        np.mean(col_data),
                        np.std(col_data),
                        np.max(col_data),
                        np.min(col_data)
                    ])
                    if data_type in ['ER', 'EDR']:
                        window_features.append(np.max(np.diff(col_data)))
                else:
                    window_features.extend([np.nan] * (5 if data_type in ['ER', 'EDR'] else 4))
            elif data_type == 'HR':
                hr_data = window_data[data_type]
                window_features.append(np.mean(hr_data) if len(hr_data) > 0 else np.nan)
        
        features.append(window_features)
    return np.array(features)
import numpy as np

def process_eyetracking(
    pupil_data,
    time_windows,
    fixations_data=None,
    blinks_data=None
    ):
    features = []
    OFFSET = pupil_data["offset"]
    pupil_data = pupil_data["data"]
    
    SACCADE_VELOCITY_THRESHOLD = 30 

    # Pre-process blinks data to pair onset and offset events
    blink_pairs = []
    if blinks_data is not None:
        onset = None
        for blink in blinks_data:
            if blink['type'] == 'onset':
                onset = blink
            elif blink['type'] == 'offset' and onset is not None:
                blink_pairs.append((onset['timestamp'], blink['timestamp']))
                onset = None

    for start, end in time_windows:
        window_data = [datum for datum in pupil_data if (datum['timestamp'] + OFFSET >= start) and (datum['timestamp'] + OFFSET <= end)]
        
        if window_data:
            timestamps = np.array([datum['timestamp'] for datum in window_data])
            diameters = np.array([datum['diameter'] for datum in window_data])
            velocities = np.array([datum['velocity'] for datum in window_data])

            avg_dilation = np.mean(diameters)
            avg_speed = np.mean(velocities)

            # Detect saccades
            saccade_indices = np.where(velocities > SACCADE_VELOCITY_THRESHOLD)[0]
            saccade_count = len(saccade_indices)

            # Calculate saccade features
            if saccade_count > 0:
                avg_saccade_velocity = np.mean(velocities[saccade_indices])
                max_saccade_velocity = np.max(velocities[saccade_indices])
            else:
                avg_saccade_velocity = max_saccade_velocity = np.nan
        else:
            avg_dilation = avg_speed = saccade_count = avg_saccade_velocity = max_saccade_velocity = np.nan

        # Process fixations
        fixation_count = 0
        fixation_durations = []
        if fixations_data is not None:
            for fixation in fixations_data:
                fixation_start = fixation['timestamp'] + OFFSET
                fixation_end = fixation_start + fixation['duration'] / 1000.0  # Convert duration to seconds
                if (fixation_start >= start and fixation_start <= end) or (fixation_end >= start and fixation_end <= end):
                    fixation_count += 1
                    fixation_durations.append(fixation['duration'] / 1000.0)  # Store duration in seconds

            avg_fixation_duration = np.mean(fixation_durations) if fixation_durations else np.nan
            max_fixation_duration = np.max(fixation_durations) if fixation_durations else np.nan
            min_fixation_duration = np.min(fixation_durations) if fixation_durations else np.nan
        else:
            avg_fixation_duration = max_fixation_duration = min_fixation_duration = np.nan

        # Process blinks
        blink_count = 0
        blink_durations = []
        for blink_start, blink_end in blink_pairs:
            blink_start += OFFSET
            blink_end += OFFSET
            if (blink_start >= start and blink_start <= end) or (blink_end >= start and blink_end <= end):
                blink_count += 1
                blink_duration = blink_end - blink_start
                blink_durations.append(blink_duration)

        avg_blink_duration = np.mean(blink_durations) if blink_durations else np.nan
        max_blink_duration = np.max(blink_durations) if blink_durations else np.nan

        features.append([
            fixation_count, 
            avg_dilation, 
            avg_speed,
            avg_fixation_duration, 
            max_fixation_duration,
            min_fixation_duration,
            saccade_count, 
            avg_saccade_velocity,
            max_saccade_velocity,
            blink_count,
            avg_blink_duration,
            max_blink_duration
        ])
    return np.array(features)

def read_time_windows(file_path):
    df = pd.read_csv(file_path)
    return list(zip(df['Start Time (Unix)'], df['End Time (Unix)'])), df['Difficulty']

def main(folder_loc, muse_file, pupil_folder_path, emotibit_folder_path, emoti_slug):
    

    muse_data, pupil_data, emotibit_data = read_all_data(folder_loc, muse_file, pupil_folder_path, emotibit_folder_path, emoti_slug=emoti_slug)
    time_windows, difficulties = read_time_windows(os.path.join(folder_loc, 'paragraph_timestamps.csv'))
    fixations_data = read_fixations(folder_location=os.path.join(folder_loc, "000"))["fixations"]
    eeg_features = process_eeg(muse_data, time_windows)
    emotibit_features = process_emotibit(emotibit_data, time_windows)
    eyetracking_features = process_eyetracking(pupil_data, time_windows, fixations_data=fixations_data)
    all_features = np.hstack((eeg_features, emotibit_features, eyetracking_features))
    
    nan_rows = np.isnan(all_features).any(axis=1)
    if np.any(nan_rows):
        print(f"Warning: {np.sum(nan_rows)} rows contain NaN values.")
        print("NaN locations:")
        for i, row in enumerate(nan_rows):
            if row:
                print(f"Row {i}: {np.isnan(all_features[i])}")
    
    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')
    all_features_imputed = imputer.fit_transform(all_features)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_features_imputed)
    
    # Plot PCA results
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=['red' if d == 'Hard' else 'blue' for d in difficulties])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of EEG, EmotiBit, and Eyetracking Features')
    
    for i, (x, y) in enumerate(pca_result):
        plt.annotate(f'{difficulties[i]} {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Hard'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Easy')],
               title='Difficulty')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_loc, 'pca_result.png'))
    plt.close()
    
    print(f"PCA plot saved as 'pca_result.png' in {folder_loc}")
    X = all_features_imputed
    y = np.array([1 if d == 'Hard' else 0 for d in difficulties])
    
    print("Unique values in 'difficulties':", np.unique(difficulties))
    print("Unique values in encoded y:", np.unique(y))
    print("Value counts in y:", np.bincount(y))
    
    
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    print("Class weights:", class_weights)
        
    eval = Evaluator(X=X, y=y)
    eval.run()

 

if __name__ == "__main__":
    folder_loc = "./Run3"
    muse_file = "muse_data.csv"
    pupil_folder_path = "000"
 
    emotibit_folder_path = "emotibit"
    emoti_slug = "2024-09-19_17-35-40-922208"
    #read_all_data(folder_loc=folder_loc, muse_file=muse_file, pupil_folder_path=pupil_folder_path, emotibit_folder_path=emotibit_folder_path)
    main(folder_loc, muse_file, pupil_folder_path, emotibit_folder_path, emoti_slug)