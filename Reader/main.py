import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os

# Import your custom modules here
from muse_reader import read_muse
from pupil_reader import read_pupils, read_fixations
from emotibit_reader import read_emotibit
from evaluator import Evaluator

from reader import process_eeg, process_emotibit, process_eyetracking, read_time_windows, calculate_psd
# Include all the processing functions from your original script here
# (calculate_psd, process_eeg, process_emotibit, process_eyetracking, read_time_windows)

def read_all_data(input_list):
    all_muse_data = []
    all_pupil_data = []
    all_emotibit_data = []
    all_time_windows = []
    all_difficulties = []
    all_fixations_data = []

    for folder_loc, muse_file, pupil_folder_path, emotibit_folder_path, emoti_slug in input_list:
        muse_data = read_muse(os.path.join(folder_loc, muse_file))
        pupil_data = read_pupils(os.path.join(folder_loc, pupil_folder_path))
        emotibit_data = read_emotibit(os.path.join(folder_loc, emotibit_folder_path), emoti_slug)
        time_windows, difficulties = read_time_windows(os.path.join(folder_loc, 'paragraph_timestamps.csv'))
        fixations_data = read_fixations(folder_location=os.path.join(folder_loc, pupil_folder_path))["fixations"]

        all_muse_data.append(muse_data)
        all_pupil_data.append(pupil_data)
        all_emotibit_data.append(emotibit_data)
        all_time_windows.append(time_windows)
        all_difficulties.extend(difficulties)
        all_fixations_data.append(fixations_data)

    return all_muse_data, all_pupil_data, all_emotibit_data, all_time_windows, all_difficulties, all_fixations_data

def process_collated_data(all_muse_data, all_pupil_data, all_emotibit_data, all_time_windows, all_fixations_data):
    all_eeg_features = []
    all_emotibit_features = []
    all_eyetracking_features = []
    print(f"Number of datasets: {len(all_muse_data)}")
    print(f"Total number of time windows: {len(all_time_windows)}")
    
    
    for i, (muse_data, pupil_data, emotibit_data, fixations_data, time_windows) in enumerate(zip(all_muse_data, all_pupil_data, all_emotibit_data, all_fixations_data, all_time_windows)):
        
        print(f"Processing dataset {i+1}")
        eeg_features = process_eeg(muse_data, time_windows)
        print(len(eeg_features))
        emotibit_features = process_emotibit(emotibit_data, time_windows)
        print(len(emotibit_features))
        eyetracking_features = process_eyetracking(pupil_data, time_windows, fixations_data=fixations_data)
        print(len(eyetracking_features))

        all_eeg_features.append(eeg_features)
        all_emotibit_features.append(emotibit_features)
        all_eyetracking_features.append(eyetracking_features)

    # Combine all features
    all_eeg_features = np.vstack(all_eeg_features)
    all_emotibit_features = np.vstack(all_emotibit_features)
    all_eyetracking_features = np.vstack(all_eyetracking_features)

    all_features = np.hstack((all_eeg_features, all_emotibit_features, all_eyetracking_features))
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    all_features_imputed = imputer.fit_transform(all_features)

    return all_features_imputed

def visualize_pca(X, difficulties, output_folder):
    print(len(difficulties), len(X), X.shape)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=['red' if d == 'Hard' else 'blue' for d in difficulties])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Collated EEG, EmotiBit, and Eyetracking Features')
    
    for i, (x, y) in enumerate(pca_result):
        plt.annotate(f'{difficulties[i]} {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Hard'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Easy')],
               title='Difficulty')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_result_collated.png'))
    plt.close()
    
    print(f"PCA plot saved as 'pca_result_collated.png' in {output_folder}")

def run_collated_analysis(input_list, output_folder):
    print("Reading all data...")
    all_muse_data, all_pupil_data, all_emotibit_data, all_time_windows, all_difficulties, all_fixations_data = read_all_data(input_list)

    print("Processing collated data...")
    X = process_collated_data(
        all_muse_data, 
        all_pupil_data, 
        all_emotibit_data, 
        all_time_windows,
        all_fixations_data
    )

    print("Preparing target variable...")
    y = np.array([1 if d == 'Hard' else 0 for d in all_difficulties])
    
    print(y, len(y))

    print("Visualizing PCA...")
    visualize_pca(X, all_difficulties, output_folder)

    print("Running models...")
    evaluator = Evaluator(X=X, y=y)
    evaluator.run()

    print("Analysis completed for all collated data")

if __name__ == "__main__":
    input_list = [
        ("./Run3", "muse_data.csv", "000", "emotibit", "2024-09-19_17-35-40-922208"),
        ("./Run4", "muse_data.csv", "000", "emotibit","2024-09-19_21-38-59-706080"),
        
        
        
        # Add more input tuples here for additional runs
    ]
    output_folder = "./CollatedResults"
    os.makedirs(output_folder, exist_ok=True)
    
    run_collated_analysis(input_list, output_folder)