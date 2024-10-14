from itertools import combinations
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os

from muse_reader import read_muse
from pupil_reader import read_pupils, read_fixations, read_blinks
from emotibit_reader import read_emotibit
from evaluator import Evaluator
from reader import process_eeg, process_emotibit, process_eyetracking, read_time_windows, calculate_psd


def read_all_data(input_list):
    all_muse_data = []
    all_pupil_data = []
    all_emotibit_data = []
    all_time_windows = []
    all_difficulties = []
    all_fixations_data = []
    all_blink_data = []
    for folder_loc, muse_file, pupil_folder_path, emotibit_folder_path, emoti_slug in input_list:
        muse_data = read_muse(os.path.join(folder_loc, muse_file))
        pupil_data = read_pupils(os.path.join(folder_loc, pupil_folder_path))
        emotibit_data = read_emotibit(os.path.join(folder_loc, emotibit_folder_path), emoti_slug)
        time_windows, difficulties = read_time_windows(os.path.join(folder_loc, 'paragraph_timestamps.csv'))
        fixations_data = read_fixations(folder_location=os.path.join(folder_loc, pupil_folder_path))["fixations"]
        blink_data = read_blinks(folder_location=os.path.join(folder_loc, pupil_folder_path))["blinks"]
        all_muse_data.append(muse_data)
        all_pupil_data.append(pupil_data)
        all_emotibit_data.append(emotibit_data)
        all_time_windows.append(time_windows)
        all_difficulties.extend(difficulties)
        all_fixations_data.append(fixations_data)
        all_blink_data.append(blink_data)

    return all_muse_data, all_pupil_data, all_emotibit_data, all_time_windows, all_difficulties, all_fixations_data, all_blink_data

def process_collated_data(all_muse_data, all_pupil_data, all_emotibit_data, all_time_windows, all_fixations_data):
    all_eeg_features = []
    all_emotibit_features = []
    all_eyetracking_features = []
    print(f"Number of datasets: {len(all_muse_data)}")
    print(f"Total number of time windows: {len(all_time_windows)}")
    
    
    for i, (muse_data, pupil_data, emotibit_data, fixations_data, time_windows) in enumerate(zip(all_muse_data, all_pupil_data, all_emotibit_data, all_fixations_data, all_time_windows)):
        
        print(f"Processing dataset {i+1}")
        eeg_features = process_eeg(muse_data, time_windows)
        emotibit_features = process_emotibit(emotibit_data, time_windows)
        eyetracking_features = process_eyetracking(pupil_data, time_windows, fixations_data=fixations_data)

        all_eeg_features.append(eeg_features)
        all_emotibit_features.append(emotibit_features)
        all_eyetracking_features.append(eyetracking_features)

    # Combine all features
    all_eeg_features = np.vstack(all_eeg_features)
    all_emotibit_features = np.vstack(all_emotibit_features)
    all_eyetracking_features = np.vstack(all_eyetracking_features)

    all_features = np.hstack((all_eeg_features, all_emotibit_features, all_eyetracking_features))
    imputer = SimpleImputer(strategy='mean')
    all_features_imputed = imputer.fit_transform(all_features)
    return all_features_imputed

def visualize_pca(X, difficulties, output_folder):
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

def run_ablation_analysis(input_list, output_folder):
    print("Reading all data...")
    all_muse_data, all_pupil_data, all_emotibit_data, all_time_windows, all_difficulties, all_fixations_data, all_blink_data = read_all_data(input_list)

    print("Processing data...")
    all_eeg_features = []
    all_emotibit_features = []
    all_eyetracking_features = []
    
    for i, (muse_data, pupil_data, emotibit_data, fixations_data, time_windows, blink_data) in enumerate(zip(all_muse_data, all_pupil_data, all_emotibit_data, all_fixations_data, all_time_windows, all_blink_data)):
        print(f"Processing dataset {i+1}")
        eeg_features = process_eeg(muse_data, time_windows)
        emotibit_features = process_emotibit(emotibit_data, time_windows)
        eyetracking_features = process_eyetracking(pupil_data, time_windows, fixations_data=fixations_data, blinks_data=blink_data)

        all_eeg_features.append(eeg_features)
        all_emotibit_features.append(emotibit_features)
        all_eyetracking_features.append(eyetracking_features)

    # Combine all features
    X_eeg = np.vstack(all_eeg_features)
    X_emotibit = np.vstack(all_emotibit_features)
    X_eyetracking = np.vstack(all_eyetracking_features)

    print("Preparing target variable...")
    y = np.array([1 if d == 'Hard' else 0 for d in all_difficulties])
    
    print("Performing ablation testing...")
    ablation_results = perform_ablation_testing(X_eeg, X_emotibit, X_eyetracking, y)
    
    print("Visualizing ablation results...")
    visualize_ablation_results(ablation_results, output_folder)
    
    print("Printing ablation results...")
    display_ablation_results(ablation_results)
    # Identify best combination
    best_combo = max(ablation_results, key=lambda x: sum(ablation_results[x].values())/len(ablation_results[x]))
    print(f"\nBest combination: {best_combo}")
    print("Analysis completed")
        
def perform_ablation_testing(X_eeg, X_emotibit, X_eyetracking, y):  
    feature_groups = {
        'EEG': X_eeg,
        'EmotiBit': X_emotibit,
        'Eyetracking': X_eyetracking
    }
    
    results = {}
    
    # Create all possible combinations of feature groups
    for r in range(1, len(feature_groups) + 1):
        for combo in combinations(feature_groups.keys(), r):
            X_combined = np.hstack([feature_groups[group] for group in combo])
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X_combined)            
            evaluator = Evaluator(X=X_imputed, y=y)
            evaluation_results = evaluator.run()            
            combo_name = ' + '.join(combo)
            results[combo_name] = evaluation_results
    
    return results

def display_ablation_results(results):
    models = list(next(iter(results.values())).keys())
    combos = list(results.keys())
    
    print("Ablation Testing Results:")
    print("--------------------------")
    
    # Find the maximum length of model names and combo names for formatting
    max_model_length = max(len(model.replace("_", " ").title()) for model in models)
    max_combo_length = max(len(combo) for combo in combos)
    
    # Print header
    header = f"{'Feature Combination':<{max_combo_length}} | " + " | ".join(f"{model.replace('_', ' ').title():<{max_model_length}}" for model in models)
    print(header)
    print("-" * len(header))
    
    # Print results for each combination
    for combo in combos:
        row = f"{combo:<{max_combo_length}} | "
        row += " | ".join(f"{results[combo][model]:.4f}".center(max_model_length) for model in models)
        print(row)
    
    print("\nBest Performing Combinations:")
    print("------------------------------")
    for model in models:
        best_combo = max(combos, key=lambda x: results[x][model])
        print(f"{model.replace('_', ' ').title():<{max_model_length}}: {best_combo} (Accuracy: {results[best_combo][model]:.4f})")

def visualize_ablation_results(results, output_folder):
    models = list(next(iter(results.values())).keys())
    combos = list(results.keys())
    
    for model in models:
        scores = [results[combo][model] for combo in combos]
        
        plt.figure(figsize=(12, 6))
        plt.bar(combos, scores)
        plt.xlabel('Feature Combinations')
        plt.ylabel('Accuracy')
        plt.title(f'Ablation Testing Results - {model.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'ablation_results_{model}.png'))
        plt.close()
    
    # Create a combined plot
    plt.figure(figsize=(15, 8))
    bar_width = 0.15
    index = np.arange(len(combos))
    
    for i, model in enumerate(models):
        scores = [results[combo][model] for combo in combos]
        plt.bar(index + i*bar_width, scores, bar_width, label=model.replace("_", " ").title())
    
    plt.xlabel('Feature Combinations')
    plt.ylabel('Accuracy')
    plt.title('Ablation Testing Results - All Models')
    plt.xticks(index + bar_width * (len(models) - 1) / 2, combos, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'ablation_results_combined.png'))
    plt.close()


if __name__ == "__main__":
    input_list = [
        # ("./Run3", "muse_data.csv", "000", "emotibit", "2024-09-19_17-35-40-922208"),
        # ("./Run4", "muse_data.csv", "000", "emotibit","2024-09-19_21-38-59-706080"),
        ("./Run5", "muse_data.csv", "000", "emotibit", "2024-10-02_12-36-47-751303"),
        ("./Run6", "muse_data.csv", "000", "emotibit","2024-10-11_17-47-15-117561"),
        ("./Run7", "muse_data.csv", "000", "emotibit", "2024-10-08_17-49-37-608268"),
        ("./Run8", "muse_data.csv", "000", "emotibit", "2024-10-04_11-46-56-629614")        
    ]
    output_folder = "./CollatedResults"
    os.makedirs(output_folder, exist_ok=True)
    
    run_ablation_analysis(input_list, output_folder)