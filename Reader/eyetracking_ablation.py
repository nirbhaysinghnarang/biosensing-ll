from itertools import combinations
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

from pupil_reader import read_pupils, read_fixations, read_blinks
from evaluator import Evaluator
from reader import read_time_windows

def read_all_data(input_list):
    all_pupil_data = []
    all_fixations_data = []
    all_blinks_data = []
    all_time_windows = []
    all_difficulties = []

    for folder_loc, _, pupil_folder_path, _, _ in input_list:
        pupil_data = read_pupils(os.path.join(folder_loc, pupil_folder_path))
        fixations_data = read_fixations(folder_location=os.path.join(folder_loc, pupil_folder_path))["fixations"]
        blinks_data = read_blinks(folder_location=os.path.join(folder_loc, pupil_folder_path))["blinks"]
        time_windows, difficulties = read_time_windows(os.path.join(folder_loc, 'paragraph_timestamps.csv'))

        all_pupil_data.append(pupil_data)
        all_fixations_data.append(fixations_data)
        all_blinks_data.append(blinks_data)
        all_time_windows.append(time_windows)
        all_difficulties.extend(difficulties)
    
    return all_pupil_data, all_fixations_data, all_blinks_data, all_time_windows, all_difficulties

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

def process_collated_data(all_pupil_data, all_time_windows, all_fixations_data, all_blinks_data):
    all_eyetracking_features = []
    print(f"Number of datasets: {len(all_pupil_data)}")
    print(f"Total number of time windows: {len(all_time_windows)}")
    
    for i, (pupil_data, fixations_data, blinks_data, time_windows) in enumerate(zip(all_pupil_data, all_fixations_data, all_blinks_data, all_time_windows)):
        print(f"Processing dataset {i+1}")
        eyetracking_features = process_eyetracking(
            pupil_data, 
            time_windows, 
            fixations_data=fixations_data,
            blinks_data=blinks_data
        )
        all_eyetracking_features.append(eyetracking_features)

    # Combine all features
    all_eyetracking_features = np.vstack(all_eyetracking_features)

    imputer = SimpleImputer(strategy='mean')
    all_features_imputed = imputer.fit_transform(all_eyetracking_features)
    return all_features_imputed

def analyze_feature_importance(X, y, feature_names):
    # 1. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    # 2. Correlation Analysis
    corr_matrix = np.abs(np.corrcoef(X.T))
    corr_with_target = np.abs(np.corrcoef(X.T, y)[-1][:-1])

    # 3. Recursive Feature Elimination
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=1)
    rfe.fit(X, y)
    rfe_ranking = rfe.ranking_

    # 4. PCA Loadings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    pca_loadings = np.abs(pca.components_[0])

    # Compile results
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'RF Importance': rf_importance,
        'Correlation': corr_with_target,
        'RFE Ranking': rfe_ranking,
        'PCA Loading': pca_loadings
    })

    feature_importance = feature_importance.sort_values('RF Importance', ascending=False)

    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance['Feature'], feature_importance['RF Importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance['Feature'], feature_importance['Correlation'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Correlation with Target')
    plt.tight_layout()
    plt.savefig('feature_importance_correlation.png')
    plt.close()

    return feature_importance

# Modify your run_eyetracking_analysis function to include this analysis

def improved_feature_selection(X, y, feature_names, output_folder):
    results = {}
    
    # Prepare the data
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 1. Recursive Feature Elimination
    rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=1)
    rfe.fit(X_scaled, y)
    rfe_ranking = rfe.ranking_
    results['RFE_ranking'] = dict(zip(feature_names, rfe_ranking))
    plot_ranking(results['RFE_ranking'], 'RFE Ranking', output_folder, ascending=True)
    
    # 2. Lasso Feature Selection
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_coef = np.abs(lasso.coef_)
    results['Lasso_coef'] = dict(zip(feature_names, lasso_coef))
    plot_ranking(results['Lasso_coef'], 'Lasso Coefficients', output_folder)
    
    # 3. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    rf_importance = rf.feature_importances_
    results['RF_importance'] = dict(zip(feature_names, rf_importance))
    plot_ranking(results['RF_importance'], 'Random Forest Feature Importance', output_folder)
    
    # 4. Correlation Analysis
    corr_matrix = np.abs(np.corrcoef(X_scaled.T))
    corr_with_target = np.abs(np.corrcoef(X_scaled.T, y)[-1][:-1])
    results['Correlation_with_target'] = dict(zip(feature_names, corr_with_target))
    plot_ranking(results['Correlation_with_target'], 'Correlation with Target', output_folder)
    
    # 5. PCA
    pca = PCA()
    pca.fit(X_scaled)
    pca_loadings = np.abs(pca.components_[0])
    results['PCA_loadings'] = dict(zip(feature_names, pca_loadings))
    plot_ranking(results['PCA_loadings'], 'PCA Loadings', output_folder)
    
    return results

def plot_ranking(ranking_dict, title, output_folder, ascending=False):
    plt.figure(figsize=(12, 6))
    sorted_items = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=not ascending)
    features, values = zip(*sorted_items)
    
    plt.bar(features, values)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance/Ranking')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

def evaluate_top_features(X, y, feature_names, output_folder, top_n=5):
    results = improved_feature_selection(X, y, feature_names, output_folder=output_folder)
    
    # Aggregate rankings
    aggregate_ranking = {}
    for feature in feature_names:
        aggregate_ranking[feature] = (
            results['RFE_ranking'][feature] +
            list(results['Lasso_coef'].keys()).index(feature) +
            list(results['RF_importance'].keys()).index(feature) +
            list(results['Correlation_with_target'].keys()).index(feature) +
            list(results['PCA_loadings'].keys()).index(feature)
        )
    
    # Select top features
    top_features = sorted(aggregate_ranking, key=aggregate_ranking.get)[:top_n]
    
    # Evaluate with top features
    X_top = X[:, [feature_names.index(f) for f in top_features]]
    evaluator = Evaluator(X=X_top, y=y)
    evaluation_results = evaluator.run()
    
    return top_features, evaluation_results

def run_eyetracking_analysis(input_list, output_folder):
    print("Reading all data...")
    all_pupil_data, all_fixations_data, all_blinks_data, all_time_windows, all_difficulties = read_all_data(input_list)

    print("Processing data...")
    X_eyetracking = process_collated_data(all_pupil_data, all_time_windows, all_fixations_data, all_blinks_data)

    print("Preparing target variable...")
    y = np.array([1 if d == 'Hard' else 0 for d in all_difficulties])
    
    feature_names = [
        'fixation_count', 'avg_dilation', 'avg_speed', 'avg_fixation_duration',
        'max_fixation_duration', 'min_fixation_duration', 'saccade_count',
        'avg_saccade_velocity', 'max_saccade_velocity', 'blink_count',
        'avg_blink_duration', 'max_blink_duration'
    ]
    
    print("Performing feature selection and evaluation...")
    top_features, evaluation_results = evaluate_top_features(X_eyetracking, y, feature_names, output_folder)
    
    print("\nTop Features:")
    print(", ".join(top_features))
    
    print("\nEvaluation Results:")
    for model, score in evaluation_results.items():
        print(f"  {model}: {score:.4f}")

    print("\nAnalysis completed")



        
def visualize_pca(X, difficulties, output_folder):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=['red' if d == 'Hard' else 'blue' for d in difficulties])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Eyetracking Features')
    
    for i, (x, y) in enumerate(pca_result):
        plt.annotate(f'{difficulties[i]} {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Hard'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Easy')],
               title='Difficulty')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_result_eyetracking.png'))
    plt.close()
    
    print(f"PCA plot saved as 'pca_result_eyetracking.png' in {output_folder}")

if __name__ == "__main__":
    input_list = [
        ("./Run5", "muse_data.csv", "000", "emotibit", "2024-10-02_12-36-47-751303"),
        ("./Run6", "muse_data.csv", "000", "emotibit","2024-10-11_17-47-15-117561"),
        ("./Run7", "muse_data.csv", "000", "emotibit", "2024-10-08_17-49-37-608268"),
        ("./Run8", "muse_data.csv", "000", "emotibit", "2024-10-04_11-46-56-629614")        
    ]
    output_folder = "./EyetrackingResults"
    os.makedirs(output_folder, exist_ok=True)
    
    run_eyetracking_analysis(input_list, output_folder)