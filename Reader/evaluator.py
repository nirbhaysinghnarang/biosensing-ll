from sklearn.calibration import LabelEncoder, cross_val_predict
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np


class Evaluator:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        
    def run(self):
        # self._evaluate_logistic_regression()
        # self._evaluate_random_forest()
        # self._evaluate_xgboost_hinge()
        self._evaluate_xgboost_log_split()
        
        
        
    def _evaluate_decision_tree(self):
        dt = DecisionTreeClassifier(max_depth=2)
        dt_scores = cross_val_score(dt, self.X, self.y, cv=LeaveOneOut())
        print(f"Decision Tree Accuracy: {np.mean(dt_scores):.2f}")

    def _evaluate_logistic_regression(self):
        loo = LeaveOneOut()
        log_model = LogisticRegression(random_state=42)
        log_predictions = []
        for train_index, test_index in loo.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            log_model.fit(X_train, y_train)
            log_predictions.append(log_model.predict(X_test)[0])
        log_accuracy = accuracy_score(self.y, log_predictions)
        print(f"LOOCV Accuracy with Logistic Regression: {log_accuracy:.2f}")
        print("\nClassification Report for Logistic Regression:")
        print(classification_report(self.y, log_predictions, target_names=['Easy', 'Hard']))

    def _evaluate_random_forest(self):
        loo = LeaveOneOut()
        rf_model = RandomForestClassifier(random_state=42)
        rf_predictions = []
        for train_index, test_index in loo.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            rf_model.fit(X_train, y_train)
            pred = rf_model.predict(X_test)[0]
            rf_predictions.append(pred)
        rf_accuracy = accuracy_score(self.y, rf_predictions)
        print(f"\nLOOCV Accuracy with RF: {rf_accuracy:.2f}")
        print("\nClassification Report with RF:")
        print(classification_report(self.y, rf_predictions, target_names=['Easy', 'Hard']))

    def _evaluate_xgboost_hinge(self):
        loo = LeaveOneOut()
        xgb_model = XGBClassifier(
            eval_metric='logloss', 
            verbosity=1,
            objective="binary:hinge",
            random_state=42
        )
        xgb_predictions = []
        for train_index, test_index in loo.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            xgb_model.fit(X_train, y_train)
            pred = xgb_model.predict(X_test)[0]
            xgb_predictions.append(pred)
        xgb_accuracy = accuracy_score(self.y, xgb_predictions)
        print(f"\nLOOCV Accuracy with XGBoost Hinge: {xgb_accuracy:.2f}")
        print("\nClassification Report with XGBoost Hinge:")
        print(classification_report(self.y, xgb_predictions, target_names=['Easy', 'Hard']))
        
    def _evaluate_xgboost_log(self):
        # print("Features:")
        # print(self.X)
        print("\nLabels:")
        print(self.y)

        # Create a LabelEncoder to transform [-1, 1] to [0, 1]
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)

        loo = LeaveOneOut()
        xgb_model = XGBClassifier(
            eval_metric='logloss',
            verbosity=1,
            objective="binary:logistic",
            random_state=42,
            base_score=0.5,
            max_depth=2,
            n_estimators=10,
            learning_rate=0.1,
            min_child_weight=2
        )

        # XGBoost evaluation
        xgb_predictions = []
        xgb_probabilities = []
        for train_index, test_index in loo.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]
            xgb_model.fit(X_train, y_train)
            prob = xgb_model.predict_proba(X_test)[0]
            xgb_probabilities.append(prob[1])  # Probability of positive class
            pred = le.inverse_transform([1 if prob[1] > 0.5 else 0])[0]
            xgb_predictions.append(pred)
            print(f"Prediction: {pred}, Probability: {prob}")
        
        xgb_accuracy = accuracy_score(self.y, xgb_predictions)
        print(f"\nLOOCV Accuracy with XGBoost: {xgb_accuracy:.2f}")
        print("\nClassification Report with XGBoost:")
        print(classification_report(self.y, xgb_predictions, target_names=['Easy', 'Hard']))


    def _evaluate_xgboost_log_split(self):
        print("\nLabels:")
        print(self.y)

        # Create a LabelEncoder to transform [-1, 1] to [0, 1]
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)

        # Splitting the data into training and testing sets with an 80/20 split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_encoded, test_size=0.2, random_state=42
        )

        xgb_model = XGBClassifier(
            eval_metric='logloss',
            verbosity=1,
            objective="binary:logistic",
            random_state=42,
            base_score=0.5,
            max_depth=2,
            n_estimators=10,
            learning_rate=0.1,
            min_child_weight=2
        )

        # Fitting the model and making predictions
        xgb_model.fit(X_train, y_train)
        predictions = xgb_model.predict(X_test)
        probabilities = xgb_model.predict_proba(X_test)[:, 1]  # Probability of positive class

        # Transforming predictions back to original class labels
        predictions_transformed = le.inverse_transform(predictions)

        # Printing out each prediction with its probability
        for pred, prob in zip(predictions_transformed, probabilities):
            print(f"Prediction: {pred}, Probability: {prob:.2f}")

        # Evaluating the model
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nAccuracy with XGBoost: {accuracy:.2f}")
        print("\nClassification Report with XGBoost:")
        print(classification_report(y_test, predictions, target_names=['Easy', 'Hard']))