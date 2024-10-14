from sklearn.calibration import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

class Evaluator:
    """
    This module runs a bunch of classification models given a feature vector X 
    and a result vector y.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.random_state = 42  # Set a fixed random state for reproducibility
        
    def run(self):
        results = {}
        results['logistic_regression'] = self._evaluate_logistic_regression_split()
        results['random_forest'] = self._evaluate_random_forest()
        results['xgboost_hinge'] = self._evaluate_xgboost_hinge()
        results['xgboost_log'] = self._evaluate_xgboost_log_split()
        results['lstm'] = self._evaluate_lstm()
        return results
        
    def _evaluate_lstm(self):
        print("\nEvaluating LSTM model...")
        X_reshaped = np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1]))
        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, self.y, test_size=0.2, random_state=self.random_state)
        
        tf.random.set_seed(self.random_state)
        model = Sequential([
            Input(shape=(1, self.X.shape[1])),
            LSTM(50),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM Test Accuracy: {accuracy:.2f}")
        return accuracy

    def _evaluate_logistic_regression_split(self):
        print("\nEvaluating Logistic Regression...")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        log_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        log_model.fit(X_train, y_train)
        predictions = log_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Logistic Regression Accuracy: {accuracy:.2f}")
        return accuracy
        
    def _evaluate_random_forest(self):
        print("\nEvaluating Random Forest...")
        rf_model = RandomForestClassifier(random_state=self.random_state)
        rf_scores = cross_val_score(rf_model, self.X, self.y, cv=5)
        accuracy = np.mean(rf_scores)
        print(f"Random Forest Accuracy: {accuracy:.2f}")
        return accuracy

    def _evaluate_xgboost_hinge(self):
        print("\nEvaluating XGBoost with Hinge Loss...")
        xgb_model = XGBClassifier(
            eval_metric='logloss', 
            verbosity=0,
            objective="binary:hinge",
            random_state=self.random_state
        )
        xgb_scores = cross_val_score(xgb_model, self.X, self.y, cv=5)
        accuracy = np.mean(xgb_scores)
        print(f"XGBoost (Hinge) Accuracy: {accuracy:.2f}")
        return accuracy
        
    def _evaluate_xgboost_log_split(self):
        print("\nEvaluating XGBoost with Logistic Loss...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        X_train, X_test, y_train, y_test = train_test_split(self.X, y_encoded, test_size=0.2, random_state=self.random_state)
        xgb_model = XGBClassifier(
            eval_metric='logloss',
            verbosity=0,
            objective="binary:logistic",
            base_score=0.5,
            max_depth=2,
            n_estimators=10,
            learning_rate=0.1,
            min_child_weight=2,
            random_state=self.random_state
        )
        xgb_model.fit(X_train, y_train)
        predictions = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"XGBoost (Logistic) Accuracy: {accuracy:.2f}")
        return accuracy