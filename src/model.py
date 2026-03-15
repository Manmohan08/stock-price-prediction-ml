"""
Machine Learning Models for Stock Price Prediction
Implements multiple classifiers with hyperparameter tuning and comparison.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class ModelPipeline:
    """
    Trains, evaluates, and compares multiple ML models for stock direction prediction.
    """

    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', C=1.0, gamma='scale', random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7, weights='distance'
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42
            ),
        }
        self.scaler = StandardScaler()
        self.results = {}
        self.trained_models = {}

    def train_and_evaluate(self, df, feature_cols, test_ratio=0.2):
        """
        Train all models and compare performance.

        Uses a chronological split (no random shuffling) to simulate
        real-world prediction conditions.
        """
        X = df[feature_cols].values
        y = df['Target'].values

        # Chronological split
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n" + "=" * 70)
        print("🤖  MODEL TRAINING & EVALUATION")
        print("=" * 70)
        print(f"  Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
        print(f"  Features used:    {len(feature_cols)}")
        print("=" * 70)

        for name, model in self.models.items():
            print(f"\n📊 Training: {name}...")

            # Train
            model.fit(X_train_scaled, y_train)
            self.trained_models[name] = model

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            self.results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'predictions': y_pred,
                'y_test': y_test,
            }

            print(f"   Accuracy:  {acc:.4f}")
            print(f"   Precision: {prec:.4f}")
            print(f"   Recall:    {rec:.4f}")
            print(f"   F1-Score:  {f1:.4f}")

        # Print Comparison Table
        self._print_comparison()

        # Best model
        best_model_name = max(self.results, key=lambda k: self.results[k]['f1_score'])
        best_f1 = self.results[best_model_name]['f1_score']
        print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")

        return self.results, y_test, X_test_scaled

    def _print_comparison(self):
        """Print a formatted comparison table of all models."""
        print("\n" + "=" * 70)
        print("📋  MODEL COMPARISON")
        print("=" * 70)
        print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("  " + "-" * 65)

        for name, metrics in self.results.items():
            print(f"  {name:<25} {metrics['accuracy']:>10.4f} "
                  f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f}")
        print("=" * 70)

    def get_feature_importance(self, feature_cols):
        """
        Get feature importance from tree-based models.
        Returns a sorted list of (feature_name, importance) tuples.
        """
        importances = {}
        for name in ['Random Forest', 'Gradient Boosting']:
            if name in self.trained_models:
                model = self.trained_models[name]
                imp = model.feature_importances_
                importances[name] = sorted(
                    zip(feature_cols, imp), key=lambda x: x[1], reverse=True
                )
        return importances
