import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm


@dataclass
class ClassificationConfig:
    """Configuration for depression classification."""
    # Data splitting
    test_size: float = 0.2  # Proportion of data for testing
    val_size: float = 0.1   # Proportion of training data for validation
    random_state: int = 42  # Random seed for reproducibility
    
    # Feature processing
    use_pca: bool = True            # Whether to use PCA
    pca_components: int = 15        # Number of PCA components
    standardize: bool = True        # Whether to standardize features
    
    # Class balancing
    balance_method: Optional[str] = 'smote'  # 'smote', 'undersample', or None
    
    # Model selection
    model_type: str = 'svm'         # 'svm', 'rf', or 'mlp'
    
    # Cross-validation
    cv_folds: int = 5               # Number of CV folds
    
    # Model hyperparameters
    # SVM
    svm_C: List[float] = None
    svm_gamma: List[str] = None
    svm_kernel: List[str] = None
    
    # Random Forest
    rf_n_estimators: List[int] = None
    rf_max_depth: List[int] = None
    
    # MLP
    mlp_hidden_layer_sizes: List[Tuple[int, ...]] = None
    mlp_alpha: List[float] = None
    mlp_learning_rate: List[str] = None
    
    def __post_init__(self):
        """Initialize default hyperparameter values if not provided."""
        # SVM defaults
        if self.svm_C is None:
            self.svm_C = [0.1, 1, 10, 100]
        if self.svm_gamma is None:
            self.svm_gamma = ['scale', 'auto']
        if self.svm_kernel is None:
            self.svm_kernel = ['rbf', 'linear']
            
        # RF defaults
        if self.rf_n_estimators is None:
            self.rf_n_estimators = [100, 200, 300]
        if self.rf_max_depth is None:
            self.rf_max_depth = [None, 10, 20, 30]
            
        # MLP defaults
        if self.mlp_hidden_layer_sizes is None:
            self.mlp_hidden_layer_sizes = [(100,), (50, 50), (100, 50), (100, 100)]
        if self.mlp_alpha is None:
            self.mlp_alpha = [0.0001, 0.001, 0.01]
        if self.mlp_learning_rate is None:
            self.mlp_learning_rate = ['constant', 'adaptive']


class DepressionClassifier:
    """Classify depression based on audio features."""

    def __init__(self, config: Optional[ClassificationConfig] = None):
        """
        Initialize classifier with configuration.
        
        Args:
            config: Classification configuration
        """
        self.config = config or ClassificationConfig()
        self.scaler = StandardScaler() if self.config.standardize else None
        self.pca = None
        self.model = None
        self.best_params = None
        self.feature_importances = None
        self.metrics = {}
    
    def preprocess_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Preprocess features before classification.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            fit: Whether to fit transformers or just transform
            
        Returns:
            Preprocessed features
        """
        # Check for NaN values
        if np.isnan(features).any():
            print(f"Warning: {np.isnan(features).sum()} NaN values found in features")
            # Replace NaN with 0
            features = np.nan_to_num(features, nan=0.0)
        
        # Check for infinite values
        if np.isinf(features).any():
            print(f"Warning: {np.isinf(features).sum()} infinite values found in features")
            # Replace inf with large value and -inf with small value
            features = np.nan_to_num(features, posinf=1e10, neginf=-1e10)
        
        # Standardize features
        if self.scaler is not None:
            if fit:
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)
        
        # Apply PCA if configured
        if self.config.use_pca:
            if fit:
                # Determine number of components (either specified or minimum of data dimensions)
                n_components = min(self.config.pca_components, features.shape[0], features.shape[1])
                self.pca = PCA(n_components=n_components, random_state=self.config.random_state)
                features = self.pca.fit_transform(features)
                
                # Report explained variance
                explained_var = np.sum(self.pca.explained_variance_ratio_)
                print(f"PCA with {n_components} components explains {explained_var:.2%} of variance")
            else:
                features = self.pca.transform(features)
        
        return features
    
    def _create_model(self) -> Any:
        """
        Create model based on configuration.
        
        Returns:
            Sklearn model instance
        """
        if self.config.model_type == 'svm':
            return SVC(probability=True, random_state=self.config.random_state)
        elif self.config.model_type == 'rf':
            return RandomForestClassifier(random_state=self.config.random_state)
        elif self.config.model_type == 'mlp':
            return MLPClassifier(random_state=self.config.random_state, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _get_param_grid(self) -> Dict:
        """
        Get parameter grid for hyperparameter tuning.
        
        Returns:
            Parameter grid dictionary
        """
        if self.config.model_type == 'svm':
            return {
                'C': self.config.svm_C,
                'gamma': self.config.svm_gamma,
                'kernel': self.config.svm_kernel
            }
        elif self.config.model_type == 'rf':
            return {
                'n_estimators': self.config.rf_n_estimators,
                'max_depth': self.config.rf_max_depth
            }
        elif self.config.model_type == 'mlp':
            return {
                'hidden_layer_sizes': self.config.mlp_hidden_layer_sizes,
                'alpha': self.config.mlp_alpha,
                'learning_rate': self.config.mlp_learning_rate
            }
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
            
    def _balance_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using the configured method.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Balanced X and y
        """
        if self.config.balance_method is None:
            return X, y
            
        # Get class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Apply balancing method
        if self.config.balance_method == 'smote':
            sampler = SMOTE(random_state=self.config.random_state)
            X_balanced, y_balanced = sampler.fit_resample(X, y)
        elif self.config.balance_method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.config.random_state)
            X_balanced, y_balanced = sampler.fit_resample(X, y)
        else:
            raise ValueError(f"Unknown balancing method: {self.config.balance_method}")
            
        # Report new distribution
        unique, counts = np.unique(y_balanced, return_counts=True)
        print(f"Balanced class distribution: {dict(zip(unique, counts))}")
        
        return X_balanced, y_balanced
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Train model and evaluate performance.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Target labels (n_samples)
            
        Returns:
            Dictionary with training results
        """
        # Split data into train and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=labels
        )
        
        # Split train into train and validation if needed
        if self.config.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=self.config.val_size / (1 - self.config.test_size),
                random_state=self.config.random_state,
                stratify=y_train_val
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = None, None
        
        # Preprocess features
        X_train = self.preprocess_features(X_train, fit=True)
        if X_val is not None:
            X_val = self.preprocess_features(X_val, fit=False)
        X_test = self.preprocess_features(X_test, fit=False)
        
        # Balance classes if configured
        X_train, y_train = self._balance_classes(X_train, y_train)
        
        # Create base model
        base_model = self._create_model()
        
        # Create cross-validator
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        # Set up grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self._get_param_grid(),
            cv=cv,
            scoring='f1',
            return_train_score=True,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        print(f"Training {self.config.model_type.upper()} model with grid search...")
        grid_search.fit(X_train, y_train)
        
        # Get best model and params
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        print(f"Best parameters: {self.best_params}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Store confusion matrix
        self.cm = confusion_matrix(y_test, y_pred)
        
        # Store feature importances if available
        if self.config.model_type == 'rf':
            self.feature_importances = self.model.feature_importances_
        
        # Print metrics
        print(f"Test set metrics:")
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Return detailed results
        return {
            'metrics': self.metrics,
            'best_params': self.best_params,
            'confusion_matrix': self.cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess features
        features = self.preprocess_features(features, fit=False)
        
        # Make predictions
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for new data.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability estimates
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess features
        features = self.preprocess_features(features, fit=False)
        
        # Get probability estimates
        return self.model.predict_proba(features)
    
    def plot_confusion_matrix(self, normalize: bool = True) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(self, 'cm'):
            raise ValueError("No confusion matrix available")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize if requested
        if normalize:
            cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = self.cm
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(title)
        ax.set_xticklabels(['Control', 'Depression'])
        ax.set_yticklabels(['Control', 'Depression'])
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Probability estimates for positive class
            
        Returns:
            Matplotlib figure
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None, 
                               top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: Names of features
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if self.feature_importances is None:
            raise ValueError("No feature importances available")
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importances))]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importances
        })
        
        # Sort by importance and get top N
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot feature importance
        sns.barplot(data=df, x='Importance', y='Feature', ax=ax)
        ax.set_title(f'Top {top_n} Features by Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        return fig
    
    def cross_validate(self, features: np.ndarray, labels: np.ndarray, 
                      n_folds: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            features: Feature matrix
            labels: Target labels
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        # Preprocess features
        features = self.preprocess_features(features, fit=True)
        
        # Balance classes if configured
        if self.config.balance_method is not None:
            print("Warning: Class balancing will be applied within each fold")
        
        # Create cross-validator
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                           random_state=self.config.random_state)
        
        # Initialize metrics lists
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Perform cross-validation
        for i, (train_idx, test_idx) in enumerate(cv.split(features, labels)):
            print(f"Fold {i+1}/{n_folds}")
            
            # Split data
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Balance classes if configured
            if self.config.balance_method is not None:
                X_train, y_train = self._balance_classes(X_train, y_train)
            
            # Create and train model
            model = self._create_model()
            model.set_params(**self.best_params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred))
            metrics['recall'].append(recall_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred))
            metrics['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
        
        # Calculate average metrics
        avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        std_metrics = {metric: np.std(values) for metric, values in metrics.items()}
        
        # Print results
        print(f"Cross-validation results ({n_folds} folds):")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
        
        return {
            'fold_metrics': metrics,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics
        }
    
    def save_model(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save the model
        """
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and preprocessors
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'config': self.config,
            'metrics': self.metrics,
            'best_params': self.best_params
        }, path)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'DepressionClassifier':
        """
        Load model from file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded classifier
        """
        import joblib
        
        # Load saved model
        saved_data = joblib.load(path)
        
        # Create classifier
        classifier = cls(saved_data['config'])
        classifier.model = saved_data['model']
        classifier.scaler = saved_data['scaler'] 
        classifier.pca = saved_data['pca']
        classifier.metrics = saved_data['metrics']
        classifier.best_params = saved_data['best_params']
        
        print(f"Model loaded from {path}")
        return classifier


# Example usage
if __name__ == "__main__":
    from src.data_loader import DepressionDataLoader
    from src.feature_extraction import FeatureExtractor, FeatureConfig
    
    # Initialize data loader
    data_loader = DepressionDataLoader("./data")
    
    # Initialize feature extractor
    feature_config = FeatureConfig(
        sample_rate=16000,
        n_mfcc=20,
        mfcc_deltas=True
    )
    feature_extractor = FeatureExtractor(feature_config)
    
    # Get reading task speakers
    reading_speakers = data_loader.group_files_by_speaker(task='reading')
    
    # Extract features for speakers
    speaker_ids = list(reading_speakers.keys())
    all_results = {}
    
    for speaker_id in tqdm(speaker_ids, desc="Processing speakers"):
        try:
            result = feature_extractor.process_speaker_data(data_loader, speaker_id, task='reading')
            all_results[speaker_id] = result
        except Exception as e:
            print(f"Error processing speaker {speaker_id}: {e}")
    
    # Extract features for classification
    features, ids = feature_extractor.extract_clusterable_features(all_results)
    
    # Get depression labels
    labels_dict = data_loader.get_labels()
    labels = np.array([labels_dict.get(id, 0) for id in ids])
    
    # Initialize classifier
    classification_config = ClassificationConfig(
        model_type='svm',
        use_pca=True,
        pca_components=10,
        balance_method='smote',
        test_size=0.2,
        cv_folds=5,
        svm_C=[0.1, 1, 10],
        svm_kernel=['rbf', 'linear']
    )
    classifier = DepressionClassifier(classification_config)
    
    # Train model
    results = classifier.train(features, labels)
    
    # Plot confusion matrix
    fig = classifier.plot_confusion_matrix()
    plt.savefig("confusion_matrix.png")
    
    # Plot ROC curve
    fig = classifier.plot_roc_curve(results['y_test'], results['y_pred_proba'])
    plt.savefig("roc_curve.png")
    
    # Cross-validate
    cv_results = classifier.cross_validate(features, labels, n_folds=5)
    
    # Save model
    classifier.save_model("./models/depression_classifier.joblib")
    
    print("Classification completed and results saved.")