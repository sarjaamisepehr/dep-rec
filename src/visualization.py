import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import librosa.display

# Import local modules
from src.data_loader import DepressionDataLoader
from src.feature_extraction import FeatureExtractor
from src.classification import DepressionClassifier


class DepressionVisualizer:
    """Visualization tools for depression detection from audio features."""

    def __init__(self, save_dir: str = "./visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default style for plots
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def plot_class_distribution(self, labels: Dict[str, int], title: str = "Class Distribution") -> plt.Figure:
        """
        Plot distribution of depression vs control classes.
        
        Args:
            labels: Dictionary mapping speaker IDs to depression labels (1 for depression, 0 for control)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Count classes
        counts = pd.Series(labels).value_counts().sort_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot distribution
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=['#2196F3', '#F44336'])
        
        # Set labels
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticklabels(['Control', 'Depression'])
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(i, count + 1, str(count), ha='center')
        
        plt.tight_layout()
        return fig
    
    def plot_audio_waveform(self, audio_file: str, sample_rate: Optional[int] = None, 
                         title: Optional[str] = None) -> plt.Figure:
        """
        Plot audio waveform from file.
        
        Args:
            audio_file: Path to audio file
            sample_rate: Sample rate (if None, will be loaded from file)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Load audio
        y, sr = librosa.load(audio_file, sr=sample_rate)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Calculate time array
        duration = len(y) / sr
        time = np.linspace(0, duration, len(y))
        
        # Plot waveform
        ax.plot(time, y, color='#2196F3', alpha=0.7)
        
        # Set labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Audio Waveform - {Path(audio_file).name}')
        
        plt.tight_layout()
        return fig
    
    def plot_audio_spectrogram(self, audio_file: str, sample_rate: Optional[int] = None,
                            n_fft: int = 2048, hop_length: int = 512,
                            title: Optional[str] = None) -> plt.Figure:
        """
        Plot spectrogram from audio file.
        
        Args:
            audio_file: Path to audio file
            sample_rate: Sample rate (if None, will be loaded from file)
            n_fft: FFT window size
            hop_length: Number of samples between frames
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Load audio
        y, sr = librosa.load(audio_file, sr=sample_rate)
        
        # Compute spectrogram
        spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot spectrogram
        img = librosa.display.specshow(
            spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax
        )
        
        # Add colorbar
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Set labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Spectrogram - {Path(audio_file).name}')
        
        plt.tight_layout()
        return fig
    
    def plot_mfcc(self, audio_file: str, sample_rate: Optional[int] = None,
                n_mfcc: int = 20, hop_length: int = 512,
                title: Optional[str] = None) -> plt.Figure:
        """
        Plot MFCCs from audio file.
        
        Args:
            audio_file: Path to audio file
            sample_rate: Sample rate (if None, will be loaded from file)
            n_mfcc: Number of MFCCs to extract
            hop_length: Number of samples between frames
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Load audio
        y, sr = librosa.load(audio_file, sr=sample_rate)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot MFCCs
        img = librosa.display.specshow(
            mfccs, sr=sr, hop_length=hop_length, x_axis='time', ax=ax
        )
        
        # Add colorbar
        fig.colorbar(img, ax=ax)
        
        # Set labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'MFCCs - {Path(audio_file).name}')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_pca(self, features: np.ndarray, labels: np.ndarray, 
                        n_components: int = 2, title: str = "PCA Visualization") -> plt.Figure:
        """
        Plot PCA visualization of features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Target labels (n_samples)
            n_components: Number of PCA components
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(features)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        df['Label'] = ['Depression' if l == 1 else 'Control' for l in labels]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PCA
        sns.scatterplot(
            data=df, x='PC1', y='PC2', hue='Label', 
            palette={'Control': '#2196F3', 'Depression': '#F44336'},
            alpha=0.7, s=80, edgecolor='w', linewidth=0.5,
            ax=ax
        )
        
        # Set labels
        ax.set_title(title)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_tsne(self, features: np.ndarray, labels: np.ndarray, 
                       perplexity: int = 30, title: str = "t-SNE Visualization") -> plt.Figure:
        """
        Plot t-SNE visualization of features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Target labels (n_samples)
            perplexity: t-SNE perplexity parameter
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(features)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
        df['Label'] = ['Depression' if l == 1 else 'Control' for l in labels]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot t-SNE
        sns.scatterplot(
            data=df, x='t-SNE 1', y='t-SNE 2', hue='Label', 
            palette={'Control': '#2196F3', 'Depression': '#F44336'},
            alpha=0.7, s=80, edgecolor='w', linewidth=0.5,
            ax=ax
        )
        
        # Set labels
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, normalize: bool = True,
                           title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Plot confusion matrix
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues', 
            xticklabels=['Control', 'Depression'],
            yticklabels=['Control', 'Depression'],
            ax=ax
        )
        
        # Set labels
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                    title: str = "ROC Curve") -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Probability estimates for positive class
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        
        # Plot random classifier line
        ax.plot([0, 1], [0, 1], color='#9E9E9E', lw=2, linestyle='--', label='Random')
        
        # Set labels and limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall(self, y_true: np.ndarray, y_proba: np.ndarray,
                           title: str = "Precision-Recall Curve") -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Probability estimates for positive class
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate AUC (average precision)
        ap = np.trapz(precision, recall)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot precision-recall curve
        ax.plot(recall, precision, color='#F44336', lw=2, label=f'PR curve (AP = {ap:.3f})')
        
        # Plot random classifier line
        positive_ratio = np.sum(y_true) / len(y_true)
        ax.plot([0, 1], [positive_ratio, positive_ratio], color='#9E9E9E', lw=2, 
              linestyle='--', label=f'Random (AP = {positive_ratio:.3f})')
        
        # Set labels and limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance: np.ndarray, feature_names: List[str], 
                             top_n: int = 20, title: str = "Feature Importance") -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance: Feature importance array
            feature_names: Names of features
            top_n: Number of top features to show
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance and get top N
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot feature importance
        sns.barplot(data=df, x='Importance', y='Feature', palette='viridis', ax=ax)
        
        # Set labels
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        return fig
    
    def plot_cross_validation_results(self, cv_results: Dict) -> plt.Figure:
        """
        Plot cross-validation results.
        
        Args:
            cv_results: Cross-validation results dictionary
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics
        metrics = cv_results['fold_metrics']
        avg_metrics = cv_results['avg_metrics']
        std_metrics = cv_results['std_metrics']
        
        # Create DataFrame for plotting
        df = pd.DataFrame(metrics)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot box plots
        sns.boxplot(data=df, palette='Set3', ax=ax)
        
        # Add individual data points
        sns.swarmplot(data=df, color='0.25', size=4, ax=ax)
        
        # Set labels
        ax.set_title('Cross-Validation Metrics')
        ax.set_ylabel('Score')
        
        # Add text for average ± std
        for i, metric in enumerate(avg_metrics.keys()):
            ax.text(i, 0.1, f'Avg: {avg_metrics[metric]:.3f} ± {std_metrics[metric]:.3f}', 
                  ha='center', va='bottom', rotation=90, fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_hyperparameter_tuning(self, cv_results: pd.DataFrame, 
                               param_name: str, metric: str = 'mean_test_score',
                               title: Optional[str] = None) -> plt.Figure:
        """
        Plot hyperparameter tuning results.
        
        Args:
            cv_results: DataFrame with CV results (from GridSearchCV.cv_results_)
            param_name: Name of parameter to plot
            metric: Metric to plot (e.g., 'mean_test_score')
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Check if parameter exists
        if f'param_{param_name}' not in cv_results.columns:
            raise ValueError(f"Parameter '{param_name}' not found in CV results")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract parameter values and metric
        param_values = cv_results[f'param_{param_name}']
        scores = cv_results[metric]
        
        # Plot hyperparameter tuning
        sns.lineplot(x=param_values, y=scores, marker='o', markersize=8, ax=ax)
        
        # Set labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Effect of {param_name} on {metric}')
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, train_sizes: np.ndarray, train_scores: np.ndarray, 
                         test_scores: np.ndarray, title: str = "Learning Curve") -> plt.Figure:
        """
        Plot learning curve.
        
        Args:
            train_sizes: Array of training set sizes
            train_scores: Array of training scores (shape: (n_samples, n_folds))
            test_scores: Array of test scores (shape: (n_samples, n_folds))
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training score')
        ax.plot(train_sizes, test_mean, 'o-', color='#F44336', label='Cross-validation score')
        
        # Add shaded regions for std
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                      alpha=0.1, color='#2196F3')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                      alpha=0.1, color='#F44336')
        
        # Set labels
        ax.set_title(title)
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')
        ax.legend(loc='lower right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_confusion_matrix(self, cm: np.ndarray, normalize: bool = True,
                                      title: str = "Confusion Matrix") -> go.Figure:
        """
        Plot interactive confusion matrix using Plotly.
        
        Args:
            cm: Confusion matrix
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            text_format = '.2%'
        else:
            text_format = 'd'
        
        # Create labels
        labels = ['Control', 'Depression']
        
        # Create heatmap
        fig = px.imshow(
            cm,
            x=labels,
            y=labels,
            color_continuous_scale='Blues',
            title=title,
            labels=dict(x="Predicted Label", y="True Label", color="Value")
        )
        
        # Add text annotations
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{cm[i, j]:{text_format}}",
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > 0.5 else "black")
                )
        
        fig.update_layout(
            width=600,
            height=500,
            xaxis_title="Predicted Label",
            yaxis_title="True Label"
        )
        
        return fig
    
    def plot_interactive_feature_importance(self, importance: np.ndarray, feature_names: List[str],
                                       top_n: int = 20, title: str = "Feature Importance") -> go.Figure:
        """
        Plot interactive feature importance using Plotly.
        
        Args:
            importance: Feature importance array
            feature_names: Names of features
            top_n: Number of top features to show
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance and get top N
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create horizontal bar chart
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=title,
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title="Importance",
            yaxis_title=None,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def create_dashboard(self, results: Dict, save_path: str = "dashboard.html") -> None:
        """
        Create an interactive dashboard of results.
        
        Args:
            results: Classification results dictionary
            save_path: Path to save the dashboard HTML
        """
        # Extract data
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        cm = results['confusion_matrix']
        metrics = results['metrics']
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Confusion Matrix",
                "ROC Curve",
                "Metrics",
                "Probability Distribution"
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "table"}, {"type": "histogram"}]
            ]
        )
        
        # Add confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=['Control', 'Depression'],
                y=['Control', 'Depression'],
                colorscale='Blues',
                showscale=False,
                text=[[f"{cm_norm[i, j]:.2%}<br>({cm[i, j]})" for j in range(len(cm[i]))] for i in range(len(cm))],
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Add ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"ROC (AUC={metrics['roc_auc']:.3f})",
                line=dict(color='#2196F3', width=3)
            ),
            row=1, col=2
        )
        
        # Add random classifier line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        # Add metrics table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    align='left',
                    fill_color='#E0E0E0'
                ),
                cells=dict(
                    values=[
                        list(metrics.keys()),
                        [f"{v:.4f}" for v in metrics.values()]
                    ],
                    align='left'
                )
            ),
            row=2, col=1
        )
        
        # Add probability distribution
        fig.add_trace(
            go.Histogram(
                x=y_pred_proba,
                histnorm='probability density',
                name='All',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=2
        )
        
        # Add probability distribution by class
        fig.add_trace(
            go.Histogram(
                x=y_pred_proba[y_test == 0],
                histnorm='probability density',
                name='Control',
                opacity=0.7,
                marker_color='#2196F3',
                nbinsx=30
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=y_pred_proba[y_test == 1],
                histnorm='probability density',
                name='Depression',
                opacity=0.7,
                marker_color='#F44336',
                nbinsx=30
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Depression Classification Results",
            width=1200,
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=2)
        fig.update_yaxes(title_text="Density", row=2, col=2)
        
        # Save to HTML
        save_path = Path(self.save_dir) / save_path
        fig.write_html(str(save_path))
        print(f"Dashboard saved to {save_path}")
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Filename (without directory)
            dpi: Resolution in dots per inch
        """
        # Create full path
        path = Path(self.save_dir) / filename
        
        # Save figure
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {path}")
    
    def create_report(self, results: Dict, output_path: str = "depression_detection_report.html") -> None:
        """
        Create an HTML report with results and visualizations.
        
        Args:
            results: Classification results dictionary
            output_path: Path to save the HTML report
        """
        # Extract data
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        cm = results['confusion_matrix']
        metrics = results['metrics']
        best_params = results.get('best_params', {})
        
        # Create figures
        cm_fig = self.plot_confusion_matrix(cm)
        roc_fig = self.plot_roc_curve(y_test, y_pred_proba)
        pr_fig = self.plot_precision_recall(y_test, y_pred_proba)
        
        # Save figures for report
        cm_path = Path(self.save_dir) / "report_cm.png"
        roc_path = Path(self.save_dir) / "report_roc.png"
        pr_path = Path(self.save_dir) / "report_pr.png"
        
        cm_fig.savefig(cm_path, dpi=120, bbox_inches='tight')
        roc_fig.savefig(roc_path, dpi=120, bbox_inches='tight')
        pr_fig.savefig(pr_path, dpi=120, bbox_inches='tight')