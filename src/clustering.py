import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm

@dataclass
class ClusteringConfig:
    """Configuration for clustering."""
    # KMeans parameters
    n_clusters: int = 2  # Number of clusters
    n_init: int = 10  # Number of initializations
    max_iter: int = 300  # Maximum number of iterations
    random_state: int = 42  # Random seed
    
    # Dimensionality reduction
    use_pca: bool = True  # Whether to use PCA
    pca_components: int = 15  # Number of PCA components
    use_tsne: bool = True  # Whether to use t-SNE for visualization
    tsne_components: int = 2  # Number of t-SNE components
    
    # Visualization
    plot_elbow: bool = True  # Whether to plot elbow curve
    max_clusters_elbow: int = 10  # Maximum number of clusters for elbow curve


class FeatureClusterer:
    """Cluster audio features and visualize feature space."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize clusterer with configuration.
        
        Args:
            config: Clustering configuration
        """
        self.config = config or ClusteringConfig()
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.tsne = None
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features before clustering.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
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
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA if configured
        if self.config.use_pca:
            # Determine number of components (either specified or minimum of data dimensions)
            n_components = min(self.config.pca_components, features_scaled.shape[0], features_scaled.shape[1])
            self.pca = PCA(n_components=n_components, random_state=self.config.random_state)
            features_reduced = self.pca.fit_transform(features_scaled)
            
            # Report explained variance
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA with {n_components} components explains {explained_var:.2%} of variance")
            
            return features_reduced
        
        return features_scaled
    
    def find_optimal_clusters(self, features: np.ndarray, 
                              max_clusters: int = 10) -> Tuple[List[int], List[float]]:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            features: Preprocessed feature matrix
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Tuple of (cluster_range, inertia_values)
        """
        cluster_range = range(1, max_clusters + 1)
        inertia_values = []
        silhouette_values = []
        
        for k in tqdm(cluster_range, desc="Finding optimal clusters"):
            if k == 1:
                # Silhouette score is not defined for k=1
                kmeans = KMeans(n_clusters=k, n_init=self.config.n_init, 
                               max_iter=self.config.max_iter, random_state=self.config.random_state)
                kmeans.fit(features)
                inertia_values.append(kmeans.inertia_)
                silhouette_values.append(0)  # Placeholder for k=1
            else:
                kmeans = KMeans(n_clusters=k, n_init=self.config.n_init, 
                               max_iter=self.config.max_iter, random_state=self.config.random_state)
                labels = kmeans.fit_predict(features)
                inertia_values.append(kmeans.inertia_)
                silhouette_values.append(silhouette_score(features, labels))
        
        # Plot elbow curve if configured
        if self.config.plot_elbow:
            self.plot_elbow_curve(cluster_range, inertia_values, silhouette_values)
        
        return list(cluster_range), inertia_values
    
    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit KMeans clustering to features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        # Preprocess features
        features_processed = self.preprocess_features(features)
        
        # Create and fit KMeans
        self.kmeans = KMeans(
            n_clusters=self.config.n_clusters, 
            n_init=self.config.n_init,
            max_iter=self.config.max_iter, 
            random_state=self.config.random_state
        )
        
        labels = self.kmeans.fit_predict(features_processed)
        
        # Calculate clustering metrics
        if self.config.n_clusters > 1:
            silhouette = silhouette_score(features_processed, labels)
            calinski_harabasz = calinski_harabasz_score(features_processed, labels)
            davies_bouldin = davies_bouldin_score(features_processed, labels)
            
            print(f"Clustering metrics:")
            print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
            print(f"  Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        
        return labels
    
    def visualize_clusters_2d(self, features: np.ndarray, labels: np.ndarray, 
                              true_labels: Optional[np.ndarray] = None, 
                              feature_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Visualize clusters in 2D space.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Cluster labels
            true_labels: True labels (depression/control), if available
            feature_names: Names of samples, if available
            
        Returns:
            Matplotlib figure
        """
        # Preprocess features if not done yet
        if self.scaler is None or not hasattr(self.scaler, 'mean_'):
            features = self.preprocess_features(features)
        else:
            # Check if features are already preprocessed
            if features.shape[1] != self.scaler.mean_.shape[0]:
                features = self.preprocess_features(features)
        
        # Reduce to 2D for visualization
        if features.shape[1] > 2:
            if self.config.use_tsne:
                # Use t-SNE for better visualization of cluster structure
                self.tsne = TSNE(
                    n_components=2,
                    random_state=self.config.random_state,
                    perplexity=min(30, features.shape[0] - 1)  # Adjust perplexity for small datasets
                )
                features_2d = self.tsne.fit_transform(features)
                method = "t-SNE"
            elif self.pca is not None:
                # Use first two PCA components
                features_2d = self.pca.transform(features)[:, :2]
                method = "PCA"
            else:
                # Apply PCA directly
                pca = PCA(n_components=2, random_state=self.config.random_state)
                features_2d = pca.fit_transform(features)
                method = "PCA"
        else:
            # Already 2D
            features_2d = features
            method = "Original"
        
        # Create figure with subplots - one for clusters, one for true labels if available
        n_plots = 2 if true_labels is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots*7, 5))
        
        # Convert to array if only one plot
        if n_plots == 1:
            axes = np.array([axes])
        
        # Plot cluster assignments
        scatter = axes[0].scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='viridis',
            alpha=0.8, s=80, edgecolors='w'
        )
        axes[0].set_title(f'K-Means Clustering ({method} projection)')
        axes[0].set_xlabel(f'{method} Component 1')
        axes[0].set_ylabel(f'{method} Component 2')
        
        # Add legend for clusters
        legend1 = axes[0].legend(*scatter.legend_elements(),
                                title="Clusters",
                                loc="upper right")
        axes[0].add_artist(legend1)
        
        # Add centroid markers if kmeans is fitted
        if self.kmeans is not None and hasattr(self.kmeans, 'cluster_centers_'):
            if self.config.use_tsne:
                # For t-SNE we can't project centroids directly
                # Instead, find the sample closest to each centroid in the original space
                closest_points = []
                for center in self.kmeans.cluster_centers_:
                    distances = np.linalg.norm(features - center, axis=1)
                    closest_idx = np.argmin(distances)
                    closest_points.append(features_2d[closest_idx])
                centroid_2d = np.array(closest_points)
            else:
                # For PCA we can project centroids directly
                if self.pca is not None:
                    centroid_2d = self.pca.transform(self.kmeans.cluster_centers_)[:, :2]
                else:
                    # Apply PCA directly
                    pca = PCA(n_components=2, random_state=self.config.random_state)
                    pca.fit(features)
                    centroid_2d = pca.transform(self.kmeans.cluster_centers_)
                    
            # Plot centroids
            axes[0].scatter(
                centroid_2d[:, 0], centroid_2d[:, 1],
                marker='X', s=200, c='red', edgecolors='k', alpha=0.8,
                label='Centroids'
            )
            axes[0].legend()
        
        # Plot true labels if available
        if true_labels is not None:
            scatter = axes[1].scatter(
                features_2d[:, 0], features_2d[:, 1],
                c=true_labels, cmap='coolwarm',
                alpha=0.8, s=80, edgecolors='w'
            )
            axes[1].set_title(f'True Labels ({method} projection)')
            axes[1].set_xlabel(f'{method} Component 1')
            axes[1].set_ylabel(f'{method} Component 2')
            
            # Add legend for true labels
            legend2 = axes[1].legend(*scatter.legend_elements(),
                                    title="Depression",
                                    loc="upper right")
            axes[1].add_artist(legend2)
        
        # Add annotations if feature_names are provided
        if feature_names is not None:
            for i, name in enumerate(feature_names):
                axes[0].annotate(name, (features_2d[i, 0], features_2d[i, 1]), 
                               fontsize=8, alpha=0.7)
                if true_labels is not None:
                    axes[1].annotate(name, (features_2d[i, 0], features_2d[i, 1]), 
                                   fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_elbow_curve(self, cluster_range: List[int], inertia_values: List[float], 
                         silhouette_values: List[float]) -> plt.Figure:
        """
        Plot elbow curve for finding optimal number of clusters.
        
        Args:
            cluster_range: Range of cluster numbers
            inertia_values: Inertia values for each number of clusters
            silhouette_values: Silhouette scores for each number of clusters
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot inertia (elbow curve)
        ax1.plot(cluster_range, inertia_values, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        
        # Plot silhouette score
        ax2.plot(cluster_range, silhouette_values, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Method for Optimal k')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def analyze_clusters(self, features: np.ndarray, labels: np.ndarray, 
                         true_labels: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze cluster characteristics.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            true_labels: True labels (depression/control), if available
            feature_names: Names of features, if available
            
        Returns:
            Dictionary with cluster analysis results
        """
        # Initialize results dictionary
        results = {
            'cluster_sizes': {},
            'cluster_centers': {},
            'feature_importance': {},
        }
        
        # Count samples in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            results['cluster_sizes'][f'Cluster {label}'] = int(count)
        
        # If we have true labels, analyze relationship between clusters and depression
        if true_labels is not None:
            results['depression_distribution'] = {}
            for label in unique_labels:
                cluster_mask = labels == label
                depression_count = np.sum(true_labels[cluster_mask] == 1)
                control_count = np.sum(true_labels[cluster_mask] == 0)
                total_count = depression_count + control_count
                
                results['depression_distribution'][f'Cluster {label}'] = {
                    'Depression': int(depression_count),
                    'Control': int(control_count),
                    'Depression Ratio': float(depression_count / total_count) if total_count > 0 else 0
                }
        
        # Calculate cluster centers
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_center = np.mean(features[cluster_mask], axis=0)
            results['cluster_centers'][f'Cluster {label}'] = cluster_center
        
        # Calculate feature importance based on variance and center distance
        if self.kmeans is not None and hasattr(self.kmeans, 'cluster_centers_'):
            # Calculate variance of each feature across all samples
            feature_variance = np.var(features, axis=0)
            
            # Calculate distances between cluster centers
            n_clusters = len(self.kmeans.cluster_centers_)
            if n_clusters > 1:
                center_distances = []
                for i in range(n_clusters):
                    for j in range(i+1, n_clusters):
                        center_distances.append(
                            np.abs(self.kmeans.cluster_centers_[i] - self.kmeans.cluster_centers_[j])
                        )
                
                # Average distance for each feature
                avg_center_distances = np.mean(center_distances, axis=0)
                
                # Importance = variance * distance between centers
                feature_importance = feature_variance * avg_center_distances
                
                # Normalize importance
                feature_importance = feature_importance / np.sum(feature_importance)
                
                # Store in results
                if feature_names is not None and len(feature_names) == len(feature_importance):
                    for name, importance in zip(feature_names, feature_importance):
                        results['feature_importance'][name] = float(importance)
                else:
                    for i, importance in enumerate(feature_importance):
                        results['feature_importance'][f'Feature {i}'] = float(importance)
        
        return results
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                               top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary of feature importances
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame for easy sorting and plotting
        df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
        
        # Sort by importance and take top_n
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=df, x='Importance', y='Feature', ax=ax)
        ax.set_title(f'Top {top_n} Features by Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        return fig
    
    def plot_depression_distribution(self, depression_distribution: Dict[str, Dict]) -> plt.Figure:
        """
        Plot distribution of depression cases across clusters.
        
        Args:
            depression_distribution: Dictionary with depression counts per cluster
            
        Returns:
            Matplotlib figure
        """
        # Process data for plotting
        clusters = list(depression_distribution.keys())
        depression_counts = [depression_distribution[c]['Depression'] for c in clusters]
        control_counts = [depression_distribution[c]['Control'] for c in clusters]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Cluster': np.repeat(clusters, 2),
            'Group': ['Depression', 'Control'] * len(clusters),
            'Count': depression_counts + control_counts
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x='Cluster', y='Count', hue='Group', ax=ax)
        ax.set_title('Distribution of Depression Cases Across Clusters')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        
        # Add percentages on top of bars
        total_counts = np.array(depression_counts) + np.array(control_counts)
        for i, cluster in enumerate(clusters):
            ax.text(
                i, total_counts[i] + 1,
                f"{depression_distribution[cluster]['Depression Ratio']:.1%}",
                ha='center'
            )
        
        plt.tight_layout()
        return fig


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
    
    # Extract features for a subset of speakers
    speaker_ids = list(reading_speakers.keys())[:20]  # Limit to 20 speakers for example
    all_results = {}
    
    for speaker_id in tqdm(speaker_ids, desc="Processing speakers"):
        try:
            result = feature_extractor.process_speaker_data(data_loader, speaker_id, task='reading')
            all_results[speaker_id] = result
        except Exception as e:
            print(f"Error processing speaker {speaker_id}: {e}")
    
    # Extract clusterable features
    features, ids = feature_extractor.extract_clusterable_features(all_results)
    
    # Get true labels
    labels_dict = data_loader.get_labels()
    true_labels = np.array([labels_dict.get(id, 0) for id in ids])
    
    # Initialize clusterer
    clustering_config = ClusteringConfig(
        n_clusters=2,
        use_pca=True,
        pca_components=10,
        use_tsne=True
    )
    clusterer = FeatureClusterer(clustering_config)
    
    # Find optimal number of clusters
    cluster_range, inertia_values = clusterer.find_optimal_clusters(features, max_clusters=5)
    
    # Fit clustering
    cluster_labels = clusterer.fit(features)
    
    # Visualize clusters
    fig = clusterer.visualize_clusters_2d(features, cluster_labels, true_labels, ids)
    plt.savefig("cluster_visualization.png")
    
    # Analyze clusters
    analysis = clusterer.analyze_clusters(features, cluster_labels, true_labels)
    
    # Plot depression distribution across clusters
    if 'depression_distribution' in analysis:
        fig = clusterer.plot_depression_distribution(analysis['depression_distribution'])
        plt.savefig("depression_distribution.png")
    
    # Plot feature importance
    if 'feature_importance' in analysis:
        fig = clusterer.plot_feature_importance(analysis['feature_importance'], top_n=10)
        plt.savefig("feature_importance.png")
    
    print("Clustering completed and visualizations saved.")