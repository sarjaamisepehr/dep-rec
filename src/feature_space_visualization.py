import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def visualize_clusters_3d(features: np.ndarray, 
                         cluster_labels: np.ndarray, 
                         true_labels: Optional[np.ndarray] = None, 
                         feature_names: Optional[List[str]] = None,
                         use_pca: bool = True,
                         use_tsne: bool = False,
                         title: str = "Feature Clusters in 3D") -> go.Figure:
    """
    Create an interactive 3D visualization of feature clusters.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        cluster_labels: Cluster labels assigned by K-means
        true_labels: True depression/control labels (optional)
        feature_names: IDs or names for each data point (optional)
        use_pca: Whether to use PCA for dimension reduction (True for PCA, False for t-SNE)
        use_tsne: Whether to use t-SNE instead of PCA
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Prepare data
    n_samples = features.shape[0]
    
    # Apply dimensionality reduction if needed
    if features.shape[1] > 3:
        if use_tsne:
            print("Applying t-SNE for 3D visualization...")
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, n_samples-1), n_iter=1000)
            reduced_features = tsne.fit_transform(features)
            method = "t-SNE"
        elif use_pca:
            print("Applying PCA for 3D visualization...")
            pca = PCA(n_components=3, random_state=42)
            reduced_features = pca.fit_transform(features)
            method = "PCA"
        else:
            # If data is already 3D or very close, take first 3 dimensions
            reduced_features = features[:, :3]
            method = "Original"
    else:
        # Data already has 3 or fewer dimensions
        if features.shape[1] < 3:
            # Pad with zeros if fewer than 3 dimensions
            padded = np.zeros((n_samples, 3))
            padded[:, :features.shape[1]] = features
            reduced_features = padded
        else:
            reduced_features = features
        method = "Original"
    
    # Create DataFrame for plotting
    df = pd.DataFrame(reduced_features, columns=['x', 'y', 'z'])
    df['Cluster'] = [f'Cluster {label}' for label in cluster_labels]
    
    if true_labels is not None:
        df['Depression'] = ['Depressed' if label == 1 else 'Control' for label in true_labels]
    
    if feature_names is not None:
        df['ID'] = feature_names
    
    # Create figure
    if true_labels is not None:
        # Create two subplot views: by cluster and by depression status
        fig = make_subplots(
            rows=1, cols=2, 
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Clusters', 'Depression Status')
        )
        
        # Add cluster view
        fig.add_trace(
            go.Scatter3d(
                x=df['x'], y=df['y'], z=df['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=cluster_labels,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Cluster', x=0.45)
                ),
                text=df['ID'] if 'ID' in df else None,
                hoverinfo='text' if 'ID' in df else None,
                name='Clusters'
            ),
            row=1, col=1
        )
        
        # Add depression status view
        colors = df['Depression'].map({'Depressed': '#FF5733', 'Control': '#3385FF'})
        fig.add_trace(
            go.Scatter3d(
                x=df['x'], y=df['y'], z=df['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    opacity=0.8
                ),
                text=df['ID'] if 'ID' in df else None,
                hoverinfo='text' if 'ID' in df else None,
                name='Depression Status'
            ),
            row=1, col=2
        )
        
        # Add legend for depression status
        for status, color in [('Depressed', '#FF5733'), ('Control', '#3385FF')]:
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=5, color=color),
                    name=status,
                    showlegend=True
                ),
                row=1, col=2
            )
        
    else:
        # Simple 3D scatter plot with clusters
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=df['x'], y=df['y'], z=df['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=cluster_labels,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Cluster')
                ),
                text=df['ID'] if 'ID' in df else None,
                hoverinfo='text' if 'ID' in df else None
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{title} ({method} projection)",
        scene=dict(
            xaxis_title=f"{method} Component 1",
            yaxis_title=f"{method} Component 2",
            zaxis_title=f"{method} Component 3",
        ),
        scene2=dict(
            xaxis_title=f"{method} Component 1",
            yaxis_title=f"{method} Component 2",
            zaxis_title=f"{method} Component 3",
        ) if true_labels is not None else None,
        width=1200,
        height=600,
        legend=dict(x=1.0, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def visualize_feature_manifold(features: np.ndarray,
                             labels: np.ndarray,
                             cluster_labels: Optional[np.ndarray] = None,
                             n_neighbors: int = 15,
                             min_dist: float = 0.1,
                             title: str = "Feature Space Manifold") -> plt.Figure:
    """
    Create a manifold visualization of the feature space using UMAP.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: True depression/control labels
        cluster_labels: Cluster labels assigned by K-means (optional)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Check if UMAP is available
    if not UMAP_AVAILABLE:
        print("UMAP not available. Install with: pip install umap-learn")
        # Fall back to t-SNE
        print("Falling back to t-SNE for manifold visualization...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
        method = "t-SNE"
    else:
        # Configure UMAP
        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, features.shape[0] - 1),
            min_dist=min_dist,
            n_components=2,
            random_state=42
        )
        method = "UMAP"
    
    # Apply dimensionality reduction
    print(f"Applying {method} for manifold visualization...")
    embedding = reducer.fit_transform(features)
    
    # Prepare for visualization
    if cluster_labels is None:
        # Single plot with depression labels
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='coolwarm',
            alpha=0.8, s=60, edgecolors='w'
        )
        
        legend1 = ax.legend(*scatter.legend_elements(),
                           title="Depression",
                           loc="upper right")
        ax.add_artist(legend1)
        
        ax.set_title(f"{title} ({method})")
        ax.set_xlabel(f"{method} Dimension 1")
        ax.set_ylabel(f"{method} Dimension 2")
        
    else:
        # Compare depression labels with cluster assignments
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot by depression status
        scatter1 = axes[0].scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='coolwarm',
            alpha=0.8, s=60, edgecolors='w'
        )
        
        legend1 = axes[0].legend(*scatter1.legend_elements(),
                                title="Depression",
                                loc="upper right")
        axes[0].add_artist(legend1)
        
        axes[0].set_title(f"Depression Status ({method})")
        axes[0].set_xlabel(f"{method} Dimension 1")
        axes[0].set_ylabel(f"{method} Dimension 2")
        
        # Plot by cluster
        scatter2 = axes[1].scatter(
            embedding[:, 0], embedding[:, 1],
            c=cluster_labels, cmap='viridis',
            alpha=0.8, s=60, edgecolors='w'
        )
        
        legend2 = axes[1].legend(*scatter2.legend_elements(),
                                title="Cluster",
                                loc="upper right")
        axes[1].add_artist(legend2)
        
        axes[1].set_title(f"Cluster Assignment ({method})")
        axes[1].set_xlabel(f"{method} Dimension 1")
        axes[1].set_ylabel(f"{method} Dimension 2")
        
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig

# Modify these to be methods on the FeatureClusterer class
def clusterer_visualize_clusters_3d(self, features: np.ndarray, 
                                 cluster_labels: np.ndarray, 
                                 true_labels: Optional[np.ndarray] = None, 
                                 feature_names: Optional[List[str]] = None,
                                 use_pca: bool = True,
                                 use_tsne: bool = False) -> go.Figure:
    """
    Method wrapper for visualize_clusters_3d for the FeatureClusterer class.
    """
    return visualize_clusters_3d(
        features=features,
        cluster_labels=cluster_labels,
        true_labels=true_labels,
        feature_names=feature_names,
        use_pca=use_pca,
        use_tsne=use_tsne
    )

def clusterer_visualize_feature_manifold(self, features: np.ndarray,
                                      labels: np.ndarray,
                                      cluster_labels: Optional[np.ndarray] = None,
                                      n_neighbors: int = 15,
                                      min_dist: float = 0.1,
                                      title: str = "Feature Space Manifold") -> plt.Figure:
    """
    Method wrapper for visualize_feature_manifold for the FeatureClusterer class.
    """
    return visualize_feature_manifold(
        features=features,
        labels=labels,
        cluster_labels=cluster_labels,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        title=title
    )

def add_visualization_methods_to_clusterer(clusterer_class):
    """Add visualization methods to the FeatureClusterer class."""
    # Add the method wrappers that handle the 'self' parameter correctly
    clusterer_class.visualize_clusters_3d = clusterer_visualize_clusters_3d
    clusterer_class.visualize_feature_manifold = clusterer_visualize_feature_manifold
    return clusterer_class