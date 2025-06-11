"""
Main script for the depression detection project.
This script orchestrates the full pipeline from data loading to model evaluation.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib
import time
import warnings
from sklearn.metrics import roc_curve, roc_auc_score

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import local modules
from src.data_loader import DepressionDataLoader
from src.feature_extraction import FeatureExtractor, FeatureConfig
from src.clustering import FeatureClusterer, ClusteringConfig
from src.classification import DepressionClassifier, ClassificationConfig
from src.visualization import DepressionVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Depression Detection from Speech')
    
    # General parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Path to save outputs')
    parser.add_argument('--task', type=str, default='reading', choices=['reading', 'interview'],
                        help='Task to analyze: reading or interview')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Pipeline control
    parser.add_argument('--extract_features', action='store_true',
                        help='Run feature extraction')
    parser.add_argument('--cluster_features', action='store_true',
                        help='Run feature clustering')
    parser.add_argument('--train_classifier', action='store_true',
                        help='Train depression classifier')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--all', action='store_true',
                        help='Run the complete pipeline')
    
    # Feature extraction parameters
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Target sample rate for audio')
    parser.add_argument('--n_mfcc', type=int, default=20,
                        help='Number of MFCCs to extract')
    parser.add_argument('--mfcc_deltas', action='store_true',
                        help='Whether to compute MFCC deltas')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing for feature extraction')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs (default: number of CPU cores - 1)')
    parser.add_argument('--cache_features', action='store_true',
                        help='Cache intermediate features to disk')
    parser.add_argument('--use_cached', action='store_true', 
                        help='Use cached features if available')
    parser.add_argument('--selected_features', type=str, default='all',
                        choices=['all', 'mfcc', 'logmel', 'pwp'],
                        help='Feature types to extract (speeds up extraction)')
    
    # Clustering parameters
    parser.add_argument('--n_clusters', type=int, default=2,
                        help='Number of clusters')
    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA for dimensionality reduction')
    parser.add_argument('--pca_components', type=int, default=15,
                        help='Number of PCA components')
    parser.add_argument('--use_tsne', action='store_true',
                        help='Use t-SNE for visualization')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                        help='Perplexity parameter for t-SNE')
    
    # Feature space visualization options
    parser.add_argument('--visualize_feature_space', action='store_true',
                        help='Create enhanced feature space visualizations')
    parser.add_argument('--interactive_viz', action='store_true',
                        help='Create interactive 3D visualization of feature space')
    parser.add_argument('--manifold_viz', action='store_true',
                        help='Create manifold visualization of feature space')
    parser.add_argument('--show_speaker_ids', action='store_true',
                        help='Show speaker IDs in visualizations')
    
    
    # Classification parameters
    parser.add_argument('--model_type', type=str, default='svm',
                        choices=['svm', 'rf', 'mlp'],
                        help='Classification model type')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test split size')
    parser.add_argument('--balance_method', type=str, default='smote',
                        choices=['smote', 'undersample', None],
                        help='Method for class balancing')
    parser.add_argument('--cross_validate', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--compare_classifiers', action='store_true',
                    help='Compare multiple classifiers')
    parser.add_argument('--classifier1', type=str, default='svm',
                    choices=['svm', 'rf', 'mlp'],
                    help='First classifier to compare')
    parser.add_argument('--classifier2', type=str, default='rf',
                    choices=['svm', 'rf', 'mlp'],
                    help='Second classifier to compare')

    args = parser.parse_args()
    
    # If --all is specified, enable all pipeline stages
    if args.all:
        args.extract_features = True
        args.cluster_features = True
        args.train_classifier = True
        args.visualize = True
    
    return args


def setup_directories(output_dir: str) -> Dict[str, Path]:
    """
    Set up output directories.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary of output directories
    """
    # Create timestamp for unique run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    
    # Create subdirectories
    dirs = {
        'features': run_dir / 'features',
        'models': run_dir / 'models',
        'results': run_dir / 'results',
        'figures': run_dir / 'figures'
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def extract_features(args, dirs: Dict[str, Path]) -> Tuple[Dict[str, Any], np.ndarray, List[str], np.ndarray]:  
    """
    Extract features from audio files with optimized performance.
    
    Args:
        args: Command line arguments
        dirs: Output directories
        
    Returns:
        Tuple of (all_results, features, speaker_ids, labels)
    """
    print("\n=== Feature Extraction ===")
    
    # Check for pre-computed features
    feature_path = dirs['features'] / 'feature_matrix.npy'
    speaker_ids_path = dirs['features'] / 'speaker_ids.npy'
    labels_path = dirs['features'] / 'labels.npy'
    
    if feature_path.exists() and speaker_ids_path.exists() and labels_path.exists():
        print("Loading pre-computed features from current run directory...")
        features = np.load(feature_path)
        speaker_ids = np.load(speaker_ids_path, allow_pickle=True).tolist()
        labels = np.load(labels_path)
        
        # Load previous results if available
        all_results_path = dirs['features'] / 'all_results.joblib'
        all_results = joblib.load(all_results_path) if all_results_path.exists() else None
        
        print(f"Loaded features for {len(speaker_ids)} speakers, shape: {features.shape}")
        return all_results, features, speaker_ids, labels
    
    # Initialize data loader
    data_loader = DepressionDataLoader(args.data_dir)
    
    # Determine which feature types to extract based on args
    feature_types = []
    if args.selected_features == 'all':
        feature_types = ['mfcc', 'logmel', 'pwp']
    else:
        feature_types = [args.selected_features]
    
    print(f"Extracting features: {', '.join(feature_types)}")
    
    # Configure feature extractor
    feature_config = FeatureConfig(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        mfcc_deltas=args.mfcc_deltas
    )
    
    # Use optimized feature extractor if available, otherwise fall back to base
    try:
        # Import the optimized feature extractor
        from src.optimized_feature_extraction import OptimizedFeatureExtractor
        
        feature_extractor = OptimizedFeatureExtractor(
            config=feature_config,
            cache_dir=str(dirs['features']) if args.cache_features else None,
            feature_types=feature_types
        )
        print("Using optimized feature extractor with caching and parallelization")
    except ImportError:
        # Fall back to original feature extractor
        feature_extractor = FeatureExtractor(feature_config)
        print("Using base feature extractor")
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Use parallel processing if enabled and optimized extractor is available
    if args.parallel and 'OptimizedFeatureExtractor' in globals():
        # Process all speakers with parallel processing
        all_results = feature_extractor.process_all_speakers(
            data_loader, 
            task=args.task,
            save_dir=str(dirs['features']) if args.cache_features else None,
            use_parallel=True,
            n_jobs=args.n_jobs
        )
    else:
        # Process all speakers sequentially
        all_results = feature_extractor.process_all_speakers(
            data_loader, 
            task=args.task,
            save_dir=str(dirs['features']) if args.cache_features else None
        )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Feature extraction completed in {elapsed_time:.2f} seconds")
    
    # Extract clusterable features
    features, speaker_ids = feature_extractor.extract_clusterable_features(all_results)
    
    # Get depression labels
    labels_dict = data_loader.get_labels()
    labels = np.array([labels_dict.get(id, 0) for id in speaker_ids])
    
    # Save feature matrix and labels
    np.save(dirs['features'] / 'feature_matrix.npy', features)
    np.save(dirs['features'] / 'speaker_ids.npy', np.array(speaker_ids, dtype=object))
    np.save(dirs['features'] / 'labels.npy', labels)
    
    # Save full results for later reference
    joblib.dump(all_results, dirs['features'] / 'all_results.joblib')
    
    # Print summary
    print(f"Processed {len(speaker_ids)} speakers")
    print(f"Features shape: {features.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return all_results, features, speaker_ids, labels


def cluster_features(args, features: np.ndarray, labels: np.ndarray, 
                   speaker_ids: List[str], dirs: Dict[str, Path]) -> None:
    """
    Cluster features and visualize results.
    
    Args:
        args: Command line arguments
        features: Feature matrix
        labels: Target labels
        speaker_ids: Speaker IDs
        dirs: Output directories
    """
    print("\n=== Feature Clustering ===")
    
    # Configure clusterer
    clustering_config = ClusteringConfig(
        n_clusters=args.n_clusters,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        use_tsne=args.use_tsne,  # Enable t-SNE for better visualization
        random_state=args.random_seed
    )
    clusterer = FeatureClusterer(clustering_config)
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    cluster_range, inertia_values = clusterer.find_optimal_clusters(
        features, 
        max_clusters=min(10, len(speaker_ids) // 5)  # Limit based on sample size
    )
    
    # Fit clustering
    print(f"Fitting clustering with {args.n_clusters} clusters...")
    cluster_labels = clusterer.fit(features)
    
    # Visualize clusters
    print("Visualizing clusters...")
    fig = clusterer.visualize_clusters_2d(features, cluster_labels, labels, speaker_ids)
    fig.savefig(dirs['figures'] / 'cluster_visualization.png', dpi=300)
    plt.close(fig)
    
    # Create enhanced visualizations of the feature space if requested
    if args.visualize_feature_space:
        print("Creating enhanced feature space visualizations...")
        
        # Check if we have the enhanced visualization methods
        if hasattr(clusterer, 'visualize_clusters_3d'):
            # Create interactive 3D visualization if requested
            if args.interactive_viz:
                try:
                    # Create 3D visualization
                    fig_3d = clusterer.visualize_clusters_3d(
                        features, 
                        cluster_labels, 
                        labels, 
                        speaker_ids if args.show_speaker_ids else None
                    )
                    
                    # Save as HTML for interactive viewing
                    import plotly.io as pio
                    pio.write_html(fig_3d, str(dirs['figures'] / 'cluster_visualization_3d.html'))
                    print(f"Interactive 3D visualization saved to {dirs['figures'] / 'cluster_visualization_3d.html'}")
                except Exception as e:
                    print(f"Error creating 3D visualization: {e}")
                    print("Falling back to 2D visualization only")
            
            # Create feature manifold visualization if requested
            if args.manifold_viz:
                try:
                    # Use UMAP for manifold visualization
                    fig_manifold = clusterer.visualize_feature_manifold(
                        features, 
                        labels, 
                        cluster_labels,
                        title="Feature Space Manifold"
                    )
                    fig_manifold.savefig(dirs['figures'] / 'feature_manifold.png', dpi=300)
                    plt.close(fig_manifold)
                except Exception as e:
                    print(f"Error creating manifold visualization: {e}")
        else:
            print("Enhanced visualization methods not available.")
            print("Make sure feature_space_visualization.py is in the src directory.")
    
    # Analyze clusters
    print("Analyzing clusters...")
    analysis = clusterer.analyze_clusters(features, cluster_labels, labels)
    
    # Plot depression distribution across clusters
    if 'depression_distribution' in analysis:
        fig = clusterer.plot_depression_distribution(analysis['depression_distribution'])
        fig.savefig(dirs['figures'] / 'depression_distribution.png', dpi=300)
        plt.close(fig)
    
    # Plot feature importance
    if 'feature_importance' in analysis:
        fig = clusterer.plot_feature_importance(analysis['feature_importance'], top_n=10)
        fig.savefig(dirs['figures'] / 'cluster_feature_importance.png', dpi=300)
        plt.close(fig)
    
    # Save clustering results
    joblib.dump(clusterer, dirs['models'] / 'feature_clusterer.joblib')
    np.save(dirs['results'] / 'cluster_labels.npy', cluster_labels)
    
    print("Clustering completed and results saved")


def train_classifier(args, features: np.ndarray, labels: np.ndarray, dirs: Dict[str, Path]) -> None:
    """
    Train and evaluate depression classifier.
    
    Args:
        args: Command line arguments
        features: Feature matrix
        labels: Target labels
        dirs: Output directories
    """
    print("\n=== Classification ===")
    
    # Configure classifier
    classification_config = ClassificationConfig(
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_seed,
        balance_method=args.balance_method,
        use_pca=args.use_pca,
        pca_components=args.pca_components
    )
    classifier = DepressionClassifier(classification_config)
    
    # Train model
    print(f"Training {args.model_type.upper()} classifier...")
    results = classifier.train(features, labels)
    
    # Cross-validate if requested
    if args.cross_validate:
        print("Performing cross-validation...")
        cv_results = classifier.cross_validate(features, labels, n_folds=5)
        # Save cross-validation results
        joblib.dump(cv_results, dirs['results'] / 'cross_validation_results.joblib')
    
    # Plot and save confusion matrix
    fig = classifier.plot_confusion_matrix()
    fig.savefig(dirs['figures'] / 'confusion_matrix.png', dpi=300)
    plt.close(fig)
    
    # Plot and save ROC curve
    fig = classifier.plot_roc_curve(results['y_test'], results['y_pred_proba'])
    fig.savefig(dirs['figures'] / 'roc_curve.png', dpi=300)
    plt.close(fig)
    
    # Save feature importances if available
    if classifier.feature_importances is not None:
        fig = classifier.plot_feature_importance()
        fig.savefig(dirs['figures'] / 'feature_importance.png', dpi=300)
        plt.close(fig)
    
    # Save model and results
    classifier.save_model(dirs['models'] / 'depression_classifier.joblib')
    joblib.dump(results, dirs['results'] / 'classification_results.joblib')
    
    # Print summary
    print("\nClassification Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    print("\nClassification completed and results saved")


def compare_classifiers(args, features, labels, speaker_ids, dirs):
    """
    Compare multiple classifiers and visualize their performance.
    
    Args:
        args: Command line arguments
        features: Feature matrix
        labels: Target labels
        speaker_ids: Speaker IDs
        dirs: Output directories
    """
    print("\n=== Classifier Comparison ===")
    
    # Define classifiers to compare
    classifier_configs = {
        'svm': ClassificationConfig(
            model_type='svm',
            test_size=args.test_size,
            balance_method=args.balance_method,
            use_pca=args.use_pca,
            pca_components=args.pca_components
        ),
        'rf': ClassificationConfig(
            model_type='rf',
            test_size=args.test_size,
            balance_method=args.balance_method,
            use_pca=args.use_pca,
            pca_components=args.pca_components
        ),
        'mlp': ClassificationConfig(
            model_type='mlp',
            test_size=args.test_size,
            balance_method=args.balance_method,
            use_pca=args.use_pca,
            pca_components=args.pca_components
        )
    }
    
    # Specify which classifiers to compare
    models_to_compare = [args.classifier1, args.classifier2]
    
    # Store results
    all_results = {}
    cv_results = {}
    
    # Train and evaluate each classifier
    for model_name in models_to_compare:
        if model_name not in classifier_configs:
            print(f"Unknown classifier: {model_name}")
            continue
            
        print(f"\nTraining {model_name.upper()} classifier...")
        classifier = DepressionClassifier(classifier_configs[model_name])
        results = classifier.train(features, labels)
        all_results[model_name] = results
        
        # Cross-validate
        print(f"Cross-validating {model_name.upper()}...")
        cv_result = classifier.cross_validate(features, labels, n_folds=5)
        cv_results[model_name] = cv_result
        
        # Save model
        classifier.save_model(dirs['models'] / f'{model_name}_classifier.joblib')
    
    # Create comparison visualizations
    visualizer = DepressionVisualizer(dirs['figures'])
    
    # Compare ROC curves
    fig = plt.figure(figsize=(10, 8))
    for model_name, results in all_results.items():
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        auc = roc_auc_score(results['y_test'], results['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{model_name.upper()} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    visualizer.save_figure(fig, 'roc_comparison.png')
    
    # Compare metrics in a table
    metrics_df = pd.DataFrame({
        model_name.upper(): {
            metric: results['metrics'][metric] 
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        }
        for model_name, results in all_results.items()
    })
    
    # Plot metrics comparison
    fig = plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar', ax=plt.gca())
    plt.title('Performance Metrics Comparison')
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1)
    visualizer.save_figure(fig, 'metrics_comparison.png')
    
    # Save results to CSV
    metrics_df.to_csv(dirs['results'] / 'classifier_comparison.csv')
    
    print("\nClassifier comparison completed.")
    print(f"Results saved to {dirs['results'] / 'classifier_comparison.csv'}")
    
    return all_results, cv_results


def create_visualizations(args, features: np.ndarray, labels: np.ndarray, 
                       speaker_ids: List[str], dirs: Dict[str, Path]) -> None:
    """
    Create additional visualizations.
    
    Args:
        args: Command line arguments
        features: Feature matrix
        labels: Target labels
        speaker_ids: Speaker IDs
        dirs: Output directories
    """
    print("\n=== Creating Visualizations ===")
    
    # Initialize visualizer
    visualizer = DepressionVisualizer(save_dir=dirs['figures'])
    
    # Plot class distribution
    labels_dict = {id: label for id, label in zip(speaker_ids, labels)}
    fig = visualizer.plot_class_distribution(labels_dict)
    visualizer.save_figure(fig, 'class_distribution.png')
    plt.close(fig)
    
    # Load classification results if available
    results_path = dirs['results'] / 'classification_results.joblib'
    if results_path.exists():
        results = joblib.load(results_path)
        
        # Create ROC and Precision-Recall curves
        fig = visualizer.plot_roc_curve(results['y_test'], results['y_pred_proba'])
        visualizer.save_figure(fig, 'detailed_roc_curve.png')
        plt.close(fig)
        
        fig = visualizer.plot_precision_recall(results['y_test'], results['y_pred_proba'])
        visualizer.save_figure(fig, 'precision_recall_curve.png')
        plt.close(fig)
        
        # Create interactive dashboard
        visualizer.create_dashboard(results, save_path='dashboard.html')
    
    # Create PCA and t-SNE visualizations
    if len(features) > 2:
        fig = visualizer.plot_feature_pca(features, labels)
        visualizer.save_figure(fig, 'feature_pca.png')
        plt.close(fig)
        
        if len(features) >= 10:  # t-SNE works better with more samples
            fig = visualizer.plot_feature_tsne(features, labels)
            visualizer.save_figure(fig, 'feature_tsne.png')
            plt.close(fig)
    
    print("Visualizations created and saved")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Set up output directories
    dirs = setup_directories(args.output_dir)
    print(f"Results will be saved to: {dirs['results']}")
    
    # Add feature space visualization methods to the FeatureClusterer class
    # This extends the clusterer with 3D and manifold visualization capabilities
    try:
        from src.feature_space_visualization import add_visualization_methods_to_clusterer
        from src.clustering import FeatureClusterer
        add_visualization_methods_to_clusterer(FeatureClusterer)
        print("Enhanced feature space visualization methods loaded")
    except ImportError as e:
        print(f"Warning: Could not import feature space visualization module: {e}")
        print("Advanced feature space visualizations will not be available")
    
    # Initialize data variables
    all_results = None
    features = None
    speaker_ids = None
    labels = None
    
    # Check for cached features if enabled
    if args.use_cached and not args.extract_features:
        try:
            # Find most recent feature files
            feature_paths = list(Path(args.output_dir).glob('*/features/feature_matrix.npy'))
            if feature_paths:
                latest_path = sorted(feature_paths, key=lambda p: p.stat().st_mtime)[-1]
                feature_dir = latest_path.parent
                
                features = np.load(latest_path)
                speaker_ids = np.load(feature_dir / 'speaker_ids.npy', allow_pickle=True).tolist()
                labels = np.load(feature_dir / 'labels.npy')
                
                # Copy the cached features to the current run directory
                import shutil
                for file in feature_dir.glob('*.npy'):
                    shutil.copy(file, dirs['features'])
                
                print(f"Loaded cached features from: {feature_dir}")
                print(f"Features shape: {features.shape}")
                print(f"Class distribution: {np.bincount(labels)}")
            else:
                print("No cached features found.")
        except Exception as e:
            print(f"Error loading cached features: {e}")
            print("Proceeding with feature extraction...")
            args.extract_features = True
    
    # Run pipeline stages
    
    # 1. Feature extraction
    if args.extract_features:
        all_results, features, speaker_ids, labels = extract_features(args, dirs)
    elif features is None:
        # If we don't have features yet and didn't extract them
        try:
            features = np.load(list(Path(args.output_dir).glob('*/features/feature_matrix.npy'))[-1])
            speaker_ids = np.load(list(Path(args.output_dir).glob('*/features/speaker_ids.npy'))[-1], allow_pickle=True).tolist()
            labels = np.load(list(Path(args.output_dir).glob('*/features/labels.npy'))[-1])
            print(f"Loaded features from disk: {features.shape}")
        except (IndexError, FileNotFoundError):
            print("No feature files found. Run with --extract_features or --all first.")
            return
    
    # 2. Feature clustering
    if args.cluster_features and features is not None:
        cluster_features(args, features, labels, speaker_ids, dirs)
    
    # 3. Classification
    if args.train_classifier and features is not None:
        train_classifier(args, features, labels, dirs)

    # 4. Compare classifiers
    # python main.py --use_cached --compare_classifiers --classifier1 svm --classifier2 rf
    if args.compare_classifiers and features is not None:
        all_results, cv_results = compare_classifiers(args, features, labels, speaker_ids, dirs)
    
    # 5. Visualizations
    if args.visualize and features is not None:
        create_visualizations(args, features, labels, speaker_ids, dirs)
    
    # 6. Feature space visualization only (if requested)
    if args.visualize_feature_space and not args.cluster_features and features is not None:
        print("\n=== Feature Space Visualization Only ===")
        try:
            # Configure a minimal clusterer just for visualization
            from src.clustering import FeatureClusterer, ClusteringConfig
            clustering_config = ClusteringConfig(
                n_clusters=args.n_clusters,
                use_pca=args.use_pca,
                pca_components=args.pca_components,
                use_tsne=args.use_tsne,
                random_state=args.random_seed
            )
            
            # Create the clusterer
            clusterer = FeatureClusterer(clustering_config)
            
            # Run k-means to get cluster labels
            print(f"Fitting K-means with {args.n_clusters} clusters for visualization...")
            cluster_labels = clusterer.fit(features)
            
            # Create visualizations
            if hasattr(clusterer, 'visualize_clusters_3d'):
                if args.interactive_viz:
                    print("Creating interactive 3D visualization...")
                    fig_3d = clusterer.visualize_clusters_3d(
                        features, 
                        cluster_labels, 
                        labels,
                        speaker_ids if args.show_speaker_ids else None,
                        use_tsne=args.use_tsne
                    )
                    
                    # Save as HTML for interactive viewing
                    import plotly.io as pio
                    html_path = dirs['figures'] / 'feature_space_3d.html'
                    pio.write_html(fig_3d, str(html_path))
                    print(f"Interactive 3D visualization saved to {html_path}")
                
                if args.manifold_viz:
                    print("Creating manifold visualization...")
                    try:
                        fig_manifold = clusterer.visualize_feature_manifold(
                            features, 
                            labels, 
                            cluster_labels,
                            title="Feature Space Manifold"
                        )
                        manifold_path = dirs['figures'] / 'feature_manifold.png'
                        fig_manifold.savefig(manifold_path, dpi=300)
                        plt.close(fig_manifold)
                        print(f"Feature manifold visualization saved to {manifold_path}")
                    except Exception as e:
                        print(f"Error creating manifold visualization: {e}")
            else:
                print("Advanced visualization methods not available. Run with --cluster_features instead.")
        except ImportError as e:
            print(f"Error initializing feature space visualization: {e}")
            print("Make sure the feature_space_visualization.py module is in the src directory.")
    
    print(f"\nAll results saved to: {dirs['results'].parent}")


if __name__ == "__main__":
    main()