# Speech-based Depression Detection Project

## Project Overview
This project aims to develop a machine learning approach for detecting depression through speech analysis, exploring various feature extraction techniques and classification methods.

## Project Structure
- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development
- `src/`: Source code for data processing, feature extraction, and modeling
- `tests/`: Unit tests for project components
- `results/`: Saved models, figures, and performance metrics

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Feature Extraction Techniques
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Log Mel Spectrogram
- PWP (Pitch-based Wavelet Peaks)

## Methodology
1. Data Collection
2. Feature Extraction
3. Clustering Analysis
4. Model Training and Evaluation

## Running the Project
```bash
python main.py
```

# Basic Pipeline Contorls

## Run complete pipeline
```bash
python main.py --all
```

## Run specific stages only
```bash
python main.py --extract_features
```
```bash
python main.py --cluster_features
```
```bash
python main.py --train_classifier
```
```bash
python main.py --visualize
```
## Run multiple stages together
```bash
python main.py --extract_features --train_classifier
```


# Common Combined Commands

## Quick development cycle with cached features
```bash
python main.py --use_cached --train_classifier --model_type svm
```

## Extract minimal features and train quickly
```bash
python main.py --extract_features --parallel --selected_features mfcc --train_classifier
```
## Full analysis with optimized feature extraction
```bash
python main.py --extract_features --parallel --cache_features --cluster_features --use_pca --train_classifier --model_type rf --cross_validate --visualize
```
## Create visualizations of feature space
```bash
python main.py --use_cached --visualize_feature_space --interactive_viz --manifold_viz
```
## Compare SVM and RF classifiers with cross-validation
```bash
python main.py --use_cached --compare_classifiers --classifier1 svm --classifier2 rf --cross_validate
```


# Examples for Specific Tasks

## Extract MFCC features using 4 parallel processes
```bash
python main.py --extract_features --selected_features mfcc --parallel --n_jobs 4
```
## Train an SVM classifier on cached features
```bash
python main.py --use_cached --train_classifier --model_type svm --cross_validate
```
## Create a 3D visualization of the feature space
```bash
python main.py --use_cached --visualize_feature_space --interactive_viz
```
## Run K-means clustering with 3 clusters
```bash
python main.py --use_cached --cluster_features --n_clusters 3 --use_pca
```
## Full pipeline with visualization for a presentation
```bash
python main.py --all --parallel --n_clusters 2 --model_type rf --visualize_feature_space --interactive_viz
```


# Data Sources Options

## Specify data directory
```bash
python main.py --data_dir /path/to/data
```
## Specify output directory
```bash
python main.py --output_dir /path/to/results
```

## Choose task type
```bash
python main.py --task reading
```
```bash
python main.py --task interview
```
## Set random seed
```bash
python main.py --random_seed 42
```


# Feature Extraction Options

## Set audio sample rate
```bash
python main.py --extract_features --sample_rate 16000
```

## Set MFCC parameters
```bash
python main.py --extract_features --n_mfcc 20
```
```bash
python main.py --extract_features --mfcc_deltas
```

## Performance optimization
```bash
python main.py --extract_features --parallel
```
```bash
python main.py --extract_features --n_jobs 4
```
```bash
python main.py --extract_features --cache_features
```
```bash
python main.py --extract_features --use_cached
```
## Extract only specific features for speed
```bash
python main.py --extract_features --selected_features mfcc
```
```bash
python main.py --extract_features --selected_features logmel
```
```bash
python main.py --extract_features --selected_features pwp
```
```bash
python main.py --extract_features --selected_features all
```


# Clustering Options

## Set number of clusters
```bash
python main.py --cluster_features --n_clusters 3
```

## Dimensionality reduction
```bash
python main.py --cluster_features --use_pca
```
```bash
python main.py --cluster_features --pca_components 15
```
```bash
python main.py --cluster_features --use_tsne
```
```bash
python main.py --cluster_features --tsne_perplexity 30.0
```


# Feature Space Visualization Options

## Enable feature space visualization
```bash
python main.py --visualize_feature_space
```

## Create interactive 3D visualization
```bash
python main.py --visualize_feature_space --interactive_viz
```
## Create manifold visualization
```bash
python main.py --visualize_feature_space --manifold_viz
```
## Show speaker IDs in visualizations
```bash
python main.py --visualize_feature_space --show_speaker_ids
```



# Classification Options

## Classifier selection
```bash
python main.py --train_classifier --model_type svm
```
```bash
python main.py --train_classifier --model_type rf
```
```bash
python main.py --train_classifier --model_type mlp
```

## Training parameters
```bash
python main.py --train_classifier --test_size 0.2
```
```bash
python main.py --train_classifier --balance_method smote
```
```bash
python main.py --train_classifier --balance_method undersample
```

## Enable cross-validation
```bash
python main.py --train_classifier --cross_validate
```

## Compare classifiers
```bash
python main.py --compare_classifiers
```
```bash
python main.py --compare_classifiers --classifier1 svm --classifier2 rf
```