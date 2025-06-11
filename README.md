# Speech-based Depression Detection Project

## Project Overview
This project aims to develop a machine learning approach for detecting depression through speech analysis, exploring various feature extraction techniques and classification methods.

## Project Structure
- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development
- `src/`: Source code for data processing, feature extraction, and modeling
- `tests/`: Unit tests for project components
- `results/`: Saved models, figures, and performance metrics

## 0. Installation

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

# 1. Basic Pipeline Contorls

## 1.1. Run complete pipeline
```bash
python main.py --all
```

## 1.2. Run specific stages only
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
## 1.3. Run multiple stages together
```bash
python main.py --extract_features --train_classifier
```


# 2. Common Combined Commands

## 2.1. Quick development cycle with cached features
```bash
python main.py --use_cached --train_classifier --model_type svm
```

## 2.2. Extract minimal features and train quickly
```bash
python main.py --extract_features --parallel --selected_features mfcc --train_classifier
```
## 2.3. Full analysis with optimized feature extraction
```bash
python main.py --extract_features --parallel --cache_features --cluster_features --use_pca --train_classifier --model_type rf --cross_validate --visualize
```
## 2.4. Create visualizations of feature space
```bash
python main.py --use_cached --visualize_feature_space --interactive_viz --manifold_viz
```
## 2.5. Compare SVM and RF classifiers with cross-validation
```bash
python main.py --use_cached --compare_classifiers --classifier1 svm --classifier2 rf --cross_validate
```


# 3. Examples for Specific Tasks

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


# 4. Data Sources Options

## 4.1. Specify data directory
```bash
python main.py --data_dir /path/to/data
```
## 4.2. Specify output directory
```bash
python main.py --output_dir /path/to/results
```

## 4.3. Choose task type
```bash
python main.py --task reading
```
```bash
python main.py --task interview
```
## 4.4. Set random seed
```bash
python main.py --random_seed 42
```


# 5. Feature Extraction Options

## 5.1. Set audio sample rate
```bash
python main.py --extract_features --sample_rate 16000
```

## 5.2. Set MFCC parameters
```bash
python main.py --extract_features --n_mfcc 20
```
```bash
python main.py --extract_features --mfcc_deltas
```

## 5.3. Performance optimization
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
## 5.4. Extract only specific features for speed
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


# 5. Clustering Options

## 5.1. Set number of clusters
```bash
python main.py --cluster_features --n_clusters 3
```

## 5.2. Dimensionality reduction
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


# 6. Feature Space Visualization Options

## 6.1. Enable feature space visualization
```bash
python main.py --visualize_feature_space
```

## 6.2. Create interactive 3D visualization
```bash
python main.py --visualize_feature_space --interactive_viz
```
## 6.3. Create manifold visualization
```bash
python main.py --visualize_feature_space --manifold_viz
```
## 6.4. Show speaker IDs in visualizations
```bash
python main.py --visualize_feature_space --show_speaker_ids
```



# 7. Classification Options

## 7.1. Classifier selection
```bash
python main.py --train_classifier --model_type svm
```
```bash
python main.py --train_classifier --model_type rf
```
```bash
python main.py --train_classifier --model_type mlp
```

## 7.2. Training parameters
```bash
python main.py --train_classifier --test_size 0.2
```
```bash
python main.py --train_classifier --balance_method smote
```
```bash
python main.py --train_classifier --balance_method undersample
```

## 7.3. Enable cross-validation
```bash
python main.py --train_classifier --cross_validate
```

## 7.4. Compare classifiers
```bash
python main.py --compare_classifiers
```
```bash
python main.py --compare_classifiers --classifier1 svm --classifier2 rf
```