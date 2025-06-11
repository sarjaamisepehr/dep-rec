import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import joblib
import functools
import warnings
import time

# For caching features
from functools import lru_cache

# Import original modules
from src.feature_extraction import FeatureExtractor as BaseFeatureExtractor
from src.feature_extraction import FeatureConfig

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class OptimizedFeatureExtractor(BaseFeatureExtractor):
    """
    Optimized version of the feature extractor with caching and selective feature extraction.
    Inherits from the base FeatureExtractor class and overrides methods for better performance.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                cache_dir: Optional[str] = None,
                feature_types: List[str] = ['mfcc', 'logmel', 'pwp']):
        """
        Initialize optimized feature extractor.
        
        Args:
            config: Feature extraction configuration
            cache_dir: Directory to cache extracted features
            feature_types: List of feature types to extract
        """
        super().__init__(config or FeatureConfig())
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_types = feature_types
        
        # Initialize cache
        self._feature_cache = {}
    
    @lru_cache(maxsize=100)
    def _cached_load_audio(self, file_path: str, sr: int) -> Tuple[np.ndarray, int]:
        """
        Load audio with caching to avoid repeated disk reads.
        
        Args:
            file_path: Path to audio file
            sr: Target sample rate
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio with librosa
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    
    def extract_features(self, audio_path: Union[str, Path], 
                         feature_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract specified features from an audio file with caching.
        
        Args:
            audio_path: Path to audio file
            feature_types: List of feature types to extract, if None use the ones specified in constructor
                Options: 'mfcc', 'logmel', 'pwp'
        
        Returns:
            Dictionary of extracted features
        """
        audio_path = str(audio_path)
        
        # Use cache if feature cache directory is set
        if self.cache_dir:
            # Create a unique cache key based on file path and config
            file_hash = joblib.hash([
                audio_path, 
                self.config.sample_rate,
                self.config.n_mfcc,
                self.config.mfcc_deltas,
                self.config.n_mels
            ])
            cache_file = self.cache_dir / f"{file_hash}.joblib"
            
            # Check if cached features exist
            if cache_file.exists():
                try:
                    features = joblib.load(cache_file)
                    
                    # If we need all features but some are missing, recompute
                    requested_types = feature_types or self.feature_types
                    if all(ft in features for ft in requested_types):
                        return {k: v for k, v in features.items() if k in requested_types}
                except Exception:
                    # If loading fails, recompute
                    pass
        
        # Use selected feature types or default
        feature_types = feature_types or self.feature_types
        
        # Load and preprocess audio
        try:
            y, sr = self._cached_load_audio(audio_path, self.config.sample_rate)
            
            # Normalize audio if configured
            if self.config.normalize_audio:
                y = librosa.util.normalize(y)
            
            # Initialize result dictionary
            features = {}
            
            # Extract requested features
            for feature_type in feature_types:
                if feature_type.lower() == 'mfcc':
                    features['mfcc'] = self.extract_mfcc(y, sr)
                elif feature_type.lower() == 'logmel':
                    features['logmel'] = self.extract_logmel_spectrogram(y, sr)
                elif feature_type.lower() == 'pwp':
                    features['pwp'] = self.extract_pitch_features(y, sr)
            
            # Cache features if cache directory is set
            if self.cache_dir:
                joblib.dump(features, cache_file)
            
            return features
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return empty features
            return {ft: np.array([]) for ft in feature_types}
    
    def extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Optimized MFCC extraction with the option to downsample audio for faster processing.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            MFCC features
        """
        # Optionally reduce audio length for very long files
        max_samples = 60 * sr  # Maximum 60 seconds
        if len(y) > max_samples:
            # Keep only the first minute
            y = y[:max_samples]
        
        # Extract base MFCCs - use more efficient hop_length for long files
        hop_length = self.config.hop_length
        if len(y) > 30 * sr:  # If longer than 30 seconds
            hop_length = hop_length * 2  # Double the hop length for efficiency
        
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=hop_length
        )
        
        # Add delta and delta-delta features if configured
        if self.config.mfcc_deltas:
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        return mfccs
    
    def process_speaker_data(self, data_loader, speaker_id: str, 
                          task: str = 'reading') -> Dict[str, Any]:
        """
        Process all audio data for a speaker with optimized implementation.
        
        Args:
            data_loader: Initialized data loader
            speaker_id: ID of the speaker to process
            task: 'reading' or 'interview'
            
        Returns:
            Dictionary with extracted features and statistics
        """
        # Create a cache key for the speaker
        if self.cache_dir:
            cache_key = f"{speaker_id}_{task}_{self.config.sample_rate}_{self.config.n_mfcc}"
            cache_file = self.cache_dir / f"speaker_{cache_key}.joblib"
            
            # Check if cached results exist
            if cache_file.exists():
                try:
                    return joblib.load(cache_file)
                except Exception as e:
                    print(f"Error loading cached speaker data for {speaker_id}: {e}")
        
        # If not cached or cache loading failed, process normally
        result = super().process_speaker_data(data_loader, speaker_id, task)
        
        # Cache results if cache directory is set
        if self.cache_dir:
            joblib.dump(result, cache_file)
        
        return result
    
    def process_all_speakers(self, data_loader, task: str = 'reading', 
                          save_dir: Optional[str] = None,
                          use_parallel: bool = False,
                          n_jobs: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all speakers and extract features with parallelization option.
        
        Args:
            data_loader: Initialized data loader
            task: 'reading' or 'interview'
            save_dir: Directory to save results
            use_parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs to use
            
        Returns:
            Dictionary with extracted features for all speakers
        """
        # Get all speakers
        speakers = data_loader.group_files_by_speaker(task=task)
        speaker_ids = list(speakers.keys())
        
        # Create a cache directory if save_dir is provided
        if save_dir:
            self.cache_dir = Path(save_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for cached full results
        if save_dir:
            full_cache = Path(save_dir) / f"all_results_{task}.joblib"
            if full_cache.exists():
                try:
                    print(f"Loading cached results for all speakers ({task})...")
                    return joblib.load(full_cache)
                except Exception as e:
                    print(f"Error loading cached results: {e}")
        
        # Initialize result
        all_results = {}
        
        if use_parallel and len(speaker_ids) > 1:
            # Use parallel processing
            from multiprocessing import Pool, cpu_count
            
            # Determine number of jobs
            if n_jobs is None:
                n_jobs = max(1, cpu_count() - 1)  # Use all but one CPU core
            else:
                n_jobs = min(n_jobs, cpu_count())  # Don't use more cores than available
            
            # Define worker function for processing a single speaker
            def process_speaker_parallel(speaker_id):
                try:
                    start_time = time.time()
                    result = self.process_speaker_data(data_loader, speaker_id, task=task)
                    elapsed = time.time() - start_time
                    print(f"Processed speaker {speaker_id} in {elapsed:.2f} seconds")
                    return speaker_id, result
                except Exception as e:
                    print(f"Error processing speaker {speaker_id}: {e}")
                    return speaker_id, None
            
            print(f"Processing {len(speaker_ids)} speakers using {n_jobs} parallel processes...")
            
            # Process speakers in parallel
            with Pool(processes=n_jobs) as pool:
                results = pool.map(process_speaker_parallel, speaker_ids)
            
            # Combine results
            for speaker_id, result in results:
                if result is not None:
                    all_results[speaker_id] = result
        else:
            # Process sequentially
            from tqdm import tqdm
            
            print(f"Processing {len(speaker_ids)} speakers sequentially...")
            
            for speaker_id in tqdm(speaker_ids, desc=f"Processing {task} speakers"):
                try:
                    result = self.process_speaker_data(data_loader, speaker_id, task=task)
                    all_results[speaker_id] = result
                except Exception as e:
                    print(f"Error processing speaker {speaker_id}: {e}")
        
        # Save combined results if save_dir is provided
        if save_dir:
            joblib.dump(all_results, Path(save_dir) / f"all_results_{task}.joblib")
        
        return all_results
    
    def extract_clusterable_features(self, all_results: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Optimized version of extract_clusterable_features with caching.
        
        Args:
            all_results: Dictionary with extracted features for all speakers
            
        Returns:
            Tuple of (feature_matrix, speaker_ids)
        """
        # Check if we have cached clusterable features
        if self.cache_dir:
            cache_file = self.cache_dir / "clusterable_features.npz"
            if cache_file.exists():
                try:
                    data = np.load(cache_file, allow_pickle=True)
                    return data['features'], data['speaker_ids'].tolist()
                except Exception as e:
                    print(f"Error loading cached clusterable features: {e}")
        
        # If not cached or loading failed, compute normally
        features, speaker_ids = super().extract_clusterable_features(all_results)
        
        # Cache results if cache directory is set
        if self.cache_dir:
            np.savez(self.cache_dir / "clusterable_features.npz", 
                    features=features, 
                    speaker_ids=np.array(speaker_ids, dtype=object))
        
        return features, speaker_ids
