import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
from dataclasses import dataclass

# For type hints
from src.data_loader import DepressionDataLoader, SpeakerInfo


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # General parameters
    sample_rate: int = 16000  # Target sample rate for resampling
    normalize_audio: bool = True  # Whether to normalize audio before feature extraction
    
    # MFCC parameters
    n_mfcc: int = 20  # Number of MFCCs to extract
    mfcc_deltas: bool = True  # Whether to compute delta and delta-delta features
    
    # Log-Mel spectrogram parameters
    n_mels: int = 128  # Number of Mel bands
    n_fft: int = 2048  # FFT window size
    hop_length: int = 512  # Hop length for STFT
    
    # PWP (pitch) parameters
    f0_min: float = 60.0  # Minimum fundamental frequency
    f0_max: float = 400.0  # Maximum fundamental frequency


class FeatureExtractor:
    """Extract audio features for depression detection."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature extractor with configuration.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureConfig()
    
    def extract_features(self, audio_path: Union[str, Path], 
                         feature_types: List[str] = ['mfcc', 'logmel', 'pwp']) -> Dict[str, np.ndarray]:
        """
        Extract specified features from an audio file.
        
        Args:
            audio_path: Path to audio file
            feature_types: List of feature types to extract
                Options: 'mfcc', 'logmel', 'pwp'
        
        Returns:
            Dictionary of extracted features
        """
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
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
        
        return features
    
    def extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCCs).
        
        Args:
            y: Audio signal
            sr: Sample rate
        
        Returns:
            MFCC features (with deltas if configured)
        """
        # Extract base MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Add delta and delta-delta features if configured
        if self.config.mfcc_deltas:
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        return mfccs
    
    def extract_logmel_spectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract log-Mel spectrogram.
        
        Args:
            y: Audio signal
            sr: Sample rate
        
        Returns:
            Log-Mel spectrogram features
        """
        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        
        # Convert to log scale (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return log_mel_spectrogram
    
    def extract_pitch_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract pitch-related features (PWP - Pitch Width Period).
        
        Args:
            y: Audio signal
            sr: Sample rate
        
        Returns:
            Pitch-related features including:
            - F0 (fundamental frequency) contour
            - Voice Activity Detection (VAD)
            - Jitter and Shimmer (variations in pitch and amplitude)
        """
        # Extract fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=y,
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            sr=sr,
            hop_length=self.config.hop_length
        )
        
        # Replace NaN values in f0 with 0 for unvoiced frames
        f0 = np.nan_to_num(f0)
        
        # Extract other pitch-related features
        
        # 1. Pitch statistics in voiced regions
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 0:
            f0_mean = np.mean(voiced_f0)
            f0_std = np.std(voiced_f0)
            f0_range = np.max(voiced_f0) - np.min(voiced_f0) if len(voiced_f0) > 1 else 0
        else:
            f0_mean = 0
            f0_std = 0
            f0_range = 0
        
        # 2. Voice Activity Detection ratio
        vad_ratio = np.mean(voiced_flag)
        
        # 3. Calculate jitter (variation in pitch)
        jitter = self._calculate_jitter(f0, voiced_flag)
        
        # 4. Calculate shimmer (variation in amplitude)
        shimmer = self._calculate_shimmer(y, sr, f0, voiced_flag)
        
        # Combine features
        # Format: [f0, voiced_flag, voiced_probs, f0_stats, vad_ratio, jitter, shimmer]
        pitch_features = {
            'f0': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_range': f0_range,
            'vad_ratio': vad_ratio,
            'jitter': jitter,
            'shimmer': shimmer
        }
        
        return pitch_features
    
    def _calculate_jitter(self, f0: np.ndarray, voiced_flag: np.ndarray) -> float:
        """
        Calculate jitter (variation in pitch periods).
        
        Args:
            f0: Fundamental frequency contour
            voiced_flag: Boolean array indicating voiced frames
            
        Returns:
            Jitter value (average absolute difference between consecutive periods)
        """
        # Get consecutive voiced frames
        voiced_f0 = f0[voiced_flag]
        
        if len(voiced_f0) < 2:
            return 0.0
        
        # Convert frequency to period (time)
        periods = 1.0 / (voiced_f0 + 1e-10)  # Add small value to avoid division by zero
        
        # Calculate absolute differences between consecutive periods
        abs_diff = np.abs(np.diff(periods))
        
        # Calculate jitter as average absolute difference divided by average period
        jitter = np.mean(abs_diff) / np.mean(periods) if np.mean(periods) > 0 else 0.0
        
        return jitter
    
    def _calculate_shimmer(self, y: np.ndarray, sr: int, 
                           f0: np.ndarray, voiced_flag: np.ndarray) -> float:
        """
        Calculate shimmer (variation in amplitude).
        
        Args:
            y: Audio signal
            sr: Sample rate
            f0: Fundamental frequency contour
            voiced_flag: Boolean array indicating voiced frames
            
        Returns:
            Shimmer value (average absolute difference between consecutive amplitudes)
        """
        if np.sum(voiced_flag) < 2:
            return 0.0
        
        # Get frame-wise RMS energy
        hop_length = self.config.hop_length
        energy = librosa.feature.rms(y=y, frame_length=self.config.n_fft, hop_length=hop_length)[0]
        
        # Only consider energy in voiced frames
        voiced_energy = energy[voiced_flag[:len(energy)]] if len(voiced_flag) > len(energy) else energy[voiced_flag]
        
        if len(voiced_energy) < 2:
            return 0.0
        
        # Calculate absolute differences between consecutive energies
        abs_diff = np.abs(np.diff(voiced_energy))
        
        # Calculate shimmer as average absolute difference divided by average energy
        shimmer = np.mean(abs_diff) / np.mean(voiced_energy) if np.mean(voiced_energy) > 0 else 0.0
        
        return shimmer
    
    def extract_global_statistics(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract global statistics from features for classification.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary of global statistics per feature type
        """
        stats = {}
        
        # Process MFCC statistics
        if 'mfcc' in features:
            mfccs = features['mfcc']
            # Calculate statistics along time dimension (axis 1)
            stats['mfcc_mean'] = np.mean(mfccs, axis=1)
            stats['mfcc_std'] = np.std(mfccs, axis=1)
            stats['mfcc_skew'] = scipy.stats.skew(mfccs, axis=1)
            stats['mfcc_kurtosis'] = scipy.stats.kurtosis(mfccs, axis=1)
        
        # Process log-Mel spectrogram statistics
        if 'logmel' in features:
            logmel = features['logmel']
            # Calculate band-wise statistics (mean across time)
            stats['logmel_mean'] = np.mean(logmel, axis=1)
            stats['logmel_std'] = np.std(logmel, axis=1)
            # Calculate global energy statistics
            stats['logmel_energy_mean'] = np.mean(logmel)
            stats['logmel_energy_std'] = np.std(logmel)
        
        # Process PWP statistics
        if 'pwp' in features and isinstance(features['pwp'], dict):
            pwp = features['pwp']
            # Copy scalar statistics directly
            for key in ['f0_mean', 'f0_std', 'f0_range', 'vad_ratio', 'jitter', 'shimmer']:
                if key in pwp:
                    stats[key] = pwp[key]
        
        return stats
    
    def process_speaker_data(self, data_loader: DepressionDataLoader, 
                             speaker_id: str, task: str = 'reading') -> Dict[str, Any]:
        """
        Process all audio data for a speaker.
        
        Args:
            data_loader: Initialized data loader
            speaker_id: ID of the speaker to process
            task: 'reading' or 'interview'
            
        Returns:
            Dictionary with extracted features and statistics
        """
        # Get speaker data
        speakers = data_loader.group_files_by_speaker(task=task)
        if speaker_id not in speakers:
            raise ValueError(f"Speaker {speaker_id} not found in {task} task")
        
        speaker_data = speakers[speaker_id]
        speaker_info = speaker_data['info']
        
        # Initialize result
        result = {
            'speaker_id': speaker_id,
            'is_patient': speaker_info.is_patient,
            'features': {},
            'statistics': {}
        }
        
        # Process main audio files
        for file_path in speaker_data['files']:
            # Extract features
            features = self.extract_features(file_path)
            file_name = os.path.basename(file_path)
            result['features'][file_name] = features
            
            # Extract global statistics
            stats = self.extract_global_statistics(features)
            result['statistics'][file_name] = stats
        
        # Process clips if available (interview task)
        if 'clips' in speaker_data and task == 'interview':
            result['clips'] = {}
            for clip_path in speaker_data['clips']:
                # Extract features
                features = self.extract_features(clip_path)
                clip_name = os.path.basename(clip_path)
                result['clips'][clip_name] = {
                    'features': features,
                    'statistics': self.extract_global_statistics(features)
                }
        
        return result
    
    def process_all_speakers(self, data_loader: DepressionDataLoader, 
                            task: str = 'reading', save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all speakers and extract features.
        
        Args:
            data_loader: Initialized data loader
            task: 'reading' or 'interview'
            save_dir: Directory to save results
            
        Returns:
            Dictionary with extracted features for all speakers
        """
        # Get all speakers
        speakers = data_loader.group_files_by_speaker(task=task)
        
        # Initialize result
        all_results = {}
        
        # Process each speaker
        for speaker_id in tqdm(speakers.keys(), desc=f"Processing {task} speakers"):
            try:
                speaker_result = self.process_speaker_data(data_loader, speaker_id, task)
                all_results[speaker_id] = speaker_result
                
                # Save individual speaker result if save_dir is provided
                if save_dir:
                    save_path = Path(save_dir) / f"{task}_{speaker_id}_features.npy"
                    np.save(save_path, speaker_result)
            except Exception as e:
                print(f"Error processing speaker {speaker_id}: {e}")
        
        # Save combined results if save_dir is provided
        if save_dir:
            save_path = Path(save_dir) / f"{task}_all_features.npy"
            np.save(save_path, all_results)
        
        return all_results
    
    def extract_clusterable_features(self, all_results: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features in a format suitable for clustering.
        
        Args:
            all_results: Dictionary with extracted features for all speakers
            
        Returns:
            Tuple of (feature_matrix, speaker_ids)
        """
        # Initialize lists to store features and corresponding speaker IDs
        feature_vectors = []
        speaker_ids = []
        
        # Process each speaker
        for speaker_id, speaker_data in all_results.items():
            # Get the first file's statistics (assuming at least one file per speaker)
            if not speaker_data['statistics']:
                continue
            
            first_file = next(iter(speaker_data['statistics']))
            file_stats = speaker_data['statistics'][first_file]
            
            # Create a single feature vector for this speaker
            # Combine various statistics into a single vector
            vector = np.concatenate([
                file_stats.get('mfcc_mean', np.array([])),
                file_stats.get('mfcc_std', np.array([])),
                # Add more features as needed
                np.array([
                    file_stats.get('f0_mean', 0),
                    file_stats.get('f0_std', 0),
                    file_stats.get('vad_ratio', 0),
                    file_stats.get('jitter', 0),
                    file_stats.get('shimmer', 0)
                ])
            ])
            
            # Add to lists
            feature_vectors.append(vector)
            speaker_ids.append(speaker_id)
        
        # Convert list of vectors to matrix
        feature_matrix = np.vstack(feature_vectors)
        
        return feature_matrix, speaker_ids
    
    def visualize_mfcc(self, mfcc: np.ndarray, sr: int = 16000, hop_length: Optional[int] = None) -> plt.Figure:
        """
        Visualize MFCC features.
        
        Args:
            mfcc: MFCC features
            sr: Sample rate
            hop_length: Hop length for time axis
            
        Returns:
            Matplotlib figure
        """
        hop_length = hop_length or self.config.hop_length
        
        # Create figure
        plt.figure(figsize=(10, 4))
        
        # Display MFCCs
        librosa.display.specshow(
            mfcc[:self.config.n_mfcc],  # Take only base MFCCs, not deltas
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel'
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_logmel(self, logmel: np.ndarray, sr: int = 16000, hop_length: Optional[int] = None) -> plt.Figure:
        """
        Visualize log-Mel spectrogram.
        
        Args:
            logmel: Log-Mel spectrogram
            sr: Sample rate
            hop_length: Hop length for time axis
            
        Returns:
            Matplotlib figure
        """
        hop_length = hop_length or self.config.hop_length
        
        # Create figure
        plt.figure(figsize=(10, 4))
        
        # Display log-Mel spectrogram
        librosa.display.specshow(
            logmel,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel'
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-Mel Spectrogram')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_pitch(self, pwp: Dict[str, Any], sr: int = 16000, hop_length: Optional[int] = None) -> plt.Figure:
        """
        Visualize pitch-related features.
        
        Args:
            pwp: Dictionary with pitch-related features
            sr: Sample rate
            hop_length: Hop length for time axis
            
        Returns:
            Matplotlib figure
        """
        hop_length = hop_length or self.config.hop_length
        
        # Create figure with subplots
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Time axis in seconds
        times = librosa.times_like(pwp['f0'], sr=sr, hop_length=hop_length)
        
        # Plot F0 contour
        ax[0].plot(times, pwp['f0'], label='F0', color='blue')
        ax[0].set_ylabel('Frequency (Hz)')
        ax[0].set_title('Fundamental Frequency (F0)')
        ax[0].grid(True)
        
        # Plot voiced/unvoiced flag
        ax[1].plot(times, pwp['voiced_flag'], label='Voiced', color='green')
        ax[1].set_ylabel('Voiced Flag')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_title('Voice Activity Detection')
        ax[1].grid(True)
        
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    from src.data_loader import DepressionDataLoader
    
    # Initialize data loader
    data_loader = DepressionDataLoader("./data")
    
    # Initialize feature extractor with default configuration
    feature_extractor = FeatureExtractor()
    
    # Get list of speaker IDs
    reading_speakers = data_loader.group_files_by_speaker(task='reading')
    speaker_ids = list(reading_speakers.keys())
    
    if speaker_ids:
        # Process first speaker as example
        speaker_id = speaker_ids[0]
        print(f"Processing speaker: {speaker_id}")
        
        speaker_data = reading_speakers[speaker_id]
        if speaker_data['files']:
            # Get first audio file
            audio_file = speaker_data['files'][0]
            print(f"Processing file: {audio_file}")
            
            # Extract features
            features = feature_extractor.extract_features(audio_file)
            
            # Print feature shapes
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, np.ndarray):
                    print(f"{feature_name} shape: {feature_data.shape}")
                elif isinstance(feature_data, dict):
                    print(f"{feature_name} keys: {list(feature_data.keys())}")
            
            # Extract global statistics
            stats = feature_extractor.extract_global_statistics(features)
            print(f"Global statistics keys: {list(stats.keys())}")
            
            # Visualize features
            feature_extractor.visualize_mfcc(features['mfcc'])
            plt.savefig("mfcc_example.png")
            
            feature_extractor.visualize_logmel(features['logmel'])
            plt.savefig("logmel_example.png")
            
            feature_extractor.visualize_pitch(features['pwp'])
            plt.savefig("pitch_example.png")
            
            print("Example visualizations saved to current directory")