import os
import re
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any


@dataclass
class SpeakerInfo:
    """Class to hold speaker information parsed from filename."""
    speaker_id: str  # nn_X format
    numeric_id: int  # nn as integer
    is_patient: bool  # True if X is 'P', False if 'C'
    gender: str  # 'M' or 'F'
    age: int  # Age in years
    education_level: int  # 1-4


class DepressionDataLoader:
    """Data loader for the depression detection dataset."""
    
    def __init__(self, data_root: str):
        """
        Initialize the data loader.
        
        Args:
            data_root: Path to the root directory of the dataset
        """
        self.data_root = Path(data_root)
        self.reading_task_dir = self.data_root / "Reading-Task"
        self.interview_task_dir = self.data_root / "Interview-Task"
        self.interview_audio_dir = self.interview_task_dir / "audio"
        self.interview_clip_dir = self.interview_task_dir / "audio_clip"
        
        # File paths for metadata
        self.fold_list_path = self.data_root / "fold-list.csv"
        self.interview_timedata_path = self.data_root / "interview_timedata.csv"
        
        # Load metadata if files exist
        self.fold_data = self._load_fold_data() if os.path.exists(self.fold_list_path) else None
        self.interview_timedata = self._load_interview_timedata() if os.path.exists(self.interview_timedata_path) else None
        
        # Dictionary to cache loaded audio data
        self.audio_cache = {}
    
    def _load_fold_data(self) -> pd.DataFrame:
        """Load fold list data."""
        return pd.read_csv(self.fold_list_path)
    
    def _load_interview_timedata(self) -> pd.DataFrame:
        """Load interview time data."""
        return pd.read_csv(self.interview_timedata_path)
    
    def _parse_filename(self, filename: str) -> SpeakerInfo:
        """
        Parse speaker information from filename based on the naming convention.
        
        Format: nn_XGmm_t.wav
        - nn: unique integer ID (may have leading zero)
        - X: 'P' for patient, 'C' for control
        - G: 'M' for male, 'F' for female
        - mm: age
        - t: education level (1-4)
        
        Returns:
            SpeakerInfo object with parsed information
        """
        # Extract components using regex
        pattern = r"(\d+)_([PC])([MF])(\d+)_(\d)\.wav"
        match = re.match(pattern, filename)
        
        if not match:
            raise ValueError(f"Filename {filename} does not match expected pattern")
        
        numeric_id = int(match.group(1))
        is_patient = match.group(2) == 'P'
        gender = match.group(3)
        age = int(match.group(4))
        education_level = int(match.group(5))
        speaker_id = f"{numeric_id:02d}_{'P' if is_patient else 'C'}"
        
        return SpeakerInfo(
            speaker_id=speaker_id,
            numeric_id=numeric_id,
            is_patient=is_patient,
            gender=gender,
            age=age,
            education_level=education_level
        )
    
    def get_all_reading_task_files(self) -> Dict[str, List[Path]]:
        """
        Get all audio files from the Reading-Task directory.
        
        Returns:
            Dictionary with keys 'HC' and 'PT', each containing a list of file paths
        """
        result = {'HC': [], 'PT': []}
        
        for group in ['HC', 'PT']:
            group_dir = self.reading_task_dir / group
            if not group_dir.exists():
                print(f"Warning: Directory {group_dir} does not exist")
                continue
                
            for file_path in group_dir.glob("*.wav"):
                result[group].append(file_path)
        
        return result
    
    def get_all_interview_audio_files(self) -> Dict[str, List[Path]]:
        """
        Get all audio files from the Interview-Task/audio directory.
        
        Returns:
            Dictionary with keys 'HC' and 'PT', each containing a list of file paths
        """
        result = {'HC': [], 'PT': []}
        
        for group in ['HC', 'PT']:
            group_dir = self.interview_audio_dir / group
            if not group_dir.exists():
                print(f"Warning: Directory {group_dir} does not exist")
                continue
                
            for file_path in group_dir.glob("*.wav"):
                result[group].append(file_path)
        
        return result
    
    def get_speaker_clips(self, speaker_id: str) -> List[Path]:
        """
        Get all audio clips for a specific speaker from Interview-Task/audio_clip.
        
        Args:
            speaker_id: Speaker ID in the format nn_X
            
        Returns:
            List of file paths to the speaker's audio clips
        """
        speaker_dir = self.interview_clip_dir / speaker_id
        
        if not speaker_dir.exists():
            print(f"Warning: Speaker directory {speaker_dir} does not exist")
            return []
            
        return list(speaker_dir.glob("*.wav"))
    
    def get_fold_files(self, fold_num: int) -> Dict[str, List[str]]:
        """
        Get files for a specific fold based on fold-list.csv.
        
        Args:
            fold_num: Fold number (1-5)
            
        Returns:
            Dictionary with keys 'train' and 'test', each containing a list of file IDs
        """
        if self.fold_data is None:
            raise ValueError("Fold data not loaded. Make sure fold-list.csv exists")
            
        # Extract files for the specified fold
        fold_mask = self.fold_data['fold'] == fold_num
        test_files = self.fold_data[fold_mask]['file_id'].tolist()
        train_files = self.fold_data[~fold_mask]['file_id'].tolist()
        
        return {
            'train': train_files,
            'test': test_files
        }
    
    def load_audio(self, file_path: Union[str, Path], sr: Optional[int] = None, force_reload: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa with caching.
        
        Args:
            file_path: Path to the audio file
            sr: Target sample rate (None to use original)
            force_reload: Whether to force reload from disk and ignore cache
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        file_path = str(file_path)
        
        # Check cache first
        cache_key = f"{file_path}_{sr}"
        if not force_reload and cache_key in self.audio_cache:
            return self.audio_cache[cache_key]
        
        # Load audio file
        audio_data, sample_rate = librosa.load(file_path, sr=sr)
        
        # Cache the result
        self.audio_cache[cache_key] = (audio_data, sample_rate)
        
        return audio_data, sample_rate
    
    def group_files_by_speaker(self, task: str = 'reading') -> Dict[str, Dict[str, Any]]:
        """
        Group files by speaker ID and include speaker information.
        
        Args:
            task: 'reading' or 'interview'
            
        Returns:
            Dictionary with speaker_id as keys, containing speaker info and file paths
        """
        if task == 'reading':
            all_files = self.get_all_reading_task_files()
            files = all_files['HC'] + all_files['PT']
        elif task == 'interview':
            all_files = self.get_all_interview_audio_files()
            files = all_files['HC'] + all_files['PT']
        else:
            raise ValueError("Task must be 'reading' or 'interview'")
        
        speakers = {}
        
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                speaker_info = self._parse_filename(filename)
                speaker_id = speaker_info.speaker_id
                
                if speaker_id not in speakers:
                    # Initialize entry with speaker info
                    speakers[speaker_id] = {
                        'info': speaker_info,
                        'files': [],
                    }
                    
                    # Add clips if available for interview task
                    if task == 'interview':
                        clips = self.get_speaker_clips(speaker_id)
                        speakers[speaker_id]['clips'] = clips
                
                # Add the main file to the files list
                speakers[speaker_id]['files'].append(file_path)
                
            except ValueError as e:
                print(f"Error parsing filename {filename}: {e}")
        
        return speakers
    
    def get_data_split(self, train_ratio: float = 0.8, stratify_by_group: bool = True,
                       random_state: int = 42) -> Dict[str, List[str]]:
        """
        Split data into training and testing sets.
        
        Args:
            train_ratio: Ratio of data to use for training
            stratify_by_group: Whether to stratify by patient/control groups
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train' and 'test' keys, each containing a list of speaker IDs
        """
        from sklearn.model_selection import train_test_split
        
        # Get all speakers
        reading_speakers = self.group_files_by_speaker(task='reading')
        
        # Create lists of speaker IDs, stratified by patient/control status
        speaker_ids = list(reading_speakers.keys())
        
        if stratify_by_group:
            # Create stratification labels (patient vs control)
            strat_labels = [reading_speakers[spk_id]['info'].is_patient for spk_id in speaker_ids]
            
            # Split with stratification
            train_ids, test_ids = train_test_split(
                speaker_ids, 
                train_size=train_ratio, 
                stratify=strat_labels,
                random_state=random_state
            )
        else:
            # Split without stratification
            train_ids, test_ids = train_test_split(
                speaker_ids, 
                train_size=train_ratio, 
                random_state=random_state
            )
        
        return {
            'train': train_ids,
            'test': test_ids
        }
    
    def get_labels(self) -> Dict[str, int]:
        """
        Get depression labels for all speakers.
        
        Returns:
            Dictionary with speaker_id as keys and binary labels as values (1 for patient, 0 for control)
        """
        # Combine speakers from both tasks to get a complete set
        reading_speakers = self.group_files_by_speaker(task='reading')
        interview_speakers = self.group_files_by_speaker(task='interview')
        
        all_speakers = {**reading_speakers, **interview_speakers}
        
        # Create labels dictionary
        labels = {}
        for speaker_id, speaker_data in all_speakers.items():
            labels[speaker_id] = 1 if speaker_data['info'].is_patient else 0
            
        return labels


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    data_loader = DepressionDataLoader("./data")
    
    # Get files for Reading Task
    reading_files = data_loader.get_all_reading_task_files()
    print(f"Reading Task - HC: {len(reading_files['HC'])} files, PT: {len(reading_files['PT'])} files")
    
    # Get files for Interview Task
    interview_files = data_loader.get_all_interview_audio_files()
    print(f"Interview Task - HC: {len(interview_files['HC'])} files, PT: {len(interview_files['PT'])} files")
    
    # Group files by speaker for Reading Task
    reading_speakers = data_loader.group_files_by_speaker(task='reading')
    print(f"Reading Task - {len(reading_speakers)} unique speakers")
    
    # Get data split
    split = data_loader.get_data_split(train_ratio=0.8, stratify_by_group=True)
    print(f"Data split - Train: {len(split['train'])} speakers, Test: {len(split['test'])} speakers")
    
    # Get depression labels
    labels = data_loader.get_labels()
    print(f"Labels - {sum(labels.values())} patients, {len(labels) - sum(labels.values())} controls")