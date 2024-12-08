import re
import os
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

def process_wavelet_file(filename: str) -> List[Dict[str, Any]]:
    """
    Processes a wavelet prosody toolkit .prom file and returns a list of utterances.
    Output format matches the Helsinki format for compatibility:
        {
        'filename': 'original_file_name',
        'word_labels': [
            {'text': 'word',
            'discrete_prominence': None,  # Not available in wavelet format
            'discrete_word_boundary': None,  # Not available in wavelet format
            'real_prominence': float,  # From prominence strength
            'real_word_boundary': float,  # From boundary strength
            'start_time': float,
            'end_time': float},
        ],
        'text': 'full utterance text'
        }
    """
    utterances = []
    current_utterance = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 6:  # Ensure we have all required columns
                file_name, start_time, end_time, unit, prom_strength, bound_strength = parts[:6]
                
                # Skip bracketed units
                if "[" in unit and "]" in unit:
                    continue
                    
                # Create new utterance if file changes
                if current_utterance is None or current_utterance["filename"] != file_name:
                    if current_utterance is not None:
                        utterances.append(current_utterance)
                    current_utterance = {
                        "filename": file_name,
                        "word_labels": [],
                        "text": "",
                    }

                prom_strength = float(prom_strength)
                bound_strength = float(bound_strength)
                
                # Add word to current utterance
                word = {
                    "text": unit,
                    "discrete_prominence": None,  # Not available in wavelet format
                    "discrete_word_boundary": None,  # Not available in wavelet format
                    "real_prominence": prom_strength if prom_strength >= 0 else 0, # adding to align with Helsinki dataset
                    "real_word_boundary": bound_strength if bound_strength >= 0 else 0,
                    "start_time": float(start_time),
                    "end_time": float(end_time)
                }
                current_utterance["word_labels"].append(word)
                current_utterance["text"] += unit + " "

    # Add final utterance
    if current_utterance is not None:
        utterances.append(current_utterance)

    return utterances

def process_wavelet_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Processes all .prom files in a directory and returns a combined list of utterances.
    
    Args:
        directory: Path to directory containing .prom files
        
    Returns:
        List of utterances from all files in the directory
    """
    all_utterances = []
    directory_path = Path(directory)
    
    # Process all .prom files in the directory
    for file_path in tqdm(directory_path.glob("*.prom")):
        utterances = process_wavelet_file(str(file_path))
        all_utterances.extend(utterances)
        
    return all_utterances

class WaveletProminenceExtractor:
    """
    Extract and access the prominence features from wavelet prosody toolkit output.
    Maintains interface compatibility with HelsinkiProminenceExtractor.
    """

    def __init__(
        self,
        root_dir: str,
        data_dir: str,  # Changed from filename to data_dir
        lowercase: bool = False,
        remove_punctuation: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing the data directories
            data_dir: Name of the directory (e.g., 'train', 'validation', 'test') containing the .prom files
        """
        self.root_dir = root_dir
        self.data_dir = data_dir
        
        # Process all files in the directory
        full_path = os.path.join(root_dir, data_dir)
        self.utterances = process_wavelet_directory(full_path)
        self.length = len(self.utterances)
        
        # Create filename mappings
        self.filename_to_index = {self.get_filename(i): i for i in range(self.length)}
        self.index_to_filename = {i: self.get_filename(i) for i in range(self.length)}

        # Preprocessing functions (keeping interface compatibility)
        if lowercase and remove_punctuation:
            from src.utils.text_processing import python_lowercase_remove_punctuation
            self.preprocess_fct = python_lowercase_remove_punctuation
        elif lowercase:
            from src.utils.text_processing import python_lowercase
            self.preprocess_fct = python_lowercase
        elif remove_punctuation:
            from src.utils.text_processing import python_remove_punctuation
            self.preprocess_fct = python_remove_punctuation
        else:
            self.preprocess_fct = lambda x: x

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.utterances[index]

    def __len__(self) -> int:
        return self.length

    def get_filename(self, index: int) -> str:
        return self.utterances[index]["filename"]

    def get_text(self, index: int) -> str:
        return self.preprocess_fct(self.utterances[index]["text"].strip())

    def get_word_labels(self, index: int) -> List[Dict[str, Any]]:
        return self.utterances[index]["word_labels"]

    def get_real_prominence(self, index: int, min_length: int = 0) -> List[float]:
        return [
            word["real_prominence"]
            for word in self.utterances[index]["word_labels"]
            if len(word["text"]) >= min_length
        ]

    def get_real_word_boundary(self, index: int, min_length: int = 0) -> List[float]:
        return [
            word["real_word_boundary"]
            for word in self.utterances[index]["word_labels"]
            if len(word["text"]) >= min_length
        ]

    def get_all_texts(self, min_length: int = 0) -> List[str]:
        return [
            self.get_text(i)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    def get_all_real_prominence(self, min_length: int = 0) -> List[List[float]]:
        return [
            self.get_real_prominence(i)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    def get_all_real_word_boundary(self, min_length: int = 0) -> List[List[float]]:
        return [
            self.get_real_word_boundary(i)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    # Note: Discrete methods are not implemented since wavelet toolkit doesn't provide discrete labels
    def get_discrete_prominence(self, *args, **kwargs):
        raise NotImplementedError("Discrete prominence not available in wavelet format")

    def get_discrete_word_boundary(self, *args, **kwargs):
        raise NotImplementedError("Discrete word boundary not available in wavelet format")

    def get_all_discrete_prominence(self, *args, **kwargs):
        raise NotImplementedError("Discrete prominence not available in wavelet format")

    def get_all_discrete_word_boundary(self, *args, **kwargs):
        raise NotImplementedError("Discrete word boundary not available in wavelet format")