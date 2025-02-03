import os
import gc

from typing import Union, Literal, List, Dict
import string
import numpy as np
import json
from tqdm import tqdm

from praatio import textgrid
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import AutoProcessor, AutoModel, AutoTokenizer

AUDIO_MODELS = {
   'wav2vec2': "facebook/wav2vec2-large-960h-lv60",
   'hubert': "facebook/hubert-large-ls960",
}

class AudioTextDataset(Dataset):
    def __init__(
        self, 
        dataset_dir: str, 
        cache_dir: str, 
        audio_model_name: str = 'wav2vec2', 
        text_model_name: str = 'gpt2', 
        split: str = 'test', 
        min_words: int = 4,
        max_words: int = 60, 
        preload_audio: bool = False,
        sorted: bool = False,
        buffer_missing_samples: bool = False,
    ):
        super().__init__()
        
        # Set up directory paths
        self.dataset_dir = dataset_dir
        self.audio_dir = os.path.join(dataset_dir, 'audio', split)
        self.textgrid_dir = os.path.join(dataset_dir, 'textgrids', split)
        self.prosody_dir = os.path.join(dataset_dir, 'prosody', split)

        # The audio embeddings are based on how the text tokenizer splits the data
        self.cache_dir = os.path.join(cache_dir, f'{audio_model_name}-{text_model_name}', split)

        # Store split and filenames & keep track of failed samples
        self.split = split
        self.file_names = sorted(os.listdir(self.textgrid_dir))

        # Set filters for dataset samples --> enforces a minimum and maximum number of words
        self.buffer_missing_samples = buffer_missing_samples
        self.min_words = min_words
        self.max_words = max_words
        self.failed_samples_count = 0
        self.edge_case_count = 0
        
        # We cache metadata within the cache directory
        self.metadata_path = os.path.join(self.cache_dir, 'metadata.json')
        self.error_metadata_path = os.path.join(self.cache_dir, 'error_metadata.json')

        # Cacheing information
        self.preload_audio = preload_audio
        self.audio_cache = {}
        
        # Determine if GPU is available
        self.audio_model_name = audio_model_name
        self.text_model_name = text_model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Make directory for caching embeddings
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Load or create metadata
        self.metadata = self._load_metadata(self.metadata_path)
        self.error_metadata = self._load_metadata(self.error_metadata_path)

        # Preload audio if requested
        if self.preload_audio:
            self._preload_audio_tensors()

    def _load_metadata(self, path):
        # Load or create metadata
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            return []

    def _save_metadata(self, path, data):
        """Save both main metadata and error metadata to their respective files."""
        # Save main metadata
        with open(path, 'w') as f:
            json.dump(data, f)

    def _preload_audio_tensors(self):
        """Preload all audio tensors into memory"""
        print("Preloading audio tensors...", flush=True)
        for item in tqdm(self.metadata):
            tensor_path = item['audio_tensor_path']
            self.audio_cache[tensor_path] = torch.load(tensor_path, map_location=self.device)
            
    def _process_single_file(self, file_name: str) -> Dict:
        """Process a single file"""
        cache_name = file_name.replace(".TextGrid", ".pt")
        textgrid_path = os.path.join(self.textgrid_dir, file_name)
        audio_path = os.path.join(self.audio_dir, file_name.replace(".TextGrid", ".wav"))
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        # Load audio and extract word segments
        waveform, sample_rate = load_audio(audio_path)
        words = parse_textgrid(textgrid_path)

        # Filter if there were no words, or if less/more than number of requested words
        if not words:
            return None
        elif len(words) < self.min_words or len(words) > self.max_words:
            return None
        
        return self._process_inputs(words, waveform, sample_rate, cache_path)
    
    def _initialize_models(self):

        print (f'Initializing audio & tokenization models...', flush=True)

        # Audio tokenization process
        self.processor = AutoProcessor.from_pretrained(AUDIO_MODELS[self.audio_model_name])
        self.audio_model = AutoModel.from_pretrained(AUDIO_MODELS[self.audio_model_name])
        
        # Move models to GPU if available
        self.audio_model = self.audio_model.to(self.device)

        # Text tokenization process
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True, add_prefix_space=True)
        
    def _remove_models(self):
        print (f'Removing audio & text tokenization models...', flush=True)
         # Move models off GPU and remove from memory
        self.audio_model = self.audio_model.to('cpu')
        del self.processor
        del self.audio_model
        del self.text_tokenizer

    def _cleanup_text_files(self):
        """Load text tokens and attention mask from files, then delete the files."""
        for item in self.metadata:
            if 'text_tokens_path' in item and 'text_attention_mask_path' in item:
                text_tokens_path = item['text_tokens_path']
                text_attention_mask_path = item['text_attention_mask_path']

                try:
                    item['text_tokens'] = torch.load(text_tokens_path) #.tolist()
                    item['text_attention_mask'] = torch.load(text_attention_mask_path) #.tolist()
                    os.remove(text_tokens_path)
                    os.remove(text_attention_mask_path)
                    del item['text_tokens_path']
                    del item['text_attention_mask_path']
                except FileNotFoundError as e:
                    print(f"Warning: File not found for {text_tokens_path} or {text_attention_mask_path}. Skipping cleanup.")

    def preprocess_data(self, force_reprocess=False):
        """
        Preprocess all data and save audio tensors to cache.
        Args:
            force_reprocess (bool): If True, reprocess all files even if they exist in metadata
        """

        # Load error metadata
        error_files = set(self.error_metadata)

        # Determine which files to process
        if not force_reprocess and self.metadata:
            print("Metadata already exists. Processing only new files...", flush=True)
            existing_files = {os.path.basename(item['audio_tensor_path']) for item in self.metadata}
            files_to_process = [
                f for f in self.file_names
                if f.replace(".TextGrid", ".pt") not in existing_files and f not in error_files
            ]
        else:
            self.metadata = []
            self.error_metadata = []
            files_to_process = self.file_names

        # Exit if no files to process
        if not files_to_process:
            print("No new files to process", flush=True)
            return
            
        # Initialize models if needed
        self._initialize_models()
        print(f"Processing {len(files_to_process)} files...", flush=True)

        # Track new files that cause errors during processing
        new_error_files = []

        for file_name in tqdm(files_to_process):

            result = self._process_single_file(file_name)

            if result:
                self.metadata.append(result)
                # self._update_metadata(result)
            elif self.buffer_missing_samples:
                self.metadata.append({
                    'text': [],
                    'text_tokens': [],
                    'text_attention_mask': [],
                    'audio_tensor_path': [],
                })
            else:
                new_error_files.append(file_name)
                self.failed_samples_count += 1

            gc.collect()  # Free up memory
        
        print(f"Split {self.split} failed samples: {self.failed_samples_count}", flush=True)

        # Update error metadata
        if new_error_files:
            error_files.update(new_error_files)
            self.error_metadata = list(error_files)
            self._save_metadata(self.error_metadata_path, self.error_metadata)

        # Clean up text tokens and attention mask files
        self._cleanup_text_files()
        
        # Save metadata
        self._save_metadata(self.metadata_path, self.metadata)

        self._remove_models()

    @torch.no_grad()  # Disable gradient computation for inference
    def _process_inputs(self, words: List[Dict], waveform: np.ndarray, 
                       sample_rate: int, cache_path: str) -> Dict:
        # Join the words together into a sentence
        text = " ".join([word['text'] for word in words])

        # Define paths for text tokens and attention mask
        text_tokens_path = cache_path.replace(".pt", "_text_tokens.pt")
        text_attention_mask_path = cache_path.replace(".pt", "_text_attention_mask.pt")

        # If any of the paths do not exist
        text_paths_not_exist = [not os.path.exists(path) for path in [text_tokens_path, text_attention_mask_path]]
        
        # Paths don't exist
        if text_paths_not_exist:
            # Tokenize the words if files don't exist
            text_tokens = self.text_tokenizer(text)
            
            # Save text tokens and attention mask to files
            torch.save(text_tokens['input_ids'], text_tokens_path)
            torch.save(text_tokens['attention_mask'], text_attention_mask_path)

        # Check if audio tensor already exists
        if os.path.exists(cache_path):
            return {
                'text': text,
                'text_tokens_path': text_tokens_path,
                'text_attention_mask_path': text_attention_mask_path,
                'audio_tensor_path': cache_path
            }

        # Process audio
        word_ids, token_counts = np.unique(text_tokens.word_ids(), return_counts=True)

        # There are weird edge cases with things like "20th"
        if (len(word_ids) != len(words)):
            self.edge_case_count += 1
            print (f"Edge case #{self.edge_case_count}", flush=True)
            return None

        segments = []

        for word, idx, n_tokens in zip(words, word_ids, token_counts):
            if n_tokens > 1:
                ratios = torch.tensor([len(x) for x in self.text_tokenizer.batch_decode(
                    text_tokens['input_ids'][idx:idx+n_tokens])])
                ratios = ratios / ratios.sum()
                word_segments = extract_word_segment(waveform, sample_rate, 
                                                  word["start"], word["end"], ratios=ratios)
            else:
                word_segments = extract_word_segment(waveform, sample_rate, 
                                                  word["start"], word["end"])
            segments.extend(word_segments)
        
        # Process all segments simultaneously
        features = self.processor(segments, sampling_rate=sample_rate, 
                                padding=True, return_attention_mask=True, 
                                return_tensors="pt")
        
        # Move features to GPU if available
        features = {k: v.to(self.device) for k, v in features.items()}

        audio_inputs = self.audio_model(**features).last_hidden_state

        # Get the attention mask for the hidden states
        attention_mask = self.audio_model._get_feature_vector_attention_mask(
            audio_inputs.shape[1], 
            features['attention_mask']
        )

        # Perform pooling over the embeddings while accounting for attention mask
        audio_inputs = pool_embeddings(audio_inputs, attention_mask)
        
        # Move back to CPU for saving
        audio_inputs = audio_inputs.cpu()
        
        # Save the audio inputs to cache
        torch.save(audio_inputs, cache_path)

        return {
            'text': text,
            'text_tokens_path': text_tokens_path,
            'text_attention_mask_path': text_attention_mask_path,
            'audio_tensor_path': cache_path
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self.metadata[idx]
        
        # Convert lists back to tensors
        text_tokens = torch.tensor(data['text_tokens'])
        text_attention_mask = torch.tensor(data['text_attention_mask'])
        
        # Load audio tensor from cache or memory
        if self.preload_audio:
            audio_inputs = self.audio_cache[data['audio_tensor_path']]
        else:
            audio_inputs = torch.load(data['audio_tensor_path'])
        
        return {
            'text': data['text'],
            'text_tokens': text_tokens,
            'text_attention_mask': text_attention_mask,
            'audio_inputs': audio_inputs
        }

######################################################
############## Utility for the dataset ###############
######################################################

def python_remove_punctuation(
    input_text: Union[str, List[str]]
) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.translate(str.maketrans("", "", string.punctuation))
    elif isinstance(input_text, list):
        return [python_remove_punctuation(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")
    

def parse_textgrid(file_path):

    # things to remove from the textgrid (indicates laughing, chewing, pauses etc)
    REMOVE_CHARACTERS = ['sp', 'br', 'lg', 'cg', 'ls', 'ns', 'sl', 'ig',
                         '{sp}', '{br}', '{lg}', '{cg}', '{ls}', '{ns}', '{sl}', '{ig}', 
                         'pause', '[bracketed]', '<unk>'
                         ]

    tg = textgrid.openTextgrid(file_path, False)
    words = []

    tier = 'words' if 'words' in tg.tierNames else 'word'
    
    for interval in tg.getTier(tier):
        if interval.label.lower() in REMOVE_CHARACTERS:
            continue
        
        words.append({
            "text": python_remove_punctuation(interval.label),
            "start": interval.start,
            "end": interval.end
        })
    return words


######################################################
################# Prosody utilities ##################
######################################################

def process_wavelet_file(filename: str) -> List[Dict[str, Any]]:
    """
    Processes a wavelet prosody toolkit .prom file and returns a list of words.
    Output format is a list of individual words and their prosody values
        [
            {'text': 'word',
            'prominence': float,  # From prominence strength
            'boundary': float,  # From boundary strength
            'start': float,
            'end': float},
        ]
    """
    words = []

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
                
                prom_strength = float(prom_strength)
                bound_strength = float(bound_strength)
                
                # Add word to current utterance
                word = {
                    "text": unit,
                    "prominence": prom_strength if prom_strength >= 0 else 0, # adding to align with Helsinki dataset
                    "boundary": bound_strength if bound_strength >= 0 else 0,
                    "start": float(start_time),
                    "end": float(end_time)
                }

                words.append(word)

    return words

def interpolate_prosody_values(current_value: float, next_value: float, num_tokens: int) -> List[float]:
    """
    Interpolate prosody values across multiple tokens.
    
    Args:
        current_value: Prosody value for the current word
        next_value: Prosody value for the next word
        num_tokens: Number of tokens to interpolate across
        
    Returns:
        List of interpolated values
    """
    if num_tokens == 1:
        return [current_value]
    
    step = (next_value - current_value) / num_tokens
    return [current_value + (step * i) for i in range(num_tokens)]

######################################################
################# Audio utilities ####################
######################################################

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Average channels if 2 channeled
    if waveform.shape[0] == 2:
        waveform = waveform.mean(0).unsqueeze(0)
    
    return waveform.numpy(), sample_rate

def extract_word_segment(waveform, sample_rate, onset, offset, ratios=None):
    """
    Extract a segment of the waveform between onset and offset, divided into parts based on the specified ratios.

    Args:
        waveform (torch.Tensor): The input waveform tensor of shape (channels, samples).
        sample_rate (int): The sample rate of the waveform.
        onset (float): The start time of the segment in seconds.
        offset (float): The end time of the segment in seconds.
        ratios (List[float]): A list of ratios to divide the segment into. Defaults to [1.0] (no splitting).

    Returns:
        List[torch.Tensor]: A list of waveform segments, each of shape (channels, samples * ratio).
    """
    if ratios is None:
        ratios = [1.0]  # Default to no splitting

    # Convert onset and offset to sample indices
    start_sample = int(onset * sample_rate)
    end_sample = int(offset * sample_rate)
    
    # Extract the full segment
    full_segment = waveform[:, start_sample:end_sample]
    total_samples = full_segment.shape[-1]
    
    # Calculate the number of samples for each segment based on the ratios
    segment_samples = [int(ratio * total_samples) for ratio in ratios]
    
    # Adjust the last segment to account for any rounding errors
    segment_samples[-1] = total_samples - sum(segment_samples[:-1])
    
    # Split the full segment into parts based on the ratios
    segments = []
    start = 0
    for samples in segment_samples:
        end = start + samples
        segment = full_segment[:, start:end].squeeze()
        segments.append(segment)
        start = end
    
    return segments

def pool_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor = None,
    pool_type: Literal["mean", "max"] = "mean",
    normalize: bool = True
) -> torch.Tensor:
    """
    Pool transformer embeddings using mean or max pooling.
    
    Args:
        embeddings: Tensor of shape (batch_size, sequence_length, hidden_size)
        attention_mask: Optional boolean mask of shape (batch_size, sequence_length)
        pool_type: Type of pooling to use ("mean" or "max")
        normalize: Whether to L2-normalize the output embeddings
        
    Returns:
        Pooled embeddings tensor of shape (batch_size, hidden_size)
    """
    if attention_mask is not None:
        # Expand mask to match embedding dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        # Zero out padding tokens
        embeddings = embeddings * mask_expanded
    
    if pool_type == "mean":
        if attention_mask is not None:
            # Calculate mean only over non-padding tokens
            sum_embeddings = torch.sum(embeddings, dim=1)
            # Add small epsilon to avoid division by zero
            token_counts = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1e-9)
            pooled = sum_embeddings / token_counts
        else:
            pooled = torch.mean(embeddings, dim=1)
            
    elif pool_type == "max":
        if attention_mask is not None:
            # Set padding tokens to large negative value before max
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            embeddings = embeddings.masked_fill(~mask_expanded.bool(), float('-inf'))
        pooled = torch.max(embeddings, dim=1)[0]
    
    else:
        raise ValueError("pool_type must be either 'mean' or 'max'")
        
    if normalize:
        pooled = F.normalize(pooled, p=2, dim=-1)
        
    return pooled