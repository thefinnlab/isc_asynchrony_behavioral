import os
from typing import Union, Literal

from typing import List
import string
import numpy as np

from praatio import textgrid

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import AutoProcessor, AutoModel, AutoTokenizer

from concurrent.futures import ThreadPoolExecutor
import librosa

AUDIO_MODELS = {
   'wav2vec2': "facebook/wav2vec2-large-960h-lv60",
   'hubert': "facebook/hubert-large-ls960",
}

class AudioTextDataset(Dataset):
    def __init__(self, audio_dir, textgrid_dir, cache_dir, audio_model_name='wav2vec2', text_model_name='gpt2', split='test'):
        self.audio_dir = os.path.join(audio_dir, split)
        self.textgrid_dir = os.path.join(textgrid_dir, split)
        self.file_names = os.listdir(self.textgrid_dir)
        self.cache_dir = os.path.join(cache_dir, split)
        
        # Make a directory for caching embeddings
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Audio tokenization process
        self.processor = AutoProcessor.from_pretrained(AUDIO_MODELS[audio_model_name])
        self.audio_model = AutoModel.from_pretrained(AUDIO_MODELS[audio_model_name])

        # Text tokenization process
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True, add_prefix_space=True, return_tensors="pt")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        textgrid_path = os.path.join(self.textgrid_dir, file_name)
        audio_path = os.path.join(self.audio_dir, file_name.replace(".TextGrid", ".wav"))
        cache_path = os.path.join(self.cache_dir, file_name.replace(".TextGrid", ".pt"))

        # Load audio and extract word segments
        waveform, sample_rate = load_audio(audio_path)
        words = parse_textgrid(textgrid_path)

        if not words:
            return None
        
        return self._process_inputs(words, waveform, sample_rate, cache_path)
    
    def _process_inputs(self, words, waveform, sample_rate, cache_path):
        # Join the words together into a sentence
        text = " ".join([word['text'] for word in words])

        # Tokenize the words
        text_tokens = self.text_tokenizer(text)

        # If we have the audio inputs cached, load them
        if os.path.exists(cache_path):
            print('Loading from cache')
            audio_inputs = torch.load(cache_path)
        else:
            # Find number of counts for each word (e.g., number of tokens each word is broken into)
            word_ids, token_counts = np.unique(text_tokens.word_ids(), return_counts=True)

            assert (len(word_ids) == len(words))

            segments = []
            
            # Ensure we get the right number of audio segments for each word (e.g., for each token)
            for word, idx, n_tokens in zip(words, word_ids, token_counts):

                # If there is more than one token, we need to divide the audio into segments
                if n_tokens > 1:
                    ratios = torch.tensor([len(x) for x in self.text_tokenizer.batch_decode(text_tokens['input_ids'][idx:idx+n_tokens])])
                    ratios = ratios / ratios.sum()
                    word_segments = extract_word_segment(waveform, sample_rate, word["start"], word["end"], ratios=ratios)
                else:
                    word_segments = extract_word_segment(waveform, sample_rate, word["start"], word["end"])

                segments.extend(word_segments)
            
            # Process all segments simultaneously
            features = self.processor(segments, sampling_rate=sample_rate, padding=True, return_attention_mask=True, return_tensors="pt")

            with torch.no_grad():
                audio_inputs = self.audio_model(**features).last_hidden_state

            # Get the attention mask for the hidden states
            attention_mask = self.audio_model._get_feature_vector_attention_mask(
                audio_inputs.shape[1], 
                features['attention_mask']
            )

            # Perform pooling over the embeddings while accounting for attention mask
            audio_inputs = pool_embeddings(audio_inputs, attention_mask)

            # Save the audio inputs to cache if a cache path is provided
            if cache_path:
                torch.save(audio_inputs, cache_path)

        data = {
            'text': text,
            'text_tokens': text_tokens['input_ids'],
            'text_attention_mask': text_tokens['attention_mask'],
            'audio_inputs': cache_path
        }

        return data
    
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
                         '{sp}', '{br}', '{lg}', '{cg}', '{ls}', '{ns}', '{sl}', '{ig}', 'pause', '[bracketed]']

    tg = textgrid.openTextgrid(file_path, False)
    words = []
    
    for interval in tg.getTier("words"):
        if interval.label in REMOVE_CHARACTERS:
            continue
        
        words.append({
            "text": python_remove_punctuation(interval.label),
            "start": interval.start,
            "end": interval.end
        })
    return words

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
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
