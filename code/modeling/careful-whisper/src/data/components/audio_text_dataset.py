import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import os, sys
from typing import Union, Literal, List, Dict, Optional, Any
import numpy as np
import json

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

class AudioTextDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        split: str = 'test',
        metadata_file: str = None,
        token_fusion_method: str = None,
        token_fusion_weights: Optional[List[float]] = None,
        preload: bool = False,  # New preload option
    ):
        super().__init__()

        # Metadata paths
        if not metadata_file:
            metadata_path = os.path.join(data_dir, split, 'metadata.json')
        else:
            metadata_path = os.path.join(data_dir, split, metadata_file)
        
        # Load or create metadata
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata(self.metadata_path)

        self.token_fusion_method = token_fusion_method
        self.token_fusion_weights = token_fusion_weights
        self.preload = preload

        # Preload data if required
        if self.preload:
            preloaded_data = []
            for idx in tqdm(range(len(self.metadata)), desc=f"Preloading {split} data"):
                preloaded_data.append(self._load_data(idx))
        else:
            self.preloaded_data = None

    #############################################
    ############ Metadata functions #############
    #############################################

    def _load_metadata(self, path):
        # Load or create metadata
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            return []
    
    def _load_data(self, idx):
        """Helper function to preload data"""
        data = self.metadata[idx]
        item = {'text': data['text']}
        
        item.update(
            {x: torch.tensor(data[x]) for x in ['text_tokens', 'attention_mask', 'prominence', 'boundary']}
        )
        
        # Load audio features if exists
        if 'audio_features_path' in data:
            item['audio_features'] = torch.load(data['audio_features_path'])
            item['audio_features'] = torch.nan_to_num(item['audio_features'])

         # Load video features if exists
        if 'video_features_path' in data:
            item['video_features'] = torch.load(data['video_features_path'])
            item['video_features'] = torch.nan_to_num(item['video_features'])

        if self.token_fusion_method is not None:
            if self.token_fusion_method == 'average':
                item['audiovisual_features'] = torch.mean(torch.stack([item['audio_features'], item['video_features']]), dim=0)
            elif self.token_fusion_method == 'mlerp':
                item['audiovisual_features'] = mlerp(
                    token_list=[item['audio_features'], item['video_features']],
                    weights=self.token_fusion_weights
                )
            elif self.token_fusion_method == 'summation':
                # take sum and then normalize
                item['audiovisual_features']  = item['audio_features'] + item['video_features']
                item['audiovisual_features'] = item['audiovisual_features'] / torch.norm(item['audiovisual_features'])
        
        return item
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.preload:
            # If preloaded, just return the item directly from the preloaded data
            return self.preloaded_data[idx]
        else:
            # Otherwise, load the data on the fly
            return self._load_data(idx)

def mlerp(token_list, weights=None):
    """
    Implements Maximum Norm Linear Interpolation (MLERP) for token fusion.
    
    MLERP computes the normalized average of tokens, then scales this 
    normalized average using the maximum norm of the individual tokens.
    
    Args:
        token_list (list of torch.Tensor): List of token tensors to merge
                                          Each tensor has shape [N_tokens x N_dim]
    
    Returns:
        torch.Tensor: The merged token tensor with the same shape as inputs

    """
    if weights is None:
        weights = torch.ones(len(token_list))
    else:
        weights = torch.tensor(weights)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Weighted average
    token_list = [w * tokens for w, tokens in zip(weights, token_list)]

    # Stack the tokens along a new dimension
    tokens = torch.stack(token_list, dim=0)  # Shape: [K, N_tokens, N_dim]
    
    # Compute the average
    average = torch.mean(tokens, dim=0)  # Shape: [N_tokens, N_dim]
    
    # Compute the norm of the average
    average_norm = torch.norm(average, dim=-1, keepdim=True)  # Shape: [N_tokens, 1]
    
    # Normalize the average
    normalized_average = average / (average_norm + 1e-8)  # Adding epsilon to avoid division by zero
    
    # Compute norms of all input tokens
    token_norms = torch.norm(tokens, dim=-1)  # Shape: [K, N_tokens]
    
    # Find the maximum norm at each token position
    max_norms, _ = torch.max(token_norms, dim=0, keepdim=False)  # Shape: [N_tokens]
    max_norms = max_norms.unsqueeze(-1)  # Shape: [N_tokens, 1]
    
    # Scale the normalized average by the maximum norm
    merged_tokens = normalized_average * max_norms
    
    return merged_tokens