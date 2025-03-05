from typing import Dict, Iterable, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .bidirectional_cross_attention import BidirectionalCrossAttention

def _check_cross_attention(x, y):
    assert not (x and y), \
    "Error: Only one of cross_attention or bidirectional_cross_attention can be True at a time."

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@dataclass
class CarefulWhisperConfig:
    n_vocab: int
    max_length: int
    embed_dim: int
    num_heads: int
    num_layers: int
    pad_token_id: int
    embed_type: str = 'text_tokens'
    embed_dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    dropout: float = 0.1
    init_std: float = 0.02

    #######################################
    ##### Cross attention arguments #######
    #######################################

    token_fusion: bool = False
    cross_attention: bool = True # Allows specification of cross attention or not --> if not, architecture is essentially GPT2
    bidirectional_cross_attention: bool = False
    use_causal_cross_attention: bool = False
    use_text_control: bool = False
    inverse_audio_text: bool = False
    shuffle_context: bool = False

    #######################################
    ##### Context specific arguments ###### 
    #######################################

    context_type: str = 'audio_features' # Key used when accessing context
    context_dim: int = None # Size of context being projected
    context_embed_dim: int = None # What the context dimensions are projected to
    context_embed_dropout: float = None
    context_pos_embed: bool = None

class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_mlp,
        dropout=0.
    ):
        super().__init__()

        # Define the MLP out projection here so as to have the initialization applied
        self.mlp_ln = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(n_mlp, embed_dim)
        
        self.mlp = nn.Sequential(
            self.mlp_ln,
            nn.Linear(embed_dim, n_mlp), 
            nn.GELU(), 
            self.out_proj,
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)

class MultiheadAttentionBlock(nn.Module):
    '''
    Modified OpenAI code to use nn.MultiheadAttention
    However, this wraps the projection heads within the same function
    '''

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0., resid_dropout: float = 0., is_causal: bool = False):
        super().__init__()
        self.num_heads = num_heads

        # self.norm = nn.LayerNorm(embed_dim)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.is_causal = is_causal
        self.resid_dropout = nn.Dropout(resid_dropout)

    def _get_attn_mask(self, x: Tensor, xa: Optional[Tensor] = None):

        if xa is not None:
            # Create a causal mask of shape (length x length)
            attn_mask = torch.zeros(x.shape[1], xa.shape[1])
        #     attn_mask = torch.ones(x.shape[1], xa.shape[1]),
        else:
            attn_mask = torch.zeros(x.shape[1], x.shape[1])

        if self.is_causal:
            attn_mask = attn_mask.fill_(-np.inf).triu_(1)
        
        return attn_mask.to(x.device)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # x = self.norm(x)
        q = self.query(x)

        # Within pytorch, we expect masked values to be -inf and tokens as 0
        # Huggingface is the inverse of this
        attn_mask = self._get_attn_mask(x, xa)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.attn(q, k, v, key_padding_mask=mask, attn_mask=attn_mask, is_causal=self.is_causal)

        # Apply out projection
        wv = self.out_proj(wv)

        # Apply residual dropout
        wv = self.resid_dropout(wv)

        return wv, qk

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dim_head: int = None,
        attn_dropout: float = 0., 
        resid_dropout: float = 0., 
        context_embed_dim: int = None,
        cross_attention: bool = False,
        bidirectional_cross_attention: bool = False,
        use_causal_cross_attention: bool = False,
    ):
        super().__init__()

        self.attn_ln = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttentionBlock(embed_dim, num_heads, attn_dropout=attn_dropout, resid_dropout=resid_dropout, is_causal=True)

        self.cross_attn_ln = nn.LayerNorm(embed_dim) if cross_attention else None
        self.cross_attn = (
            MultiheadAttentionBlock(embed_dim, num_heads, attn_dropout=attn_dropout, resid_dropout=resid_dropout, is_causal=use_causal_cross_attention) if cross_attention else None
        )

        n_mlp = embed_dim * 4
        self.ff = FeedForward(embed_dim=embed_dim, n_mlp=n_mlp, dropout=resid_dropout)

        # If we want bidirectional
        dim_head = default(dim_head, embed_dim // num_heads)

        self.bi_cross_attn = (
            BidirectionalCrossAttention(dim=embed_dim, num_heads=num_heads, dim_head=dim_head, context_dim=context_embed_dim, attn_dropout=attn_dropout, resid_dropout=resid_dropout, prenorm=True, is_causal=use_causal_cross_attention) if bidirectional_cross_attention else None
        )

        # Context MLP
        self.context_ff = (
            FeedForward(embed_dim=embed_dim, n_mlp=n_mlp, dropout=resid_dropout) if bidirectional_cross_attention else None
        )

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        
        # x (usually text) attends to self
        # Applies the following steps:
        #    1. Pre-Attn LayerNorm of hidden_states
        #    2. self attn + attn_dropout
        #    3. out projection of linear layer + resid_dropout
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]

        # x (text) and xa (audio) attend to each other
        if self.bi_cross_attn:
            x, xa = self.bi_cross_attn(x, xa, mask, context_mask, return_attn=False)

            x = x + self.ff(x)
            xa = x + self.context_ff(xa)

            return x, xa

        # x (text) attends to xa (audio))
        elif self.cross_attn:
            # Changing cross_attention to accept a context mask (e.g., masking )
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, mask=mask, kv_cache=kv_cache)[0]
        
        # LayerNorm before MLP + MLP + MLP dropout
        x = x + self.ff(x)

        return x    

class CarefulWhisper(nn.Module):
    def __init__(
        self, 
        config: CarefulWhisperConfig,
    ):
        super().__init__()
        '''
        Adding the following based on GPT2 architecture:
            - Embedding Dropout (following token + pos embedding)
            - LayerNorm before attn mechanism (ln_1 v. ln_2)
            - AttentionDropout --> dropout of attn_weights within MHA 
            - ResidualDropout -->
        '''

        self.config = config

        _check_cross_attention(self.config.cross_attention, self.config.bidirectional_cross_attention)

        self.token_embedding = nn.Embedding(self.config.n_vocab, self.config.embed_dim, self.config.pad_token_id)
        self.positional_embedding = nn.Embedding(self.config.max_length, self.config.embed_dim)

        self.dropout = nn.Dropout(self.config.embed_dropout)

        # Context embedding information --> 
        context_embed_dim = default(self.config.context_embed_dim, self.config.embed_dim)

        if self.config.context_dim:

            # Context projection layer --> not learnable
            self.context_embed = nn.Linear(self.config.context_dim, context_embed_dim, bias=False)
            self.context_embed.weight.requires_grad = False

        else:
            self.context_embed = nn.Identity()

        self.context_positional_embedding = nn.Embedding(self.config.max_length, self.config.embed_dim) if self.config.context_pos_embed else None
        self.context_embed_dropout = nn.Dropout(self.config.embed_dropout) if self.config.context_embed_dropout else nn.Identity()

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    embed_dim = self.config.embed_dim, 
                    num_heads = self.config.num_heads, 
                    attn_dropout = self.config.attn_dropout, 
                    resid_dropout = self.config.resid_dropout, 
                    cross_attention=self.config.cross_attention,
                    bidirectional_cross_attention=self.config.bidirectional_cross_attention,
                    use_causal_cross_attention=self.config.use_causal_cross_attention
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Final layer norm after attention blocks
        self.ln = nn.LayerNorm(self.config.embed_dim)
        
        # Initialize weights
        self.init_std = self.config.init_std
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_initialized", False):
            return
        
        std = self.init_std  # Example standard deviation, can be configurable
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)  # Initialize LayerNorm weights to 1
            module.bias.data.zero_()  # Initialize LayerNorm biases to 0
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)  # Initialize positional embedding


        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/

        for name, p in module.named_parameters():
            if name == "out_proj.weight" or name == "context_out_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(std / np.sqrt(2 * self.config.num_layers)))


        # Note initialization
        module._is_initialized = True
    
    def forward(
        self, 
        x: Tensor, 
        xa: Tensor, 
        mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        return_hidden_states: bool = False,
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        mask : torch.Tensor, shape = (batch_size, max_length)
            The padding mask specified is 1s for tokens 0s for padding --> implicitly converted to pytorch 
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        position_ids = torch.arange(x.shape[-1], dtype=torch.long, device=x.device)

        # Convert mask to pytorch expected format
        mask = torch.where(mask == 1., 0.0, float('-inf'))

        # Embed tokens (text) + add positional_embeddings
        x = (
            self.token_embedding(x)
            + self.positional_embedding(position_ids)
        )

        ####################################
        ######## Context embedding #########
        ####################################

        # Embed if specified into a static embedding (e.g., prosody is 1D but needs to be projected)
        # Otherwise takes the identity
        xa = self.context_embed(xa) 

        # Optional position embedding
        if self.context_positional_embedding:
            xa_position_ids = torch.arange(xa.shape[-2], dtype=torch.long, device=xa.device)
            xa = + xa + self.context_positional_embedding(xa_position_ids)

        ####################################
        ####### Dropouts and fusion ########
        ####################################

        if self.config.token_fusion:
            x = x + xa

        # Embedding dropout
        x = self.dropout(x).to(xa.dtype)

        # Optional context dropout
        xa = self.context_embed_dropout(xa)

        ####################################
        ######## Attention blocks ##########
        ####################################

        for block in self.blocks:
            # Have sequences attend to each other
            if self.config.bidirectional_cross_attention:
                x, xa = block(x, xa, mask=mask, context_mask=context_mask, kv_cache=kv_cache)

            # x attends to self then xa
            elif self.config.cross_attention:
                x = block(x, xa, mask=mask, kv_cache=kv_cache)
            
            # x attends to x (can be twice)
            elif self.config.use_text_control:
                x = block(x, x, mask=mask, kv_cache=kv_cache)
            # normal attention
            else:
                x = block(x, mask=mask, kv_cache=kv_cache)

        # Out normalization
        x = self.ln(x)

        # Tied weights to the token embeddings for the logit calculation
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        
        if return_hidden_states:
            return x, logits
        else:
            return logits