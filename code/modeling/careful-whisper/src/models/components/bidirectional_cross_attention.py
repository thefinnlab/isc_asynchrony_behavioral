'''
Modified from https://github.com/lucidrains/bidirectional-cross-attention/tree/main

Added causal cross attention

'''
import torch
import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# Bidirectional Cross Attention - have two sequences attend to each other with 1 attention step
class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        context_dim: int = None, 
        dim_head: int,
        num_heads: int,
        attn_dropout = 0.,
        resid_dropout = 0.,
        talking_heads = False,
        prenorm = False,
        is_causal = False, 
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.is_causal = is_causal

        # Set dimensionality of heads (dim head * num heads) --> lets this be a linear projection
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * num_heads

        # Sequence x projections – qk (query for x, key for xa) 
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity() # optional step for different heads to interact
        self.out_proj = nn.Linear(inner_dim, dim)
        self.resid_dropout = nn.Dropout(resid_dropout)

        # Sequence 2 projections – qk (query for x, key for xa) 
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.context_attn_dropout = nn.Dropout(attn_dropout)
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity() # optional step for different heads to interact
        self.context_out_proj = nn.Linear(inner_dim, context_dim)
        self.context_resid_dropout = nn.Dropout(resid_dropout)


    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        # batch, x - sequence length, xa - sequence length, number of heads
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.num_heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # get similarities

        # qk and context_qk are used directly to compute the similarity matrix (sim), 
        # which represents the dot product between the queries and keys. 
        # This avoids the need to separately project q and k and then compute their dot product.

        # qk represents the query for x and the key for xa (e.g., as in cross attention)
        # context_qk represents the key for x and the query for xa
        # not the symmetry

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # Set masking value
        _fill = -torch.finfo(sim.dtype).max #-torch.inf 

        # relative positional bias, if supplied

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # mask --> this functions as a padding mask
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.zeros((b, i), device = device, dtype = torch.bool)).to(torch.bool)
            context_mask = default(context_mask, torch.zeros((b, j), device = device, dtype = torch.bool))

            # Create the attention mask
            mask = mask.bool()
            context_mask = context_mask.bool()

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') + rearrange(context_mask, 'b j -> b 1 1 j')
            
            # Convert to boolean and fill with small values
            sim = sim.masked_fill(attn_mask, _fill)

        if self.is_causal:
            # Create a mask of seq_length x context_length
            x_mask = torch.zeros(i, j, device=device).fill_(_fill).triu_(1) #.bool().fill_(True).triu_(1)
            context_mask = torch.zeros(j, i, device=device).fill_(_fill).triu_(1) #.fill_(-np.inf).triu_(1)

            # Add needed dimensions to apply to attention
            x_mask = rearrange(x_mask, 'i j -> 1 1 i j')
            context_mask = rearrange(context_mask, 'j i -> 1 1 i j')

            attn = (sim + x_mask).softmax(dim = -1)
            context_attn = (sim + context_mask).softmax(dim = -2)
        else:
            attn = sim.softmax(dim = -1)
            context_attn = sim.softmax(dim = -2)

        # dropouts

        attn = self.attn_dropout(attn)
        context_attn = self.context_attn_dropout(context_attn)

        # talking heads

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates values from src sequence

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # merge heads and combine out
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        # Out projection
        out = self.out_proj(out)
        context_out = self.context_out_proj(context_out)

        # Residual dropout
        out = self.resid_dropout(out)
        context_out = self.context_resid_dropout(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out