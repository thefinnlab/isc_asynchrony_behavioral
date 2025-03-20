import torch
import torch.nn as nn

class TokenFusionMLP(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim=1024):
        """
        Initialize the autoencoder
        
        Args:
        - input_dim1 (int): Dimension of the first input vector
        - input_dim2 (int): Dimension of the second input vector
        - hidden_dim (int): Dimension of the compressed representation
        """
        super(TokenFusionMLP, self).__init__()
        
        # Total input dimension
        self.total_dim = input_dim1 + input_dim2
        self.hidden_dim = hidden_dim

        # Pass through the encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.input1_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.input2_decoder = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, vec1, vec2):
        """
        Forward pass of the autoencoder
        
        Args:
        - vec1 (torch.Tensor): First input vector
        - vec2 (torch.Tensor): Second input vector
        
        Returns:
        - reconstructed_vec1 (torch.Tensor): Reconstructed first vector
        - reconstructed_vec2 (torch.Tensor): Reconstructed second vector
        - compressed_representation (torch.Tensor): Compressed joint representation
        """
        
        # Concatenate input vectors
        combined_input = torch.cat([vec1, vec2], dim=-1)

        # Compress
        fused = self.encoder(combined_input)

        # Reconstruct
        reconstructed_vec1 = self.input1_decoder(fused)
        reconstructed_vec2 = self.input2_decoder(fused)
        
        return reconstructed_vec1, reconstructed_vec2, fused