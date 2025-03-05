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

        # Layer normalization for each input vector
        self.norm1 = nn.LayerNorm(input_dim1)
        self.norm2 = nn.LayerNorm(input_dim2)

        # self.leaky_relu = nn.LeakyReLU(0.1)

        # Encoder - compresses to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(self.total_dim // 2, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
        # Decoder - expands from bottleneck to output dimension
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, self.total_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(self.total_dim // 2, self.total_dim),
        )

        # # Reconstruction networks
        # self.reconstruct_vec1 = nn.Linear(self.total_dim, input_dim1)
        # self.reconstruct_vec2 = nn.Linear(self.total_dim, input_dim2)
    
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

        # Normalize inputs
        norm_vec1 = self.norm1(vec1)
        norm_vec2 = self.norm2(vec2)
        
        # Concatenate input vectors
        combined_input = torch.cat([vec1, vec2], dim=-1)

        # Compress
        compressed = self.encoder(combined_input)
        
        # Reconstruct
        reconstructed = self.decoder(compressed)
        
        # Split reconstructed vector back into two
        reconstructed_vec1 = reconstructed[:, :vec1.shape[1]]
        reconstructed_vec2 = reconstructed[:, vec1.shape[1]:]
        
        return reconstructed_vec1, reconstructed_vec2, compressed