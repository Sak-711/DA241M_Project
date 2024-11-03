# pip install torch
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute the Scaled Dot-Product Attention
    Args:
        Q: Query matrix of shape (batch_size, num_heads, seq_len, d_k)
        K: Key matrix of shape (batch_size, num_heads, seq_len, d_k)
        V: Value matrix of shape (batch_size, num_heads, seq_len, d_v)
        mask: Optional mask to prevent attention to certain tokens (e.g., padding)
    
    Returns:
        Attention output and attention weights
    """
    d_k = Q.size(-1)  # Key dimension
    # Step 1: Compute QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Step 2: Optionally apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 3: Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 4: Multiply the attention weights by the values
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention mechanism.
        Args:
            d_model: The input/output dimensionality (e.g., 512 for Transformer)
            num_heads: Number of attention heads (e.g., 8)
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query_linear = torch.nn.Linear(d_model, d_model)
        self.key_linear = torch.nn.Linear(d_model, d_model)
        self.value_linear = torch.nn.Linear(d_model, d_model)
        self.out_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Step 1: Linear projections for Q, K, V
        Q = self.query_linear(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_linear(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 2: Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 3: Concatenate heads and pass through output linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out_linear(attn_output)
        
        return output, attn_weights

import numpy as np

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Step 1: Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Step 2: Create a position and divide by a power of 10000
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Step 3: Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Model hyperparameters
d_model = 512  # Dimensionality of the model
num_heads = 8  # Number of attention heads

# Initialize layers
multi_head_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
pos_encoding = PositionalEncoding(d_model=d_model)

# Dummy input: batch_size=64, seq_len=10, embedding_dim=d_model
Q = torch.randn(64, 10, d_model)
K = torch.randn(64, 10, d_model)
V = torch.randn(64, 10, d_model)

# Apply positional encoding
Q = pos_encoding(Q)
K = pos_encoding(K)
V = pos_encoding(V)

# Perform multi-head attention
output, attn_weights = multi_head_attn(Q, K, V)

print("Attention output shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)
