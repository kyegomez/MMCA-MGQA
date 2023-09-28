[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Multi-Modal Casual Multi-Grouped Query Attention
Experiments around using Multi-Modal Casual Attention with Multi-Grouped Query Attention


# Appreciation
* Lucidrains
* Agorians


# Install
`pip install mmmgqa`

# Usage
```python
import torch 
from mmca_mgqa.attention import SimpleMMCA

# Define the dimensions
dim = 512
head = 8
seq_len = 10
batch_size = 32

#attn
attn = SimpleMMCA(dim=dim, heads=head)

#random tokens
v = torch.randn(batch_size, seq_len, dim)
t = torch.randn(batch_size, seq_len, dim)

#pass the tokens throught attn
tokens = attn(v, t)

print(tokens)
```

# Architecture

# Todo


# License
MIT
