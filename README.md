[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Multi-Modal Casual Multi-Grouped Query Attention
Experiments around using Multi-Modal Casual Attention with Multi-Grouped Query Attention

# Appreciation
* Lucidrains
* Agorians


# Install
`pip install mmqqa`

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
---

# Architectural Overview and Analysis of Multi-Modal Causal Attention

The Multi-Modal Causal Attention (MMCA) mechanism is a novel approach to multi-modal learning that combines the strengths of causal attention and cross attention. It is designed to handle both visual and textual data, making it particularly useful for tasks that involve both types of data, such as image captioning, visual question answering, and multi-modal translation.

The MMCA mechanism is unique in its use of MultiGrouped Query Attention (MGQA), a variant of the attention mechanism that allows for more flexible and efficient attention computations. This report provides an in-depth analysis of the MMCA mechanism, focusing on its architecture, operation, and potential benefits for multi-modal learning.

---

## Architecture

The MMCA mechanism consists of three main components: a MGQA mechanism for visual tokens, a MGQA mechanism for textual tokens, and a cross-attention mechanism that allows textual tokens to attend to visual tokens.

```
+-----------------+     +-----------------+     +-----------------+
| Visual Features | --> | Visual MGQA     | --> | Visual Attention|
|                 |     |                 |     | Output          |
+-----------------+     +-----------------+     +-----------------+

+-----------------+     +-----------------+     +-----------------+     +-----------------+
| Textual Features| --> | Textual MGQA    | --> | Textual MGQA    | --> | Textual Attention|
|                 |     |                 |     | Output          |     | Output          |
+-----------------+     +-----------------+     +-----------------+     +-----------------+

+-----------------+     +-----------------+     +-----------------+
| Textual MGQA    | --> | Cross-Attention | --> | Cross-Attention |
| Output + Visual |     | with Visual     |     | Output          |
| Attention Output|     | Attention Output|     |                 |
+-----------------+     +-----------------+     +-----------------+

```
----

## How It Works

The MMCA mechanism works by first applying MGQA to the visual and textual features separately. The MGQA mechanism is a variant of the attention mechanism that allows for more flexible and efficient attention computations. It works by dividing the queries into multiple groups and computing the attention for each group separately. This allows the model to capture different types of dependencies within the data, which can help to improve performance.

For visual tokens, the MGQA mechanism is sufficient because visual tokens are already fully encoded in a bidirectional manner and do not need further attention from other visual tokens or the beginning of textual tokens.

For textual tokens, however, the MGQA mechanism is combined with a cross-attention mechanism that allows textual tokens to attend to visual tokens. This is based on the intuition that the attention weight for one modality may affect the other modality. For instance, a textual token may pay more attention to textual information than visual information. Therefore, if the attention weight matrix is normalized across both modalities, the attention score for visual tokens might be very small.

The outputs of the MGQA and cross-attention mechanisms for the textual tokens are then combined to produce the final textual attention output. This combined attention output captures both the dependencies within the text and the dependencies between the text and the image, which can help to improve the performance of the model on multi-modal tasks.

---

## How It Accelerates Multi-Modal Learning

The MMCA mechanism can accelerate multi-modal learning in several ways:

1.  Efficient Use of Computational Resources: By using MGQA, the MMCA mechanism can make more efficient use of computational resources. This is because MGQA allows for more flexible and efficient attention computations, which can help to reduce the computational cost of the model.

2.  Improved Data Efficiency: The MMCA mechanism can improve data efficiency by allowing textual tokens to attend to visual tokens. This can help to align visual features with textual features, which can improve the performance of the model on multi-modal tasks.

3.  Flexibility: The MMCA mechanism is flexible and can be easily adapted to different tasks and data types. For instance, it can be used with different types of MGQA and cross-attention mechanisms, and it can be combined with other techniques, such as pretraining, to further improve performance.

4.  Scalability: The MMCA mechanism is scalable and can handle large amounts of data and complex tasks. This is because it uses a linear attention mechanism, which has a time complexity that is linear in the sequence length, making it suitable for long sequences and large datasets.


to finally conclude the Multi-Modal Causal Attention (MMCA) mechanism is a promising approach to multi-modal learning that combines the strengths of causal attention and cross attention. By using MultiGrouped Query Attention (MGQA), it allows for more flexible and efficient attention computations, which can help to improve the performance of the model on


# License
MIT
