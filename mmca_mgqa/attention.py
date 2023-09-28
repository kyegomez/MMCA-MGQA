from torch import nn 
from mgqa.attention import MGQA

class SimpleMMCA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        kv_heads = 2, #number of kv heads
        dropout = 0.1,
        causal = False,
        qk_norm = False,
        flash = False
    ):
        super().__init__()
        
        self.visual_self_attn = MGQA(
            dim=dim,
            heads=heads,
            causal=causal,
            kv_heads=kv_heads,
            qk_norm=qk_norm,
            dropout=dropout,
            flash=flash
        )
        
        self.textual_self_attn = MGQA(
            dim=dim,
            heads=heads,
            causal=causal,
            kv_heads=kv_heads,
            qk_norm=qk_norm,
            dropout=dropout,
            flash=flash
        )

        self.cross_attn = MGQA(
            dim=dim,
            heads=heads,
            causal=causal,
            kv_heads=kv_heads,
            qk_norm=qk_norm,
            dropout=dropout,
            flash=flash
        )
    
    def forward(self, v, t):
        # Self attention for visual tokens
        visual_attention_output = self.visual_self_attn(
            v
        )[0]

        # Self attention for textual tokens
        textual_self_attention_output = self.textual_self_attn(
            t
        )[0]

        # Cross attention for textual tokens with visual features
        cross_attention_output = self.cross_attn(
            t + 
            visual_attention_output
        )[0]

        # Combine the self attention and cross attention for textual tokens
        textual_attention_output = textual_self_attention_output + cross_attention_output

        return visual_attention_output, textual_attention_output
    

