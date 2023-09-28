from torch import nn 
from mgqa.attention import MGQA

class SimpleMMCA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        kv_heads = 2,
        dropout=0.1,
        causal=False,
        qk_norm=False,
        flash=False
    ):
        super().__init__()
        
        self.self_attn = MGQA(
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
        #self attention for visual tokens
        v = self.self_attn(v)[0]

        #cross attention for textual tokens
        t = self.cross_attn(t)[0] + self.cross_attn(t + v)[0]

        return t
    

