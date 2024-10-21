import torch
from torch import nn
from attention import SelfAttention
from torch.nn import functional as F


class CLIPEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_embed:int,seq_len:int):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,d_embed)
        self.position_embedding=nn.Parameter(torch.zeros(seq_len,d_embed))
    
    def forward(self,tokens:torch.LongTensor)->torch.FloatTensor:
        x=self.token_embedding(tokens)
        x+=self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self,n_embd:int,n_head:int):
        super().__init__()
        self.layernorm_1=nn.LayerNorm(n_embd)
        self.attention=SelfAttention(n_head,n_embd)
        self.layernorm_2=nn.LayerNorm(n_embd)
        self.mlp=nn.Sequential(
            nn.Linear(n_embd,n_embd*4),
            nn.GELU(),
            nn.Linear(n_embd*4,n_embd),
        )
    def forward(self,x:torch.FloatTensor)->torch.FloatTensor:
        residual=x
        x=self.layernorm_1(x)
        x=self.attention(x,causal_mask=True)
        x+=residual
        residual=x
        x=self.layernorm_2(x)
        x=self.mlp(x)
        x+=residual
        return x
        

class CLIP(nn.Module):
    def __init__(self):
        self.embedding= CLIPEmbedding(49408,768,77)

        self.layers=nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])

        self.layernorm=nn.LayerNorm(768)
    
    def forward(self,tokens:torch.LongTensor)->torch.FloatTensor:
        tokens=tokens.type(torch.long)
        #bz,seq_len-> bz,seq_len,d_embed
        state=self.embedding(tokens)

        for layer in self.layers:
            state=layer(state)

        state=self.layernorm(state)
        return state

