import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,n_head:int,d_embed:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()

        self.n_head=n_head
        self.d_embed=d_embed
        self.d_head=d_embed//n_head
        self.scale=d_embed**-0.5

        self.in_proj=nn.Linear(d_embed,d_embed*3,bias=in_proj_bias)
        self.out_proj=nn.Linear(d_embed,d_embed,bias=out_proj_bias)
    def forward(self,x:torch.Tensor,causal_mask:bool=False)->torch.Tensor:
        # x: (bz,seq_len,d_embed)
        bz,seq_len,d_embed=x.shape
        q,k,v=self.in_proj(x).chunk(3,dim=-1)
        # (bz,seq_len,d_embed) -> (bz,seq_len,n_head,d_head) -> (bz,n_head,seq_len,d_head)
        q=q.view(bz,seq_len,self.n_head,self.d_head).transpose(1,2)
        k=k.view(bz,seq_len,self.n_head,self.d_head).transpose(1,2)
        v=v.view(bz,seq_len,self.n_head,self.d_head).transpose(1,2)

        # bz,h,seq_len,seq_len
        qk=q@k.transpose(-1,-2)
        qk=qk*self.scale
        if causal_mask:
            qk=qk.masked_fill(~torch.tril(torch.ones_like(qk)),float('-inf'))
        attn=F.softmax(qk,dim=-1)

        # bz,h,seq_len,d_head
        attn=attn@v
        # bz,seq_len,h,d_head
        attn=attn.transpose(1,2)
        # bz,seq_len,d_embed
        attn=attn.reshape(bz,seq_len,d_embed)

        output=self.out_proj(attn)
        return output

class CrossAttention(nn.Module):
    def __init__(self,n_head:int,d_embed:int,d_context:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.n_head=n_head
        self.d_embed=d_embed
        self.d_head=d_embed//n_head
        self.scale=d_embed**-0.5
        self.q_proj=nn.Linear(d_embed,d_embed,bias=in_proj_bias)
        self.kv_proj=nn.Linear(d_context,d_embed*2,bias=in_proj_bias)
        self.out_proj=nn.Linear(d_embed,d_embed,bias=out_proj_bias)
    
    def forward(self,x:torch.Tensor,context:torch.Tensor,causal_mask:bool=False)->torch.Tensor:
        # x: (bz,seq_len,d_embed)
        # context: (bz,seq_len,d_context)
        bz,seq_len,d_embed=x.shape
        q=self.q_proj(x)
        kv=self.kv_proj(context)
        k,v=kv.chunk(2,dim=-1)
        # (bz,seq_len,d_embed) -> (bz,seq_len,n_head,d_head) -> (bz,n_head,seq_len,d_head)
        q=q.view(bz,-1,self.n_head,self.d_head).transpose(1,2)
        k=k.view(bz,-1,self.n_head,self.d_head).transpose(1,2)
        v=v.view(bz,-1,self.n_head,self.d_head).transpose(1,2)

        # bz,h,seq_len,seq_len
        qk=q@k.transpose(-1,-2)
        qk=qk*self.scale
        if causal_mask:
            qk=qk.masked_fill(~torch.tril(torch.ones_like(qk)),float('-inf'))
        attn=F.softmax(qk,dim=-1)

        # bz,h,seq_len,d_head
        attn=attn@v
        # bz,seq_len,h,d_head
        attn=attn.transpose(1,2)
        # bz,seq_len,d_embed
        attn=attn.reshape(bz,seq_len,d_embed)

        output=self.out_proj(attn)
        return output


