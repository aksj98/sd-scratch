import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__()

        self.attention=SelfAttention(1,channels)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: (bz,channels,h,w)
        residue=x
        n,c,h,w=x.shape
        # (bz,channels,h,w) -> (bz,channels,h*w)
        x=x.view(n,c,h*w)
        # (bz,channels,h*w) -> (bz,h*w,channels)
        x=x.transpose(-1,-2)
        x=self.attention(x)
        # (bz,h*w,channels) -> (bz,channels,h*w)
        x=x.transpose(-1,-2)
        # (bz,channels,h*w) -> (bz,channels,h,w)
        x=x.view(n,c,h,w)

        return x+residue

class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels

        self.norm_1=nn.GroupNorm(32,in_channels)
        self.conv_1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

        self.norm_2=nn.GroupNorm(32,out_channels)
        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels==out_channels:
            self.residual=nn.Identity()
        else:
            self.residual=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: (bz,in_channels,h,w)
        residue=x
        x=self.norm_1(x)
        x=F.silu(x)
        x=self.conv_1(x)

        x=self.norm_2(x)
        x=F.silu(x)
        x=self.conv_2(x)

        return x+self.residual(residue)
