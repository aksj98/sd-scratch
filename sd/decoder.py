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
    

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            # (bz,512,h/8,w/8) -> (bz,512,h/8,w/8)
            VAE_ResidualBlock(512,512),
            # (bz,512,h/8,w/8) -> (bz,512,h/4,w/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            # (bz,512,h/4,w/4) -> (bz,512,h/2,w/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            # (bz,256,h/2,w/2) -> (bz,256,h,w)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.GroupNorm(32,128),
            nn.SiLU(),
            nn.Conv2d(128,3,kernel_size=3,padding=1),
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: (bz,4,h/8,w/8)
        x/=0.18215
        for module in self:
            x=module(x)
        return x

