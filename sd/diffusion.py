import torch
from torch import nn
from attention import SelfAttention, CrossAttention
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self,d_model:int):
        super().__init__()
        self.linear_1=nn.Linear(d_model,4*d_model)
        self.linear_2=nn.Linear(4*d_model,4*d_model)

    def forward(self,time:torch.Tensor)->torch.Tensor:
        time=self.linear_1(time)
        time=F.silu(time)
        time=self.linear_2(time)
        return time 

class SwitchSequential(nn.Module):
    def forward(self,x:torch.Tensor,context:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x=layer(x,context)
            elif isinstance(layer,UNET_ResidualBlock):
                x=layer(x,time)
            else:
                x=layer(x)
        return x
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders=nn.Module([
            SwitchSequential(
                nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSequential(UNET_residualBlock(320,320),UNET_AttentionBlock(8,40)),
        ])

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding=TimeEmbedding(320)
        self.unet=UNET()
        self.final=UNET_OutputLayer(320,4)
    
    def forward(self,latent:torch.Tensor,context: torch.Tensor,time:torch.Tensor)->torch.Tensor:
        # latent: (bz,4,h/8,w/8)
        #context: (bz,seq_len,d_embed)
        # time: (1,320)
        # (1,320) -> (1,1280)
        time=self.time_embedding(time)
        # (bz,4,h/8,w/8) -> (bz,320,h/8,w/8)
        output=self.unet(latent,context,time)
        #(bz,320,h/8,w/8) -> (bz,4,h/8,w/8)
        output=self.final(output)
        return output