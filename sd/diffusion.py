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

class SwitchSequential(nn.Sequential):
    def forward(self,x:torch.Tensor,context:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x=layer(x,context)
            elif isinstance(layer,UNET_ResidualBlock):
                x=layer(x,time)
            else:
                x=layer(x)
        return x
class UNET_residualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,n_time=1280):
        super().__init__()
        self.groupnorm_1=nn.GroupNorm(32,in_channels)
        self.conv_1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear_time=nn.Linear(n_time,out_channels)
        self.groupnorm_merged=nn.GroupNorm(32,out_channels)
        self.conv_merged=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if in_channels==out_channels:
            self.residual_path=nn.Identity()
        else:
            self.residual_path=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self,x:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        # x= (bz,in_channel,h,w)
        # time= (1,1280)
        residual=x
        x=self.groupnorm_1(x)
        x=F.silu(x)
        x=self.conv_1(x)
        time=F.silu(time)
        time=self.linear_time(time)
        x=x+time.unsqueeze(-1).unsqueeze(-1)
        x=self.groupnorm_merged(x)
        x=F.silu(x)
        x=self.conv_merged(x)
        x=x+self.residual_path(residual)
        return x 
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self,n_head:int,n_embed:int,d_context:int=768):
        super().__init__()
        channels=n_head*n_embed
        self.groupnorm=nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input=nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        self.layernorm_1=nn.LayerNorm(channels)
        self.attention_1=SelfAttention(n_head,channels,in_proj_bias=False)
        self.layernorm_2=nn.LayerNorm(channels)
        self.attention_2=CrossAttention(n_head,channels,d_context,in_proj_bias=False)
        self.layernorm_3=nn.LayerNorm(channels)
        self.linear_geglu_1=nn.Linear(channels,4*channels*2)
        self.linear_geglu_2=nn.Linear(4*channels,channels)

        self.conv_output=nn.Conv2d(channels,channels,kernel_size=1,padding=0)
    def forward(self,x:torch.Tensor,context:torch.Tensor)->torch.Tensor:
        # x -> (bz,channels,h,w)
        # context -> (bz,seq_len,d_context)
        residual_long=x
        x=self.groupnorm(x)
        x=self.conv_input(x)
        n,c,h,w=x.shape
        x=x.view(n,c,h*w).permute(0,2,1)
        residual_short=x
        x=self.layernorm_1(x)
        x=self.attention_1(x)
        x+=residual_short
        # cross_attn
        residual_short=x
        x=self.layernorm_2(x)
        x=self.attention_2(x,context)

        x+=residual_short
        # mlp+gelu
        residual_short=x
        x=self.layernorm_3(x)
        x,gate=self.linear_geglu_1(x).chunk(2,dim=-1)
        x=x*F.gelu(gate)
        x=self.linear_geglu_2(x)
        x+=residual_short
        x=x.transpose(-1,-2)
        x=x.view((n,c,h,w))

        return self.conv_output(x)+residual_long




class Upsample(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.conv=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # (bz,channels,h,w) -> (bz,channels,2h,2w)
        x=F.interpolate(x,scale_factor=2.0,mode="nearest")
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders=nn.Module([
            # (bz,4,h/8,w/8)
            SwitchSequential(
                nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSequential(UNET_residualBlock(320,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_residualBlock(320,320),UNET_AttentionBlock(8,40)),
            # (bz,320,h/16,w/16))
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_residualBlock(320,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_residualBlock(640,640),UNET_AttentionBlock(8,80)),
            # (bz,640,h/32,w/32))
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_residualBlock(640,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residualBlock(1280,1280),UNET_AttentionBlock(8,160)),
            # (bz,1280,h/64,w/64))
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNET_residualBlock(1280,1280)),
            SwitchSequential(UNET_residualBlock(1280,1280)),
        ])

        self.bottneck=SwitchSequential(
            UNET_residualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_residualBlock(1280,1280),
        )

        self.decoders=nn.ModuleList([
            # (bz,2560,h/64,w/64)
            SwitchSequential(UNET_residualBlock(2560,1280)),
            SwitchSequential(UNET_residualBlock(2560,1280)),
            SwitchSequential(UNET_residualBlock(2560,1280),Upsample(1280)),
            SwitchSequential(UNET_residualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residualBlock(1920,1280),UNET_AttentionBlock(8,160),Upsample(1280)),
            SwitchSequential(UNET_residualBlock(1920,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_residualBlock(1280,640),UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_residualBlock(960,640),UNET_AttentionBlock(8,80),Upsample(640)),
            SwitchSequential(UNET_residualBlock(960,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_residualBlock(640,320),UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_residualBlock(640,320),UNET_AttentionBlock(8,40)),
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.groupnorm=nn.GroupNorm(32,in_channels)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # (bz,in_channels,h/8,w/8) -> (bz,out_channels,h/8,w/8)

        x=self.groupnorm(x)
        x=F.silu(x)
        x=self.conv(x)
        return x 

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