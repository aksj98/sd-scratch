import torch
from torch import nn
import numpy as np

class DDPMSampler:
    def __init__(self,generator:torch.Generator,
                 num_training_steps=1000,
                 beta_start=0.00085,
                 beta_end=0.0120):
        #beta indicates the variance in the noise with each of the steps
        self.betas=torch.linspace(beta_start**0.5,beta_end **0.5,num_training_steps,dtype=torch.float32) **2
        self.alpha= 1.0- self.betas 
        self.alpha_cumprod=torch.cumprod(self.alphas,0)
        self.one = torch.tensor(1.0)
        self.generator=generator
        self.num_training_steps=num_training_steps
        self.timesteps=torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self,num_inference_steps=50):
        self.num_inference_steps=num_inference_steps
        step_ratio=self.num_training_steps//self.num_inference_steps
        self.timesteps=torch.tensor((np.arange(0,num_inference_steps)*step_ratio)[::-1].copy().astype(np.int64))

    def add_noise(self,original_samples:torch.FloatTensor,timesteps:torch.IntTensor)-> torch.FloatTensor:
        alpha_cumprod=self.alpha_cumprod[timesteps].to(device=original_samples.device,dtype=original_samples.dtype)
        timesteps=timesteps.to(original_samples.device)

        sqrt_alpha_cumprod=alpha_cumprod.sqrt()
        sqrt_alpha_cumprod= sqrt_alpha_cumprod.flatten()
        while len(sqrt_alpha_cumprod.shape)<len(original_samples.shape):
            sqrt_alpha_cumprod=sqrt_alpha_cumprod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_cumprod=((1-alpha_cumprod).sqrt()).flatten() #stdv
        while len(sqrt_one_minus_alpha_cumprod.shape)<len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod=sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        noise=torch.randn(original_samples.shape,generator=self.generator,device=original_samples.device,dtype=original_samples.dtype)
        #mean+stdv*noise 
        # eq. 4 of the ddpm paper
        noisy_samples= (sqrt_alpha_cumprod*original_samples)+sqrt_one_minus_alpha_cumprod*noise
        return noisy_samples
    def _get_previous_timestep(self,t:int)->int:
        prev_t= timestep - (self.num_training_steps//self.num_inference_steps)
        return prev_t
    def step(self,timestep: int,latents:torch.Tensor,model_output:torch.Tensor)->torch.Tensor:
        t=timestep
        prev_t=self._get_previous_timestep(t)
        
