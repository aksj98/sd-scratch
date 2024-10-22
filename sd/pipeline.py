import torch
import numpy as np
from tqdm import tdqm
from ddpm import DDPMSampler

WIDTH=512
HEIGHT=512
LATENTS_WIDTH= WIDTH//8
LATENTS_HEIGHT= HEIGHT//8

def generate(prompt:str,uncond_prompt:str,input_image:Optional[PIL.Image.Image]=None, 
             strength:int=0.8,do_cfg:bool=True, cfg_scale:float=7.5,sampler_name:str="ddpm",
             n_inference_steps:int=50,models={},seed=None,device=None,idle_device=None,tokenizer=None)->PIL.Image.Image:
    
    with torch.no_grad():
        if not (0<strength<=1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle:lambda x:x.to(idle_device)
        else:
            to_idle: lambda x:x

        generator=torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        
        clip= models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert prompt to tokens
            cond_tokens= tokenizer.batch_encode_plus([prompt],return_tensors="pt",padding="max_length",max_length=77).input_ids
            #(bz,seq_len)
            cond_tokens=torch.tensor(cond_tokens,dtype=torch.long,device=device)
            # (bz,seq_len) -> (bz,seq_len,77)
            cond_context=clip(cond_tokens)

            uncond_tokens= tokenizer.batch_encode_plus([uncond_prompt],return_tensors="pt",padding="max_length",max_length=77).input_ids
            uncond_tokens=torch.tensor(uncond_tokens,dtype=torch.long,device=device)
            uncond_context=clip(uncond_tokens)

            # (2,seq_len,dim) -> 2,77,768

            context=torch.cat([cond_context,uncond_context])
        else:
            cond_tokens=tokenizer.batch_encode_plus([prompt],return_tensors="pt",padding="max_length",max_length=77).input_ids
            cond_tokens=torch.tensor(cond_tokens,dtype=torch.long,device=device)
            # (1,77,768)
            context=clip(cond_tokens)

        #offload clip to out of GPU
        to_idle(clip)
        
        if sampler_name=="ddpm":
            sampler=DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} not supported")
        
        latents_shape=(1,4,LATENTS_HEIGHT,LATENTS_WIDTH)
        if input_image:
            encoder=models["encoder"]
            encoder.to(device)
            input_image_tensor=torch.tensor(np.array(input_image.resize((WIDTH,HEIGHT))),dtype=torch.float32,device=device)
            
            input_image_tensor=rescale(input_image_tensor,(0,255),(-1,1))
            input_image_tensor=input_image_tensor.unsqueeze(0)
            # h,w,c-> b,h,w,c
            # b,c,h,w 
            input_image_tensor=input_image_tensor.permute(0,3,1,2)
            # encoder noise is a random sampling variable and isn't actually noise.
            encoder_noise=torch.randn(latents_shape,generator=generator,device=device)
            latents=encoder(input_image_tensor,encoder_noise)

            sampler.set_strength(strength=strength)
            latents=sampler.add_noise(latents,sampler.timesteps[0])

            to_idle(encoder)
        else:
            #text-> image
            latents=torch.randn(latents_shape,generator=generator,device=device)
        
        diffusion=models["diffusion"]
        diffusion.to(device)
        # timesteps are taken from 1000 to 0, if 50 steps are done, then 1000, 980 and so on till 0
        sampler.set_inference_steps(n_inference_steps)
        latents=sampler.sample(context,latents,generator)

        decoder=models["decoder"]
        decoder.to(device)

        image=decoder(latents)