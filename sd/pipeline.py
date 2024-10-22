import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH=512
HEIGHT=512
LATENTS_WIDTH= WIDTH//8
LATENTS_HEIGHT= HEIGHT//8

def rescale(x,old_range:tuple[int,int],new_range:tuple[int,int],clamp:bool=False):
    old_min,old_max=old_range
    new_min,new_max=new_range
    x=(x-old_min)/(old_max-old_min)
    x=(x*(new_max-new_min))+new_min
    if clamp:
        x=x.clamp(new_min,new_max)
    return x

def get_time_embedding(timestep:int)->torch.Tensor:
    #160 is the number of frequencies used in the time embedding
    freqs=torch.pow(10000, -torch.arange(0,160,dtype=torch.float32)/160)
    x=torch.tensor([timestep],dtype=torch.float32)[:,None]*freqs[None]
    return torch.cat([torch.cos(x),torch.sin(x)],dim=-1)

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
        timesteps=tqdm(sampler.timesteps):
        for i,timestep in enumerate(timesteps):
            # (1,320)
            time_embedding=get_time_embedding(timestep).to(device)
            # bz,4,h/8,w/8
            model_input=latents
            if do_cfg:
                # (bz,4,h/8,w/8) -> (2*bz,4,h/8,w/8)
                model_input=model_input.repeat(2,1,1,1)
            model_output= diffusion(model_input,context,time_embedding)
            if do_cfg:
                # (2*bz,4,h/8,w/8) -> (bz,4,h/8,w/8)
                output_cond,output_uncond=torch.chunk(model_output,2,dim=0)
                model_output=(output_cond-output_uncond)*cfg_scale + output_uncond
            else:
                model_output=model_output
            # remove noise predicted by the model (model_output) from the latents
            latents=sampler.step(timestep,latents,model_output)
        
        to_idle(diffusion)

        decoder=models["decoder"]
        decoder.to(device)

        images=decoder(latents)
        images=rescale(images,(-1,1),(0,255),clamp=True)
        # b,c,h,w -> b,h,w,c
        images=images.permute(0,2,3,1)
        images=images.detach().cpu().numpy()
        images=images.astype(np.uint8)
        images=images[0]
        return images
