import runpod
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
import os

import warnings
warnings.filterwarnings("ignore")

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

network_volumn_path = Path('/runpod-volume/')
model_file_name = os.getenv("MODEL_FILE")
model_file_path = network_volumn_path / model_file_name

if model_file_name: 
    pipe = StableDiffusionPipeline.from_single_file(model_file_path, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe = pipe.to(device)

def pil_to_base64(image: Image.Image):
    # Convert the image to a byte array
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    
    # Encode the byte array to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input.get('prompt')
    if prompt is None: raise Exception('Prompt cannot be None')
    image = pipe(prompt).images[0]  
        
    image.save("result.png")

    return {"image_base64": pil_to_base64(image)}


runpod.serverless.start({"handler": handler})