import torch
import imageio
from diffusers import TextToVideoZeroPipeline

def generate_video(prompt):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = TextToVideoZeroPipeline.from_pretrained(model_id).to("cpu")
    result = pipe(prompt=prompt, video_length=20).images
    result = [(r *   255).astype("uint8") for r in result]
    video_path = "video.mp4"
    imageio.mimsave(video_path, result, fps=4)
    return video_path    
