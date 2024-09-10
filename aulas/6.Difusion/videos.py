from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import numpy as np

loaded_model = DiffusionPipeline.from_pretrained('videos')

prompt = "Superman dancing in the rain"
num_inference_steps = 15
num_frames = 16 #padrão 16

video_frames = loaded_model(prompt=prompt,num_inference_steps=num_inference_steps, num_frames=num_frames).frames
video_frames = video_frames.squeeze(0)
video_frames_list = [np.array(frame) for frame in video_frames]
export_to_video(video_frames_list, "superman2.mp4")