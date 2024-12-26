# Text-to-Video Diffusion Model with Gradio UI

## Overview
This project uses a diffusion model to generate videos from text prompts. It integrates a basic user interface (UI) using **Gradio** to allow users to input text prompts and generate videos interactively.

## Features
- Generate videos based on textual descriptions.
- Adjustable video duration (1 to 10 seconds).
- User-friendly interface for ease of use.

## Requirements
Ensure you have the following libraries installed in your environment:

- `torch`
- `diffusers`
- `gradio`
- `imageio`
- `matplotlib`
- `IPython`

You can install the required libraries using:
```bash
pip install torch diffusers gradio imageio matplotlib
```

## How to Use
This project is designed for use in Google Colab or similar Jupyter-based environments. Follow the steps below to execute the code:

### 1. Import Required Libraries
Run the following code to import necessary libraries:
```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import gradio as gr
```

### 2. Load the Diffusion Model
Load the pre-trained diffusion model:
```python
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
```

### 3. Define the Video Generation Function
This function takes a text prompt and video duration as inputs, generates the video frames, and exports the video file:
```python
def generate_video(prompt, duration):
    video_duration_seconds = duration
    num_frames = video_duration_seconds * 10
    video_frames = pipe(prompt, negative_prompt="low quality", num_inference_steps=25, num_frames=num_frames).frames
    video_frames = video_frames.squeeze(0)  # Remove the batch dimension
    video_path = export_to_video(video_frames)  # Export the video
    return video_path
```

### 4. Set Up the Gradio Interface
Define the user interface with inputs for the text prompt and video duration, and an output area for the generated video:
```python
interface = gr.Interface(
    fn=generate_video,  # The function to generate videos
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your text prompt here"),
        gr.Slider(label="Video Duration (seconds)", minimum=1, maximum=10, step=1, value=3),
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Text-to-Video Diffusion Model",
    description="Generate videos from text prompts using a diffusion model."
)
```

### 5. Launch the Gradio Interface
Run the following code to start the Gradio interface:
```python
interface.launch()
```

### 6. Interact with the UI
- Open the Gradio link generated in the last step.
  ![Alt Text](URL_or_relative_path_to_image)
- Enter your text prompt (e.g., "A girl swinging in a swing in a blossom tree").
- Adjust the video duration slider as needed.
- Click "Submit" to generate and view the video.

## Notes
- Ensure sufficient GPU resources are available when running the model.
- Video generation time depends on the complexity of the prompt and the duration.

## Example Prompt
```
A serene sunset over a calm ocean with gentle waves.
```
Expected output: A short video depicting the described scene.

## Acknowledgments
- This project utilizes the [Diffusers library](https://huggingface.co/docs/diffusers/) for text-to-video generation.
- Gradio is used for building the user interface.

## License
This project is for educational and non-commercial use only. Please refer to the respective libraries’ licenses for more details.

