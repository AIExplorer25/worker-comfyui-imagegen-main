# Stage 1: Base image with common dependencies cuda:12.4.0-runtime-ubuntu22.04  nvidia/cuda:12.4.0-runtime-ubuntu22.04
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv


# Install comfy-cli pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install comfy-cli --system
RUN pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.34 --cuda-version 12.4 --nvidia
#ADD src/requirements.txt ./
#RUN pip install -r requirements.txt
RUN pip install xformers==0.0.29.post3 --no-deps
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    cd ComfyUI_essentials && \
    pip install -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git && \
    cd was-node-suite-comfyui && \
    pip install -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt


RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git && \
    cd ComfyUI-Inspire-Pack && \
    pip install -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/filliptm/ComfyUI_Fill-Nodes.git && \
    cd ComfyUI_Fill-Nodes && \
    pip install -r requirements.txt


RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Extraltodeus/Skimmed_CFG.git

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92.git

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Clybius/ComfyUI-Extra-Samplers.git && \
    cd ComfyUI-Extra-Samplers && \
    pip install -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/rgthree/rgthree-comfy.git




    
# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# install dependencies
RUN uv pip install runpod requests --system


# Add files
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
#ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
#RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

# ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=flux

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/clip_vision models/loras  models/upscale_models 

ADD loras/ ./models/loras/
ADD vae/ ./models/vae/

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN bash -c '\
  if [ "$MODEL_TYPE" = "flux" ]; then \
    wget -O models/clip/ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/resolve/main/ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors && \
    wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
    wget -O models/unet/flux1-dev-Q8_0.gguf https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf && \
    wget -O models/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth \
  elif [ "$MODEL_TYPE" = "sd3" ]; then \
    echo "SD3 selected, skipping downloads (uncomment lines to enable)"; \
  elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
    wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
    wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
    wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
    wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
  elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
    wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
    wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors; \
  elif [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
    wget -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
  fi'

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
