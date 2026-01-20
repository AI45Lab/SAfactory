#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR=/mnt/shared-storage-user/leishanzhe/pip_wheels
mkdir -p "${TARGET_DIR}"

pip download --dest "${TARGET_DIR}" \
  numpy==1.26.4 \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  av==16.0.1 \
  opencv-python==4.11.0.86 \
  imageio==2.37.2 \
  imageio-ffmpeg==0.3.0 \
  gymnasium==1.2.2 \
  gym==0.26.2 \
  gym3==0.3.3 \
  "shimmy[gym-v21]==0.2.1" \
  ipython==8.37.0 \
  typing==3.7.4.3 \
  wandb==0.23.0 \
  ray==2.51.1 \
  timm==1.0.22 \
  transformers==4.57.1 \
  x-transformers==0.27.1 \
  pyrender==0.1.25 \
  pyglet==1.4.0b1 \
  pytorch-lightning==2.5.6 \
  lightning==2.5.6 \
  cuda-python==12.8.0 \
  albumentations==2.0.8 \
  einops==0.8.1 \
  lmdb==1.7.5 \
  dm-tree==0.1.9 \
  hydra-colorlog==1.2.0 \
  hydra-core==1.3.2 \
  scipy==1.15.3 \
  rich==14.2.0 \
  coloredlogs==15.0.1 \
  daemoniker==0.2.3 \
  lxml==6.0.2 \
  diskcache==5.6.3 \
  Pyro4==4.82 \
  xmltodict==1.0.2 \
  imgui==2.0.0 \
  pyopengl==3.1.0 \
  absl-py==2.3.1 \
  pydantic==2.12.4 \
  tortoise-orm==0.25.1 \
  aiosqlite==0.21.0 \
  pillow==12.0.0 \
  pyyaml==6.0.3 \
  pandas==2.3.3 \
  matplotlib==3.10.7 \
  boto3==1.40.73 \
  openai==2.7.2 \
  fastapi==0.115.5 \
  "uvicorn[standard]==0.32.0" \
  python-dotenv==1.2.1 \
  minecraft-data==3.20.0
