#!/bin/bash
# Automatically detect number of GPUs, or use NPROC_PER_NODE env var if set
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
echo "Using ${NPROC_PER_NODE} GPUs for distributed sampling."

SAMPLE_DIR='outputs'

# torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} \
#   src/stage1_sample_ddp.py \
#   --config configs/stage1/pretrained/DINOv2-B_512.yaml \
#   --data-path ${DATA_PATH} \
#   --sample-dir ${SAMPLE_DIR} \
#   --image-size ${IMAGE_SIZE}

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} \
  src/test_decoder.py \
  --config configs/stage1/pretrained/DINOv2-B.yaml \
  --sample-dir ${SAMPLE_DIR} \
  --video-fps 10
