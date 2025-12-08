#!/bin/bash
# Automatically detect number of GPUs, or use NPROC_PER_NODE env var if set
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
echo "Using ${NPROC_PER_NODE} GPUs for distributed sampling."

# torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} \
#   src/stage1_sample_ddp.py \
#   --config /coc/flash7/bli678/projects/egowm/external/RAE/configs/stage1/pretrained/DINOv2-B_512.yaml \
#   --data-path /coc/flash7/bli678/projects/egowm/data/egoverse/processed_wm/put_cup_on_saucer_rl2_lab_scene_1_recording_1_processed/data/chunk-000/episode_000000.parquet \
#   --sample-dir recon_samples \
#   --image-size 256

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} \
  src/stage1_sample_ddp.py \
  --config /coc/flash7/bli678/projects/egowm/external/RAE/configs/stage1/pretrained/DINOv2-B.yaml \
  --data-path /coc/flash7/bli678/projects/egowm/data/egoverse/processed_wm/put_cup_on_saucer_rl2_lab_scene_1_recording_1_processed/data/chunk-000/episode_000000.parquet \
  --sample-dir recon_samples \
  --image-size 256