python src/stage1_sample.py \
  --config /coc/flash7/bli678/projects/egowm/external/RAE/configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image /coc/flash7/bli678/projects/egowm/data/egoverse/processed_wm/put_cup_on_saucer_rl2_lab_scene_1_recording_1_processed/data/chunk-000/episode_000000.parquet \
  --output outputs/DINOv2-B_512.png

# python src/stage1_sample.py \
#   --config /coc/flash7/bli678/projects/egowm/external/RAE/configs/stage1/pretrained/DINOv2-B.yaml \
#   --image /coc/flash7/bli678/projects/egowm/data/egoverse/processed_wm/put_cup_on_saucer_rl2_lab_scene_1_recording_1_processed/data/chunk-000/episode_000000.parquet \
#   --output outputs/DINOv2-B.png

# python src/stage1_sample.py \
#   --config /coc/flash7/bli678/projects/egowm/external/RAE/configs/stage1/pretrained/DINOv2-B_512.yaml \
#   --image assets/pixabay_cat.png \
#   --output outputs/DINOv2-B_512.png