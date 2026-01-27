
# python -u run_single_class_split_experiment.py \
#     --device cuda:2 \
#     --segmentation_model DAAMSDXL \
#     --image_save_dir results/segmentation_predictions/daam_sdxl \
#     > results/logs/daam_sdxl.log

python -u run_single_class_split_experiment.py \
    --device cuda:2 \
    --segmentation_model DAAMSDXL \
    --num_samples 5 \
    --image_save_dir results/segmentation_predictions/daam_sdxl_5_sample \
    > results/logs/daam_sdxl_5_sample.log