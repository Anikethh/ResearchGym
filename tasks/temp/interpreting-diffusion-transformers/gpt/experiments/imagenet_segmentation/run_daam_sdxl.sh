
# python -u run_experiment.py \
#     --device cuda:3 \
#     --segmentation_model DAAMSDXL \
#     --image_save_dir results/segmentation_predictions/daam_sdxl \
#     > results/logs/daam_sdxl.log

python -u run_experiment.py \
    --device cuda:3 \
    --segmentation_model DAAMSDXL \
    --num_samples 5 \
    --image_save_dir results/segmentation_predictions/daam_sdxl_5_sample_new \
    > results/logs/daam_sdxl_5_sample_new.log