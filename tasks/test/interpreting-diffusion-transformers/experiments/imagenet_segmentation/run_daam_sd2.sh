
# python -u run_experiment.py \
#     --device cuda:3 \
#     --segmentation_model DAAMSDXL \
#     --image_save_dir results/segmentation_predictions/daam_sdxl \
#     > results/logs/daam_sdxl.log

python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model DAAMSD2 \
    --num_samples 1 \
    --image_save_dir results/segmentation_predictions/daam_sd2 \
    > results/logs/daam_sd2.log