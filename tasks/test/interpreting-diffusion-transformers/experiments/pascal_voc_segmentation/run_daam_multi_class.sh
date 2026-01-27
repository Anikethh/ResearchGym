
python -u run_multi_class_seg_experiment.py \
    --device cuda:2 \
    --segmentation_model DAAMSDXL \
    --num_samples 1 \
    --image_save_dir results/segmentation_predictions/multi_class_daam_sdxl \
    > results/logs/multi_class_daam_sdxl.log