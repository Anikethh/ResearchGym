python -u run_single_class_split_experiment.py \
    --device cuda:2 \
    --segmentation_model DINO \
    --num_samples 1 \
    --image_save_dir results/segmentation_predictions/dino_single_class \
    > results/logs/dino_single_class.log