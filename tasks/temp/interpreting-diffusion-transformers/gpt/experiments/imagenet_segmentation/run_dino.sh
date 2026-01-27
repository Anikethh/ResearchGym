python -u run_experiment.py \
    --device cuda:3 \
    --segmentation_model DINO \
    --num_samples 1 \
    --image_save_dir results/segmentation_predictions/dino_model \
    > results/logs/dino_model.log