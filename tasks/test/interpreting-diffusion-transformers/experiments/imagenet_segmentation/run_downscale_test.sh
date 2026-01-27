# 
python -u run_experiment.py \
    --device cuda:0 \
    --num_samples 1 \
    --num_steps 4 \
    --noise_timestep 1 \
    --segmentation_model RawOutputSpace \
    --concept_cross_attention \
    --concept_self_attention \
    --normalize_concepts \
    --downscale_for_eval \
    --image_save_dir results/segmentation_predictions/downscale_eval \
    > results/logs/downscale_eval.log
