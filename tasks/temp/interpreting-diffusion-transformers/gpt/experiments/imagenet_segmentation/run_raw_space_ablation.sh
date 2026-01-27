# 
python -u run_experiment.py \
    --device cuda:1 \
    --num_samples 1 \
    --num_steps 4 \
    --noise_timestep 1 \
    --segmentation_model RawOutputSpace \
    --concept_cross_attention \
    --concept_self_attention \
    --normalize_concepts \
    --softmax \
    --image_save_dir results/segmentation_predictions/raw_space_ablation/raw_output_softmax_normalized_concepts \
    > results/logs/raw_space_ablation/raw_output_normed_softmax.log

python -u run_experiment.py \
    --device cuda:1 \
    --num_samples 1 \
    --num_steps 4 \
    --noise_timestep 1 \
    --segmentation_model RawOutputSpace \
    --concept_cross_attention \
    --concept_self_attention \
    --normalize_concepts \
    --image_save_dir results_link/raw_space_ablation/raw_output_normalized_concepts \
    > results/logs/raw_space_ablation/raw_output_normed_no_softmax.log
