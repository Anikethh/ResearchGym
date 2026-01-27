
# Run cross and self
python -u run_experiment.py \
    --device cuda:0 \
    --num_samples 5 \
    --model_name flux-schnell \
    --num_steps 4 \
    --noise_timestep 2 \
    --segmentation_model RawOutputSpace \
    --concept_cross_attention \
    --concept_self_attention \
    --apply_blur \
    --normalize_concepts \
    --softmax \
    --image_save_dir results/segmentation_predictions/output_space_multi_step_exclude_last_layer_with_blur/ \
    > results/logs/output_space_multi_step_exclude_last_layer_with_blur.log

# # Run cross only
# python -u run_experiment.py \
#     --device cuda:1 \
#     --num_samples 1 \
#     --num_steps 4 \
#     --noise_timestep 1 \
#     --segmentation_model RawOutputSpace \
#     --concept_cross_attention \
#     --normalize_concepts \
#     --image_save_dir results/segmentation_predictions/cross_self_ablation/cross_only \
#     > results/logs/cross_self_ablation/cross_only.log

# # Run self only
# python -u run_experiment.py \
#     --segmentation_model RawOutputSpace \
#     --num_samples 1 \
#     --num_steps 4 \
#     --noise_timestep 1 \
#     --concept_self_attention \
#     --device cuda:1 \
#     --image_save_dir results/segmentation_predictions/cross_self_ablation/self_only \
#     > results/logs/cross_self_ablation/self_only.log

# # Run neither cross nor self
# python -u run_experiment.py \
#     --device cuda:1 \
#     --num_samples 1 \
#     --num_steps 4 \
#     --noise_timestep 1 \
#     --segmentation_model RawOutputSpace \
#     --image_save_dir results/segmentation_predictions/cross_self_ablation/neither_cross_or_self \
#     > results/logs/cross_self_ablation/neither_cross_or_self.log