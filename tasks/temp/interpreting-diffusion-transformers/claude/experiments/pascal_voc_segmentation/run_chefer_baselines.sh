
# python -u run_single_class_split_experiment.py \
#     --segmentation_model CheferLRP \
#     --device cuda:1 \
#     --image_save_dir results/segmentation_predictions/chefer_lrp \
#     > results/logs/chefer_lrp

# python -u run_single_class_split_experiment.py \
#     --segmentation_model CheferRollout \
#     --device cuda:1 \
#     --image_save_dir results/segmentation_predictions/chefer_rollout \
#     > results/logs/chefer_rollout

# python -u run_single_class_split_experiment.py \
#     --segmentation_model CheferLastLayerAttention \
#     --device cuda:1 \
#     --image_save_dir results/segmentation_predictions/chefer_last_layer_attention \
#     > results/logs/chefer_last_layer_attention

python -u run_single_class_split_experiment.py \
    --segmentation_model CheferAttentionGradCAM \
    --device cuda:1 \
    --image_save_dir results/segmentation_predictions/chefer_attention_gradcam \
    > results/logs/chefer_attention_gradcam

# python -u run_single_class_split_experiment.py \
#     --segmentation_model CheferTransformerAttribution \
#     --device cuda:1 \
#     --image_save_dir results/segmentation_predictions/chefer_transformer_attribution \
#     > results/logs/chefer_transformer_attribution

# python -u run_single_class_split_experiment.py \
#     --segmentation_model CheferFullLRP \
#     --device cuda:1 \
#     --image_save_dir results/segmentation_predictions/chefer_full_lrp \
#     > results/logs/chefer_full_lrp

# python -u run_single_class_split_experiment.py \
#     --segmentation_model CheferLastLayerLRP \
#     --device cuda:1 \
#     --image_save_dir results/segmentation_predictions/chefer_last_layer_lrp \
#     > results/logs/chefer_last_layer_lrp
