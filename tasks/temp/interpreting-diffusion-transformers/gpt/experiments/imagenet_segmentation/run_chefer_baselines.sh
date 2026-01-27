python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model CheferLRP \
    --image_save_dir results/segmentation_predictions/chefer_baselines/chefer_lrp \
    > results/logs/chefer_baselines/chefer_lrp.log

python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model CheferRollout \
    --image_save_dir results/segmentation_predictions/chefer_baselines/chefer_rollout \
    > results/logs/chefer_baselines/chefer_rollout.log

python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model CheferLastLayerAttention \
    --image_save_dir results/segmentation_predictions/chefer_baselines/chefer_last_layer_attention \
    > results/logs/chefer_baselines/chefer_last_layer_attention.log

python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model CheferAttentionGradCAM \
    --image_save_dir results/segmentation_predictions/chefer_baselines/chefer_attention_gradcam \
    > results/logs/chefer_baselines/chefer_attention_gradcam.log

python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model CheferTransformerAttribution \
    --image_save_dir results/segmentation_predictions/chefer_baselines/chefer_transformer_attribution \
    > results/logs/chefer_baselines/chefer_transformer_attribution.log

python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model CheferFullLRP \
    --image_save_dir results/segmentation_predictions/chefer_baselines/chefer_full_lrp \
    > results/logs/chefer_baselines/chefer_full_lrp.log

python -u run_experiment.py \
    --device cuda:1 \
    --segmentation_model CheferLastLayerLRP \
    --image_save_dir results/segmentation_predictions/chefer_baselines/chefer_last_layer_lrp \
    > results/logs/chefer_baselines/chefer_last_layer_lrp.log