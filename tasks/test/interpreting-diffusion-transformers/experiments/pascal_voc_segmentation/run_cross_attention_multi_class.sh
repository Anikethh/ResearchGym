python -u run_multi_class_seg_experiment.py \
    --segmentation_model RawCrossAttention \
    --device cuda:1 \
    --image_save_dir results/mulit_class_cross_attention \
    > results/logs/multi_class_cross_attention 