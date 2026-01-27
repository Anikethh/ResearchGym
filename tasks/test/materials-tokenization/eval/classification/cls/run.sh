model_save_dir=model/
preds_save_dir=preds/
cache_dir=../.cache/
export CUDA_VISIBLE_DEVICES=1


# Define the seed list
seed_list="42 43 44"

for model_name in wp PickyBPE sage; do
    echo $model_name
    # Pass the seed list to the script
    python -u cls.py --model_name $model_name --model_save_dir $model_save_dir --preds_save_dir $preds_save_dir --cache_dir $cache_dir --seeds $seed_list
done

