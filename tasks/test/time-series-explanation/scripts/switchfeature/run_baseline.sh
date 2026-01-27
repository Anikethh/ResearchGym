wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=4

for cv in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python synthetic/switchstate/main.py \
        --fold $cv \
        --device cuda:0 \
        --output-file switch_feature_${cv}_results_baseline.csv \
        --deterministic \
        2>&1 &
    wait_n
    i=$((i + 1))
done