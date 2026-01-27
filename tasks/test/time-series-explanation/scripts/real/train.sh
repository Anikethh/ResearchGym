### Train Model

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


# for data in mimic3 boiler epilepsy freezer wafer
for data in freezer wafer mimic3 PAM boiler epilepsy
do
    for cv in 0 1 2 3 4
    do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
            --model_type state \
            --train True \
            --data $data \
            --explainers empty \
            --fold $cv \
            --device cuda:0 \
            2>&1 &
        wait_n
        i=$((i + 1))
    done
done