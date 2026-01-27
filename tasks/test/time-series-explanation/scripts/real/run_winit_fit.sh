wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=4


explainer_list="winit fit"
for explainer in ${explainer_list}; do
    for cv in 0 1 2 3 4
    do
        for top in 100
        do
            # when already train feature generator.
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                --model_type state \
                --explainers $explainer \
                --data mimic3 \
                --fold $cv \
                --testbs 500 \
                --areas 0.2 \
                --top $top \
                --skip_train_timex \
                --output-file state_mimic3_${cv}_${top}_results_baseline.csv \
                --device cuda:0 \
                2>&1 &
            wait_n
            i=$((i + 1))

            # With training
            # CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
            #     --model_type state \
            #     --explainers $explainer \
            #     --data mimic3 \
            #     --fold $cv \
            #     --testbs 500 \
            #     --areas 0.2 \
            #     --top $top \
            #     --output-file state_mimic3_${cv}_${top}_results_baseline.csv \
            #     --device cuda:0 \
            #     2>&1 &
            # wait_n
            # i=$((i + 1))
        done
    done
done