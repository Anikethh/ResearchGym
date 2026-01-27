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

data_list="wafer freezer boiler epilepsy PAM mimic3"
for data in ${data_list}; do
    explainer_list="timex timex++"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 0
            do
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                    --model_type state \
                    --explainers $explainer \
                    --data $data \
                    --fold $cv \
                    --testbs 50 \
                    --areas 0.1 \
                    --top $top \
                    --output-file state_${data}_${cv}_${top}_results_baseline.csv \
                    --device cuda:0 \
                    2>&1 &
                wait_n
                i=$((i + 1))
            done
        done
    done

    explainer_list="gate_mask"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 0
            do
                for mask_lr in 0.1
                do
                    for lambda_1 in 0.005
                    do
                        for lambda_2 in 0.01
                        do
                            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                                --model_type state \
                                --explainers $explainer \
                                --data $data \
                                --fold $cv \
                                --testbs 10 \
                                --areas 0.1 \
                                --lambda-1 $lambda_1 \
                                --lambda-2 $lambda_2 \
                                --mask_lr $mask_lr \
                                --top $top \
                                --output-file state_${data}_${cv}_${top}_results_baseline.csv \
                                --device cuda:0 \
                                --deterministic \
                                2>&1 &
                            wait_n
                            i=$((i + 1))
                        done
                    done
                done
            done
        done
    done

    explainer_list="extremal_mask"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 0
            do
                for mask_lr in 0.01
                do
                    for lambda_1 in 0.01
                    do
                        for lambda_2 in 10
                        do
                            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                                --model_type state \
                                --explainers $explainer \
                                --data $data \
                                --fold $cv \
                                --testbs 10 \
                                --areas 0.1 \
                                --lambda-1 $lambda_1 \
                                --lambda-2 $lambda_2 \
                                --mask_lr $mask_lr \
                                --top $top \
                                --output-file state_${data}_${cv}_${top}_results_baseline.csv \
                                --device cuda:0 \
                                --deterministic \
                                2>&1 &
                            wait_n
                            i=$((i + 1))
                        done
                    done
                done
            done
        done
    done


    explainer_list="augmented_occlusion integrated_gradients_base_abs gradientshap_abs"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 0
            do
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                    --model_type state \
                    --explainers $explainer \
                    --data $data \
                    --fold $cv \
                    --testbs 10 \
                    --areas 0.1 \
                    --top $top \
                    --output-file state_${data}_${cv}_${top}_results_baseline.csv \
                    --device cuda:0 \
                    2>&1 &
                wait_n
                i=$((i + 1))
            done
        done
    done
done