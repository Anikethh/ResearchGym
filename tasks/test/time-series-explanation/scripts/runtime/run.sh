wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=8


# testbs=1
for testbs in 1
do
    for cv in 0 1 2 3 4
    do
        for top in 100
        do
            for num_segments in 50
            do
                for min_seg_len in 10
                do
                    for max_seg_len in 48
                    do
                        explainer_list="integrated_gradients_base_abs gradientshap_abs deeplift_abs occlusion augmented_occlusion lime"
                        # explainer_list="our"
                        # explainer_list="gradientshap_abs"
                        for explainer in ${explainer_list}; do
                            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main_runtime.py \
                                --model_type state \
                                --explainers $explainer \
                                --data mimic3 \
                                --fold $cv \
                                --testbs $testbs \
                                --areas 0.2 \
                                --top $top \
                                --num_segments $num_segments \
                                --min_seg_len $min_seg_len \
                                --max_seg_len $max_seg_len \
                                --output-file runtime_state_mimic3_${cv}_${top}_runtime_batch_size${testbs}.csv \
                                --device cuda:0 \
                                2>&1 &
                            wait_n
                            i=$((i + 1))
                        done
                    done
                done
            done
        done
    done
done
