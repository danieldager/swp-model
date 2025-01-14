#!/bin/bash

# Define the path to your training script
TRAIN_SCRIPT="./scripts/train_repetition.sh"

# Define arrays for each hyperparameter

fold_ids=(0)
m_types=("lstm")
h_sizes=(64)
n_layers=(1)
l_rates=(0.0005 0.0001)
dropouts=(0.2)
tf_ratios=(0.0)

# Initialize counter for total combinations
total=0

# Nested loops to iterate through all combinations
for fold_id in "${fold_ids[@]}"; do
    for m_type in "${m_types[@]}"; do
        for h_size in "${h_sizes[@]}"; do
            for n_layer in "${n_layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for l_rate in "${l_rates[@]}"; do
                        for tf_ratio in "${tf_ratios[@]}"; do
                            echo "Submitting \
                                f=$fold_id, m=$m_type, h=$h_size, n=$n_layer, \
                                l=$l_rate, d=$dropout, tf=$tf_ratio"

                            # Submit job with parameters passed as environment variables
                            sbatch --export=ALL,         \
                                FOLD_ID      =$fold_id,  \
                                MODEL_TYPE   =$m_type,   \
                                HIDDEN_SIZE  =$h_size,   \
                                NUM_LAYERS   =$n_layer,  \
                                LEARN_RATE   =$l_rate,   \
                                DROPOUT      =$dropout,  \
                                TF_RATIO     =$tf_ratio  \
                                "$TRAIN_SCRIPT"

                            # Increment counter
                            ((total++))

                            echo "Submitted job $total"
                            echo "----------------------------------------"
                        done
                    done
                done
            done
        done
    done
done

echo "All jobs submitted! Total combinations: $total"

# scp -r ddager@oberon2:/scratch2/ddager/single-word-processing-model/weights ~/Desktop/single-word-processing-model/weights
# scp -r ddager@oberon2:/scratch2/ddager/single-word-processing-model/figures ~/Desktop/single-word-processing-model/figures
# scp -r ddager@oberon2:/scratch2/ddager/single-word-processing-model/data/grid_search.csv ~/Desktop/single-word-processing-model/data