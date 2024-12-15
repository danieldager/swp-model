#!/bin/bash

# Define the path to your training script
TRAIN_SCRIPT="./scripts/train_repetition.sh"

# Define arrays for each hyperparameter
h_sizes=(64)
n_layers=(1)
dropouts=(0.1 0.3)
tf_ratios=(0.0)
l_rates=(0.0005)

# Initialize counter for total combinations
total=0

# Nested loops to iterate through all combinations
for h_size in "${h_sizes[@]}"; do
    for n_layer in "${n_layers[@]}"; do
        for dropout in "${dropouts[@]}"; do
            for l_rate in "${l_rates[@]}"; do
                for tf_ratio in "${tf_ratios[@]}"; do
                    echo "Submitting h=$h_size, n=$n_layer, d=$dropout, tf=$tf_ratio, l=$l_rate"

                    # Submit job with parameters passed as environment variables
                    sbatch --export=ALL,H_SIZE=$h_size,N_LAYERS=$n_layer,DROPOUT=$dropout,L_RATE=$l_rate,TF_RATIO=$tf_ratio "$TRAIN_SCRIPT"

                    # Increment counter
                    ((total++))

                    echo "Submitted job $total"
                    echo "----------------------------------------"
                done
            done
        done
    done
done

echo "All jobs submitted! Total combinations: $total"

# scp -r ddager@oberon2:/scratch2/ddager/single-word-processing-model/weights ~/Desktop/single-word-processing-model/weights
# scp -r ddager@oberon2:/scratch2/ddager/single-word-processing-model/figures ~/Desktop/single-word-processing-model/figures
# scp -r ddager@oberon2:/scratch2/ddager/single-word-processing-model/data/grid_search.csv ~/Desktop/single-word-processing-model/data