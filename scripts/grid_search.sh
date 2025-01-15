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
for f in "${fold_ids[@]}"; do
    for m in "${m_types[@]}"; do
        for h in "${h_sizes[@]}"; do
            for l in "${n_layers[@]}"; do
                for d in "${dropouts[@]}"; do
                    for r in "${l_rates[@]}"; do
                        for t in "${tf_ratios[@]}"; do
                            echo "Submitting f=$f, m=$m, h=$h, n=$l, l=$r, d=$d, tf=$t"

                            # Submit job with parameters passed as environment variables
                            sbatch --export=ALL,FOLD_ID=$f,MODEL_TYPE=$m,HIDDEN_SIZE=$h,NUM_LAYERS=$l,LEARN_RATE=$r,DROPOUT=$d,TF_RATIO=$t "$TRAIN_SCRIPT"

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

# git submodule update --init --recursive
# scp -r ddager@oberon2:/scratch2/ddager/swp-model/weights ~/Desktop/swp-model/weights
# scp -r ddager@oberon2:/scratch2/ddager/swp-model/results/figures ~/Desktop/swp-model/results/figures
# scp -r ddager@oberon2:/scratch2/ddager/swp-model/results/gridsearch ~/Desktop/swp-model/results/gridsearch