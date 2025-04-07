#!/bin/bash

# Define the path to your training script
TRAIN_SCRIPT="./scripts/train_repetition.sh"

# Define arrays for each hyperparameter
n_epochs=(100)
fold_ids=("")
b_sizes=(1024)
r_types=("lstm")
h_sizes=(128)
n_layers=(1)
l_rates=(0.001)
dropouts=(0.0)
tf_ratios=(0.0)
seeds=(111 124 31 74 0 43 21 41 8 87)

# Initialize counter for total combinations
total=0

# Nested loops to iterate through all combinations
for e in "${n_epochs[@]}"; do
    for b in "${b_sizes[@]}"; do
        for f in "${fold_ids[@]}"; do
            for m in "${r_types[@]}"; do
                for h in "${h_sizes[@]}"; do
                    for l in "${n_layers[@]}"; do
                        for d in "${dropouts[@]}"; do
                            for r in "${l_rates[@]}"; do
                                for t in "${tf_ratios[@]}"; do
                                    for s in "${seeds[@]}"; do
                                        export NUM_EPOCHS=$e
                                        export BATCH_SIZE=$b
                                        export RECUR_TYPE=$m
                                        export HIDDEN_SIZE=$h
                                        export NUM_LAYERS=$l
                                        export LEARN_RATE=$r
                                        export DROPOUT=$d
                                        export TF_RATIO=$t
                                        export FOLD_ID=$f
                                    echo "Submitting e=$e b=$b m=$m h=$h l=$l r=$r d=$d t=$t f=$f s=$s"
                                    sbatch --export=ALL "$TRAIN_SCRIPT"
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
    done
done

echo "All jobs submitted! Total combinations: $total"

# git submodule update --init --recursive
# scp -r ddager@oberon2:/scratch2/ddager/swp-model/weights ~/Desktop/swp-model/
# scp -r ddager@oberon2:/scratch2/ddager/swp-model/results/gridsearch/train ~/Desktop/swp-model/results/gridsearch/