#!/bin/bash
# cleanup_embeddings.sh - Script to clean up embedding files

# Usage:
# ./cleanup_embeddings.sh [min_ngram] [max_ngram]
# Example: ./cleanup_embeddings.sh 8 10  # Deletes embeddings for 8, 9 and 10-grams

# Default values
MIN_NGRAM=${1:-3}
MAX_NGRAM=${2:-10}
MODEL_NAME="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
TRAIN_NAME="b1024_l0.001_fall_s42_sn_ec"
CHECKPOINT="75"

# Create an array of n-gram values
NGRAMS=($(seq $MIN_NGRAM $MAX_NGRAM))

# Confirmation
echo "This will delete embedding files for n-grams: ${NGRAMS[@]}"
echo "Are you sure you want to continue? (y/n)"
read -r confirm

if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Delete the embedding files
for n in "${NGRAMS[@]}"; do
    # Define paths to the embedding files
    RESULTS_DIR="results/evaluation/${MODEL_NAME}/${TRAIN_NAME}/${CHECKPOINT}/control"
    H_FILE="${RESULTS_DIR}/${n}grams_h.npy"
    C_FILE="${RESULTS_DIR}/${n}grams_c.npy"
    CSV_FILE="${RESULTS_DIR}/${n}grams.csv"
    
    # Check if files exist and delete them
    for file in "$H_FILE" "$C_FILE" "$CSV_FILE"; do
        if [ -f "$file" ]; then
            echo "Deleting: $file"
            rm "$file"
        else
            echo "File not found: $file"
        fi
    done
done

echo "Cleanup completed." 