#!/bin/bash
#SBATCH --job-name=ngram_pipeline
#SBATCH --partition=gpu
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ngram_%j.out
#SBATCH --error=logs/ngram_%j.err
#SBATCH --nice=100

# Parameters for the pipeline
MODEL_NAME="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1"
TRAIN_NAME="b1024_l0.001_fall_s42_sn_ec"
CHECKPOINT="75"
BATCH_SIZE=2048
MAX_SAMPLES=300000
BASE_N=3
MODEL_TYPE="matrix"  # Options: bias, matrix

# Function to calculate elapsed time
function elapsed_time() {
    local start_time=$1
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    echo "$hours h $minutes m $seconds s"
}

# Record start time for the entire pipeline
PIPELINE_START=$(date +%s)

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting ngram pipeline"
echo "Model: $MODEL_NAME"
echo "Training: $TRAIN_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Batch Size: $BATCH_SIZE"
echo "Max Samples (n=10): $MAX_SAMPLES"
echo "Base N: $BASE_N"
echo "Model Type: $MODEL_TYPE"
echo "-----------------------------------"

# Step 1: Generate ngram datasets from 3 to 10
echo "Generating ngram datasets..."
STEP1_START=$(date +%s)

for n in {3..10}; do
    echo "Generating ${n}grams dataset..."
    TASK_START=$(date +%s)
    
    python scripts/generate_ngrams.py --n $n --max_samples $MAX_SAMPLES --base_n $BASE_N
    
    if [ $? -ne 0 ]; then
        echo "Error generating ${n}grams dataset"
        exit 1
    fi
    
    TASK_ELAPSED=$(elapsed_time $TASK_START)
    echo "Time to generate ${n}grams: $TASK_ELAPSED"
    echo "-----------------------------------"
done

STEP1_ELAPSED=$(elapsed_time $STEP1_START)
echo "All datasets generated successfully"
echo "Total time for dataset generation: $STEP1_ELAPSED"
echo "==================================="

# Step 2: Run embeddings.py on all datasets
echo "Extracting embeddings for all datasets..."
STEP2_START=$(date +%s)

for n in {3..10}; do
    echo "Extracting embeddings for ${n}grams..."
    TASK_START=$(date +%s)
    
    python scripts/embeddings.py \
        --model_name $MODEL_NAME \
        --train_name $TRAIN_NAME \
        --batch_size $BATCH_SIZE \
        --checkpoint $CHECKPOINT \
        --dataset "ngrams" \
        --ngrams $n \
        --retest
    
    if [ $? -ne 0 ]; then
        echo "Error extracting embeddings for ${n}grams"
        exit 1
    fi
    
    TASK_ELAPSED=$(elapsed_time $TASK_START)
    echo "Time to extract embeddings for ${n}grams: $TASK_ELAPSED"
    echo "-----------------------------------"
done

STEP2_ELAPSED=$(elapsed_time $STEP2_START)
echo "All embeddings extracted successfully"
echo "Total time for embedding extraction: $STEP2_ELAPSED"
echo "==================================="

# Step 3: Run interventions.py on all datasets
echo "Running interventions on all datasets..."
STEP3_START=$(date +%s)

for n in {3..10}; do
    echo "Running interventions for ${n}grams..."
    TASK_START=$(date +%s)
    
    python scripts/interventions.py \
        --model_name $MODEL_NAME \
        --train_name $TRAIN_NAME \
        --batch_size $BATCH_SIZE \
        --checkpoint $CHECKPOINT \
        --length $n \
        --edit_type "substitution" \
        --target_type "type" \
        --model_type $MODEL_TYPE \
        --verbose
    
    if [ $? -ne 0 ]; then
        echo "Error running interventions for ${n}grams"
        exit 1
    fi
    
    TASK_ELAPSED=$(elapsed_time $TASK_START)
    echo "Time to run interventions for ${n}grams: $TASK_ELAPSED"
    echo "-----------------------------------"
done

STEP3_ELAPSED=$(elapsed_time $STEP3_START)
echo "All interventions completed successfully"
echo "Total time for interventions: $STEP3_ELAPSED"
echo "==================================="

# Calculate and display total pipeline runtime
PIPELINE_ELAPSED=$(elapsed_time $PIPELINE_START)
echo "Pipeline completed successfully"
echo "Total pipeline runtime: $PIPELINE_ELAPSED"
echo "==================================="

# Summary of timings
echo "TIMING SUMMARY:"
echo "Dataset Generation: $STEP1_ELAPSED"
echo "Embedding Extraction: $STEP2_ELAPSED"
echo "Interventions: $STEP3_ELAPSED"
echo "Total: $PIPELINE_ELAPSED" 