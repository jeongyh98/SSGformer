#!/bin/bash

# Define arrays for checkpoints, model types, and gt paths
CHECKPOINT_PATHS=(
    net_g_best.pth
)

MODEL_TYPES=(
    SSGformer
)

DATA_PATHS=(
  "~/allweather_dataset/test/rain_drop_test"
  "~/allweather_dataset/test/test1"
  "~/allweather_dataset/test/Snow100K-L"
)

# Define the evaluation command
evaluate_model() {
  local checkpoint=$1
  local model=$2
  local data=$3

  echo "Evaluating with checkpoint: $checkpoint, model type: $model, data: $data"
  CUDA_VISIBLE_DEVICES=0 python Allweather/evaluate.py --checkpoint "$checkpoint" --model "$model" --data "$data"
  
  if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully for checkpoint $checkpoint with model $model and data $data."
  else
    echo "Evaluation failed for checkpoint $checkpoint with model $model and data $data."
  fi
}

# Perform evaluation for each combination of checkpoints, models, and gt paths
for checkpoint in "${CHECKPOINT_PATHS[@]}"; do
  for model in "${MODEL_TYPES[@]}"; do
    for data in "${DATA_PATHS[@]}"; do
      evaluate_model "$checkpoint" "$model" "$data"
    done
  done
done
