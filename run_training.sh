#!/bin/bash

CONFIG_DIR="./config"

TRAIN_CMD="python main.py"

# for CONFIG_FILE in "$CONFIG_DIR"/*.toml; do  (Countries_S1_interactive.toml Countries_S1.toml Countries_S2_interactive.toml Countries_S2.toml Countries_S3_interactive.toml Countries_S3.toml)
for CONFIG_FILE in Countries_S2_interactive.toml Countries_S2_interactive.toml spurious.toml; do
  CONFIG_NAME=$(basename "$CONFIG_FILE" .toml)

  echo "Starting training with $CONFIG_NAME..."
  $TRAIN_CMD "$CONFIG_NAME"

  if [ $? -eq 0 ]; then
    echo "Training with $CONFIG_NAME completed successfully."
  else
    echo "Training with $CONFIG_NAME failed."
  fi
done

echo "All training runs completed."
