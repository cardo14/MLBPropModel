#!/bin/bash

# Set the working directory to the MLBPropModel directory
cd "$(dirname "$0")/.."

# Activate the conda environment if needed
# source /path/to/your/conda/bin/activate your_env_name

# Run the update script and log output
python update_mlb_database.py >> logs/daily_update.log 2>&1

# Check the exit status
if [ $? -eq 0 ]; then
    echo "$(date): Update completed successfully." >> logs/daily_update.log
else
    echo "$(date): Update failed. Check logs for details." >> logs/daily_update.log
fi 