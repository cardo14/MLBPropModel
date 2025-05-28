#!/bin/bash

# Set the working directory to the MLBPropModel directory
cd "$(dirname "$0")/.."

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the conda environment if needed
# source /path/to/your/conda/bin/activate your_env_name

# Run the update script from the new location and log output
python database_updates/daily_database_update.py >> logs/daily_update.log 2>&1

# Check the exit status
if [ $? -eq 0 ]; then
    echo "$(date): Update completed successfully." >> logs/daily_update.log
else
    echo "$(date): Update failed. Check logs for details." >> logs/daily_update.log
fi 