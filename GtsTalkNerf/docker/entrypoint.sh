#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh

# Function to handle kill signals
cleanup() {
    echo "Stopping all services..."
    kill $(jobs -p)
    exit
}
trap cleanup SIGINT SIGTERM

# 1. Start GPT-SoVITS
echo "Starting GPT-SoVITS API..."
cd /app/GPT-SoVITS
conda activate gpt-sovits
# api_v2.py usually runs on 9880. -a 0.0.0.0 binds to all interfaces.
python api_v2.py -a 0.0.0.0 &
GPT_PID=$!

# 2. Start Backend
echo "Starting GtsTalkNeRF Backend..."
cd /app
conda activate GtsTalkNeRF
python backend/backend.py &
BACKEND_PID=$!

# 3. Start Frontend
echo "Starting GtsTalkNeRF Frontend..."
# We are already in /app and GtsTalkNeRF is active
python frontend/frontend.py &
FRONTEND_PID=$!

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
