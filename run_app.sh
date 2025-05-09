#!/bin/bash
# rm -rf venv
# deactivate
# python3 -m venv venv
# source venv/bin/activate
# Change to the project directory
cd "$(dirname "$0")"

# Create a directory for storing data if it doesn't exist
mkdir -p cognition_engine/data

# Function to kill background processes on exit
cleanup() {
    echo "Shutting down processes..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set the trap for cleanup on exit
trap cleanup EXIT INT TERM

# Install dependencies
echo "Installing backend dependencies..."
pip install -r cognition_engine/requirements.txt
echo "Installing frontend dependencies..."
pip install -r cognition_frontend/requirements.txt

# Start the backend
echo "Starting backend server..."
cd cognition_engine
python src/main.py &
BACKEND_PID=$!
cd ..

# Wait for the backend to start
sleep 5

# Start the frontend
echo "Starting frontend..."
cd cognition_frontend
streamlit run app.py &
FRONTEND_PID=$!
cd ..

echo "Application is running!"
echo "Backend API: http://localhost:8005"
echo "Frontend UI: http://localhost:8501"
echo "Press Ctrl+C to stop all processes"

# Wait for both processes to complete (or until Ctrl+C)
wait 