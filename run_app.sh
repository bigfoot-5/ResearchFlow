#!/bin/bash
# rm -rf venv
# deactivate
# python3 -m venv venv
# source venv/bin/activate
# Change to the project directory
cd "$(dirname "$0")"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a service is running
check_service() {
    if ! command_exists "$1"; then
        echo "Error: $1 is not installed. Please install it first."
        exit 1
    fi
}

# Function to check if a port is in use
check_port() {
    if ! lsof -i :"$1" >/dev/null 2>&1; then
        echo "Error: Nothing is running on port $1. Please start the required service."
        exit 1
    fi
}

# Check required services
echo "Checking required services..."
check_service "python3"
check_service "pip"
check_service "psql"

# Check if PostgreSQL is running
echo "Checking PostgreSQL..."
if ! pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
    echo "Error: PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

# Check if Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

# Create required directories
echo "Creating required directories..."
mkdir -p cognition_engine/data
mkdir -p cognition_engine/chroma_db
mkdir -p cognition_engine/chroma_db_deals

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

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Populate vector database
echo "Populating vector database..."
python scripts/populate_vector_db.py

cd ..

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