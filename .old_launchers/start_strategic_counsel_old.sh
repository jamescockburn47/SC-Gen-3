#!/bin/bash

echo "======================================================"
echo "    Strategic Counsel - AI Legal Analysis Platform"
echo "======================================================"
echo "Starting application..."
echo

# Default port
PORT=8501

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to find next available port
find_free_port() {
    local port=$1
    while check_port $port; do
        port=$((port + 1))
    done
    echo $port
}

# Function to kill existing streamlit processes
kill_streamlit() {
    echo "Stopping existing Streamlit processes..."
    pkill -f streamlit 2>/dev/null
    sleep 2
}

# Check if Streamlit is already running on default port
if check_port $PORT; then
    echo "⚠️  Port $PORT is already in use!"
    echo
    echo "Options:"
    echo "1) Open browser to existing app (http://localhost:$PORT)"
    echo "2) Kill existing and restart"
    echo "3) Start on next available port"
    echo "4) Exit"
    echo
    read -p "Choose option (1-4): " choice
    
    case $choice in
        1)
            echo "Opening browser..."
            if command -v xdg-open > /dev/null; then
                xdg-open "http://localhost:$PORT"
            elif command -v open > /dev/null; then
                open "http://localhost:$PORT"
            else
                echo "Please open your browser to: http://localhost:$PORT"
            fi
            exit 0
            ;;
        2)
            kill_streamlit
            ;;
        3)
            PORT=$(find_free_port $((PORT + 1)))
            echo "Using port $PORT instead..."
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Using next available port..."
            PORT=$(find_free_port $((PORT + 1)))
            ;;
    esac
fi

# Start the application
echo "🚀 Starting Strategic Counsel on port $PORT..."
echo "📱 Access at: http://localhost:$PORT"
echo

# Check if we're in the right directory
if [[ ! -f "app.py" ]]; then
    echo "❌ app.py not found. Please run from the SC-Gen-3 directory."
    exit 1
fi

# Function to open browser after delay
open_browser() {
    sleep 3  # Wait for Streamlit to start
    local url="http://localhost:$PORT"
    echo "🌐 Opening browser to $url..."
    
    # Try different browser opening methods
    if command -v wslview > /dev/null; then
        # WSL-specific browser launcher
        wslview "$url"
    elif command -v xdg-open > /dev/null; then
        # Linux default
        xdg-open "$url"
    elif command -v open > /dev/null; then
        # macOS
        open "$url"
    elif command -v start > /dev/null; then
        # Windows
        start "$url"
    else
        echo "⚠️  Could not auto-open browser. Please manually navigate to: $url"
    fi
}

# Start browser opener in background
open_browser &

# Start Streamlit
echo "🌐 Browser will open automatically in 3 seconds..."
~/.local/bin/streamlit run app.py \
    --server.port $PORT \
    --server.headless false \
    --browser.gatherUsageStats false \
    --browser.serverAddress localhost

echo
echo "Application stopped." 