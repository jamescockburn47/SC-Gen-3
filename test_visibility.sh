#!/bin/bash

echo "======================================================"
echo "    Text Visibility Test - Strategic Counsel"
echo "======================================================"
echo "Starting text visibility test..."
echo

# Kill any existing test processes
pkill -f "test_text_visibility.py" 2>/dev/null
sleep 1

# Start the test on a different port
echo "🚀 Starting text visibility test on port 8502..."
echo "📱 Access at: http://localhost:8502"
echo

~/.local/bin/streamlit run test_text_visibility.py \
    --server.port 8502 \
    --server.headless false \
    --browser.gatherUsageStats false \
    --browser.serverAddress localhost

echo
echo "Test completed." 