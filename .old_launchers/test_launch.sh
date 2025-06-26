#!/bin/bash
echo "======================================================"
echo "    Strategic Counsel - AI Legal Analysis Platform"
echo "======================================================"
echo "Testing launch process..."
echo ""

# Set up environment
cd /home/jcockburn/SC-Gen-3
export PATH=/home/jcockburn/.local/bin:$PATH

# Check if streamlit is accessible
echo "Checking streamlit installation..."
which streamlit
if [ $? -eq 0 ]; then
    echo "✅ Streamlit found at: $(which streamlit)"
else
    echo "❌ Streamlit not found in PATH"
    exit 1
fi

# Check if app.py exists
if [ -f "app.py" ]; then
    echo "✅ app.py found"
else
    echo "❌ app.py not found in current directory"
    exit 1
fi

echo ""
echo "🚀 Starting Strategic Counsel..."
echo "📡 Server will be available at: http://localhost:8501"
echo "🌐 Browser should open automatically"
echo ""

# Launch the application
/home/jcockburn/.local/bin/streamlit run app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false 