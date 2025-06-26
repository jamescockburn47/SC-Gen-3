#!/bin/bash

# ============================================================================
#                         Strategic Counsel Launcher
#                      Professional Legal AI Platform  
# ============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="Strategic Counsel"
DEFAULT_PORT=8501
LOGO_PATH="$SCRIPT_DIR/static/icons/strategic_counsel_logo.svg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
print_header() {
    clear
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                      ⚖️  Strategic Counsel ⚖️                     ${NC}"
    echo -e "${BLUE}            Multi-Agent AI Legal Analysis Platform              ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo
}

# Check if port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Find next available port
find_free_port() {
    local port=$1
    while check_port $port; do
        port=$((port + 1))
    done
    echo $port
}

# Kill existing streamlit processes
kill_streamlit() {
    echo -e "${YELLOW}🛑 Stopping existing Streamlit processes...${NC}"
    pkill -f streamlit 2>/dev/null
    sleep 2
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Open browser with multiple fallback methods
open_browser() {
    sleep 3  # Wait for Streamlit to start
    local url="http://localhost:$1"
    
    echo -e "${BLUE}🌐 Opening browser to $url...${NC}"
    
    # Try different browser opening methods in order of preference
    if command -v wslview >/dev/null 2>&1; then
        wslview "$url" 2>/dev/null
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" 2>/dev/null
    elif command -v open >/dev/null 2>&1; then
        open "$url" 2>/dev/null
    elif command -v start >/dev/null 2>&1; then
        start "$url" 2>/dev/null
    else
        echo -e "${YELLOW}⚠️  Auto-browser opening not available${NC}"
        echo -e "${BLUE}🔗 Please manually navigate to: $url${NC}"
    fi
}

# Check system requirements
check_requirements() {
    echo -e "${BLUE}🔍 Checking system requirements...${NC}"
    
    # Check if we're in the right directory
    if [[ ! -f "$SCRIPT_DIR/app.py" ]]; then
        echo -e "${RED}❌ app.py not found. Please run from the SC-Gen-3 directory.${NC}"
        exit 1
    fi
    
    # Check Python/Python3
    if ! command -v python3 >/dev/null 2>&1; then
        echo -e "${RED}❌ Python 3 not found. Please install Python 3.${NC}"
        exit 1
    fi
    
    # Check Streamlit
    if ! python3 -c "import streamlit" 2>/dev/null; then
        echo -e "${RED}❌ Streamlit not installed. Run: pip install streamlit${NC}"
        exit 1
    fi
    
    # Check Ollama connection
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Ollama not accessible. Multi-agent features may be limited.${NC}"
        echo -e "${YELLOW}   Start Ollama service to enable full AI capabilities.${NC}"
    else
        echo -e "${GREEN}✅ Ollama service detected${NC}"
    fi
    
    echo -e "${GREEN}✅ System requirements met${NC}"
    echo
}

# Main launch function
launch_app() {
    local port=$1
    
    print_header
    
    echo -e "${BLUE}🚀 Starting $APP_NAME on port $port...${NC}"
    echo -e "${BLUE}📱 Access URL: http://localhost:$port${NC}"
    echo -e "${BLUE}🌐 Browser will open automatically in 3 seconds...${NC}"
    echo -e "${YELLOW}💡 Press Ctrl+C to stop the application${NC}"
    echo
    
    # Start browser opener in background
    open_browser $port &
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Start Streamlit with optimal settings
    python3 -m streamlit run app.py \
        --server.port $port \
        --server.headless false \
        --browser.gatherUsageStats false \
        --browser.serverAddress localhost \
        --theme.base light \
        --theme.primaryColor "#0066cc" \
        --logger.level info
}

# Handle interactive port selection
handle_port_conflict() {
    echo -e "${YELLOW}⚠️  Port $DEFAULT_PORT is already in use!${NC}"
    echo
    echo "Options:"
    echo "1) 🌐 Open browser to existing app (http://localhost:$DEFAULT_PORT)"
    echo "2) 🔄 Kill existing process and restart"  
    echo "3) 🔀 Start on next available port"
    echo "4) ❌ Exit"
    echo
    read -p "Choose option (1-4): " choice
    
    case $choice in
        1)
            echo -e "${GREEN}🌐 Opening browser to existing application...${NC}"
            open_browser $DEFAULT_PORT
            exit 0
            ;;
        2)
            kill_streamlit
            return $DEFAULT_PORT
            ;;
        3)
            local new_port=$(find_free_port $((DEFAULT_PORT + 1)))
            echo -e "${BLUE}🔀 Using port $new_port instead...${NC}"
            return $new_port
            ;;
        4)
            echo -e "${BLUE}👋 Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}⚠️  Invalid choice. Using next available port...${NC}"
            local new_port=$(find_free_port $((DEFAULT_PORT + 1)))
            return $new_port
            ;;
    esac
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--port)
                DEFAULT_PORT="$2"
                shift 2
                ;;
            -h|--help)
                echo "Strategic Counsel Launcher"
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -p, --port PORT    Specify port number (default: 8501)"
                echo "  -h, --help         Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check system requirements
    check_requirements
    
    # Handle port conflicts
    local port=$DEFAULT_PORT
    if check_port $port; then
        port=$(handle_port_conflict)
    fi
    
    # Launch the application
    launch_app $port
}

# Handle script interruption gracefully
trap 'echo -e "\n${BLUE}👋 Strategic Counsel stopped. Goodbye!${NC}"; exit 0' INT

# Run main function
main "$@" 