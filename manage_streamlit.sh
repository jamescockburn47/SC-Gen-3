#!/bin/bash

# Streamlit Process Manager for Strategic Counsel
# Usage: ./manage_streamlit.sh [start|stop|restart|status|ports]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="Strategic Counsel"
DEFAULT_PORT=8501

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}======================================================${NC}"
    echo -e "${BLUE}    $APP_NAME - Process Manager${NC}"
    echo -e "${BLUE}======================================================${NC}"
}

check_streamlit_processes() {
    pgrep -f "streamlit.*app.py" 2>/dev/null
}

get_port_from_process() {
    local pid=$1
    ps -p $pid -o args --no-headers | grep -o -- '--server.port [0-9]*' | awk '{print $2}'
}

show_status() {
    echo -e "\n${YELLOW}📊 Streamlit Status:${NC}"
    local pids=$(check_streamlit_processes)
    
    if [[ -z "$pids" ]]; then
        echo -e "${RED}❌ No Streamlit processes running${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Running Streamlit processes:${NC}"
        for pid in $pids; do
            local port=$(get_port_from_process $pid)
            local port_display=${port:-"unknown"}
            echo -e "   PID: $pid, Port: $port_display"
            echo -e "   URL: ${BLUE}http://localhost:$port_display${NC}"
        done
        return 0
    fi
}

show_ports() {
    echo -e "\n${YELLOW}🔌 Port Usage:${NC}"
    echo "Checking ports 8501-8510..."
    for port in {8501..8510}; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            local pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
            local process=$(ps -p $pid -o comm --no-headers 2>/dev/null || echo "unknown")
            echo -e "   Port $port: ${RED}BUSY${NC} (PID: $pid, Process: $process)"
        else
            echo -e "   Port $port: ${GREEN}FREE${NC}"
        fi
    done
}

stop_streamlit() {
    echo -e "\n${YELLOW}🛑 Stopping Streamlit processes...${NC}"
    local pids=$(check_streamlit_processes)
    
    if [[ -z "$pids" ]]; then
        echo -e "${YELLOW}ℹ️  No Streamlit processes to stop${NC}"
        return 0
    fi
    
    for pid in $pids; do
        echo "Stopping PID: $pid"
        kill $pid 2>/dev/null
    done
    
    sleep 2
    
    # Force kill if still running
    local remaining=$(check_streamlit_processes)
    if [[ -n "$remaining" ]]; then
        echo -e "${YELLOW}Force killing remaining processes...${NC}"
        for pid in $remaining; do
            kill -9 $pid 2>/dev/null
        done
    fi
    
    echo -e "${GREEN}✅ Streamlit stopped${NC}"
}

find_available_port() {
    local port=$DEFAULT_PORT
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

start_streamlit() {
    echo -e "\n${YELLOW}🚀 Starting $APP_NAME...${NC}"
    
    # Check if already running
    if show_status >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Streamlit is already running!${NC}"
        show_status
        echo -e "\nUse ${BLUE}'./manage_streamlit.sh restart'${NC} to restart"
        return 1
    fi
    
    # Find available port
    local port=$(find_available_port)
    
    if [[ $port -ne $DEFAULT_PORT ]]; then
        echo -e "${YELLOW}ℹ️  Port $DEFAULT_PORT busy, using port $port${NC}"
    fi
    
    # Check if app.py exists
    if [[ ! -f "$SCRIPT_DIR/app.py" ]]; then
        echo -e "${RED}❌ app.py not found in $SCRIPT_DIR${NC}"
        return 1
    fi
    
    echo -e "${GREEN}📱 Starting on port $port...${NC}"
    echo -e "${GREEN}🌐 Access at: http://localhost:$port${NC}"
    
    cd "$SCRIPT_DIR"
    ~/.local/bin/streamlit run app.py \
        --server.port $port \
        --server.headless false \
        --browser.gatherUsageStats false \
        --browser.serverAddress localhost &
    
    sleep 3
    
    if show_status >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $APP_NAME started successfully!${NC}"
    else
        echo -e "${RED}❌ Failed to start $APP_NAME${NC}"
        return 1
    fi
}

restart_streamlit() {
    echo -e "\n${YELLOW}🔄 Restarting $APP_NAME...${NC}"
    stop_streamlit
    sleep 1
    start_streamlit
}

show_help() {
    print_header
    echo -e "\n${YELLOW}Usage:${NC}"
    echo "  ./manage_streamlit.sh [command]"
    echo
    echo -e "${YELLOW}Commands:${NC}"
    echo "  start    - Start Streamlit application"
    echo "  stop     - Stop all Streamlit processes"
    echo "  restart  - Stop and start Streamlit"
    echo "  status   - Show running processes"
    echo "  ports    - Show port usage"
    echo "  help     - Show this help message"
    echo
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./manage_streamlit.sh start"
    echo "  ./manage_streamlit.sh status"
    echo "  ./manage_streamlit.sh restart"
}

# Main script logic
case "${1:-help}" in
    start)
        print_header
        start_streamlit
        ;;
    stop)
        print_header
        stop_streamlit
        ;;
    restart)
        print_header
        restart_streamlit
        ;;
    status)
        print_header
        show_status
        ;;
    ports)
        print_header
        show_ports
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac 