#!/bin/bash
# Compact Training Progress Monitor
# Usage: ./monitor_training.sh [interval_seconds] [log_file]

INTERVAL=${1:-60}
LOG_FILE=${2:-"/home/mima2416/repositories/PROVE/logs/monitor_training.log"}
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOCKS_DIR="/scratch/aaa_exchange/AWARE/training_locks"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

mkdir -p "$(dirname "$LOG_FILE")"
echo "Monitor started: ${INTERVAL}s interval, logging to ${LOG_FILE}"

# Log function - strips colors for file, keeps for terminal
log_output() {
    while IFS= read -r line; do
        echo -e "$line"
        echo "$line" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"
    done
}

while true; do
    clear
    {
    echo -e "${BOLD}${CYAN}=== Training Monitor - $(date '+%Y-%m-%d %H:%M:%S') ===${NC}"
    
    # Compact job status
    echo ""
    echo -e "${BLUE}[JOBS]${NC} chge7185: $(bjobs -u chge7185 -o 'stat' 2>/dev/null | tail -n +2 | sort | uniq -c | tr '\n' ' ')"
    echo -e "${BLUE}[JOBS]${NC} mima2416: $(bjobs -u mima2416 -o 'stat' 2>/dev/null | tail -n +2 | sort | uniq -c | tr '\n' ' ')"
    
    # Recently finished - compact format
    echo ""
    echo -e "${RED}[FAILED]${NC} chge7185:"
    bjobs -a -u chge7185 -o "stat jobid exit_code finish_time job_name:40" 2>/dev/null | awk '$1=="EXIT" {$1=""; print}' | head -5
    echo -e "${RED}[FAILED]${NC} mima2416:"
    bjobs -a -u mima2416 -o "stat jobid exit_code finish_time job_name:40" 2>/dev/null | awk '$1=="EXIT" {$1=""; print}' | head -5
    echo ""
    echo -e "${GREEN}[DONE]${NC} chge7185:"
    bjobs -a -u chge7185 -o "stat jobid finish_time job_name:40" 2>/dev/null | awk '$1=="DONE" {$1=""; print}' | head -5
    echo -e "${GREEN}[DONE]${NC} mima2416:"
    bjobs -a -u mima2416 -o "stat jobid finish_time job_name:40" 2>/dev/null | awk '$1=="DONE" {$1=""; print}' | head -5
    
    # Active locks (compact)
    LOCK_COUNT=$(ls -1 "$LOCKS_DIR"/*.lock 2>/dev/null | wc -l)
    echo ""
    echo -e "${YELLOW}[LOCKS]${NC} $LOCK_COUNT active"
    
    # Recent checkpoints (compact)
    RECENT_CKPT=$(find "$WEIGHTS_ROOT" -name "*.pth" -mmin -30 2>/dev/null | wc -l)
    echo -e "${YELLOW}[CHECKPOINTS]${NC} $RECENT_CKPT new in last 30 min"
    
    echo ""
    echo -e "${CYAN}Next: ${INTERVAL}s | Ctrl+C to stop${NC}"
    } | log_output
    
    sleep $INTERVAL
done
done
