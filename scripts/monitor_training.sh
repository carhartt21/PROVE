#!/bin/bash
# Compact Training Progress Monitor
# Usage: ./monitor_training.sh [interval_seconds] [log_file]

INTERVAL=${1:-60}
LOG_FILE=${2:-"${HOME}/repositories/PROVE/logs/monitor_training.log"}
WEIGHTS_ROOT="${AWARE_DATA_ROOT}"
WEIGHTS_DIRS=(
    "WEIGHTS"
    "WEIGHTS_STAGE_2"
    "WEIGHTS_CITYSCAPES"
    "WEIGHTS_RATIO_ABLATION"
    "WEIGHTS_EXTENDED"
    "WEIGHTS_LOSS_ABLATION"
)
LOCKS_DIR="${AWARE_DATA_ROOT}/training_locks"

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

# Function to count jobs finished within last N minutes
# Args: $1=user, $2=status (DONE or EXIT), $3=minutes
count_recent_jobs() {
    local user=$1
    local status=$2
    local minutes=$3
    local cutoff_epoch=$(date -d "$minutes minutes ago" '+%s')
    local current_year=$(date '+%Y')
    
    bjobs -a -u "$user" -o "stat:8 finish_time:20" 2>/dev/null | \
    awk -v status="$status" -v cutoff="$cutoff_epoch" -v year="$current_year" '
        NR > 1 && $1 == status {
            # finish_time format: "Feb  4 13:45" or similar
            finish_str = $2 " " $3 " " $4 " " year
            cmd = "date -d \"" finish_str "\" +%s 2>/dev/null"
            cmd | getline finish_epoch
            close(cmd)
            if (finish_epoch != "" && finish_epoch >= cutoff) {
                count++
            }
        }
        END { print count + 0 }
    '
}

while true; do
    clear
    {
    echo -e "${BOLD}${CYAN}=== Training Monitor - $(date '+%Y-%m-%d %H:%M:%S') ===${NC}"
    
    # Compact job status with totals
    echo ""
    CHGE_STATS=$(bjobs -u chge7185 -o 'stat' 2>/dev/null | tail -n +2 | sort | uniq -c)
    CHGE_RUN=$(echo "$CHGE_STATS" | awk '$2=="RUN" {print $1}')
    CHGE_PEND=$(echo "$CHGE_STATS" | awk '$2=="PEND" {print $1}')
    CHGE_RUN=${CHGE_RUN:-0}
    CHGE_PEND=${CHGE_PEND:-0}
    CHGE_TOTAL=$((CHGE_RUN + CHGE_PEND))
    echo -e "${BLUE}[JOBS]${NC} chge7185: ${GREEN}RUN: $CHGE_RUN${NC} | ${YELLOW}PEND: $CHGE_PEND${NC} | Total: $CHGE_TOTAL"
    
    MIMA_STATS=$(bjobs -u ${USER} -o 'stat' 2>/dev/null | tail -n +2 | sort | uniq -c)
    MIMA_RUN=$(echo "$MIMA_STATS" | awk '$2=="RUN" {print $1}')
    MIMA_PEND=$(echo "$MIMA_STATS" | awk '$2=="PEND" {print $1}')
    MIMA_RUN=${MIMA_RUN:-0}
    MIMA_PEND=${MIMA_PEND:-0}
    MIMA_TOTAL=$((MIMA_RUN + MIMA_PEND))
    echo -e "${BLUE}[JOBS]${NC} ${USER}: ${GREEN}RUN: $MIMA_RUN${NC} | ${YELLOW}PEND: $MIMA_PEND${NC} | Total: $MIMA_TOTAL"

    OTHER_PEND=$(($(bjobs -u all -o 'stat' -q BatchGPU 2>/dev/null | tail -n +2 | grep -c '^PEND')-$((CHGE_PEND + MIMA_PEND))))
    OTHER_PEND=${OTHER_PEND:-0}
    echo -e "${BLUE}[JOBS]${NC} Other Users: ${YELLOW}PEND: $OTHER_PEND${NC}"
    
    # Count failed/done jobs in last 4 hours (240 minutes)
    echo ""
    CHGE_FAILED=$(count_recent_jobs chge7185 EXIT 240)
    MIMA_FAILED=$(count_recent_jobs ${USER} EXIT 240)
    CHGE_DONE=$(count_recent_jobs chge7185 DONE 240)
    MIMA_DONE=$(count_recent_jobs ${USER} DONE 240)
    
    echo -e "${RED}[FAILED 4h]${NC} chge7185: $CHGE_FAILED | ${USER}: $MIMA_FAILED"
    echo -e "${GREEN}[DONE 4h]${NC}   chge7185: $CHGE_DONE | ${USER}: $MIMA_DONE"
    
    # Active locks (compact)
    LOCK_COUNT=$(ls -1 "$LOCKS_DIR"/*.lock 2>/dev/null | wc -l)
    echo ""
    echo -e "${YELLOW}[LOCKS]${NC} $LOCK_COUNT active"
    
    # Recent checkpoints (compact)
    RECENT_CKPT=$(find "${WEIGHTS_DIRS[@]/#/$WEIGHTS_ROOT/}" -name "*.pth" -mmin -30 2>/dev/null | wc -l)
    echo -e "${YELLOW}[CHECKPOINTS]${NC} $RECENT_CKPT new in last 30 min"
    
    echo ""
    echo -e "${CYAN}Next: ${INTERVAL}s | Ctrl+C to stop${NC}"
    } | log_output
    
    sleep $INTERVAL
done
