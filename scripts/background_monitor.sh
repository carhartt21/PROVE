#!/bin/bash
# Background monitor for training progress
# Logs to logs/training_monitor.log
# Run with: nohup ./scripts/background_monitor.sh &

LOG_FILE="/home/chge7185/repositories/PROVE/logs/training_monitor.log"
INTERVAL=1800  # 30 minutes

mkdir -p /home/chge7185/repositories/PROVE/logs

echo "Starting background monitor at $(date)" >> "$LOG_FILE"
echo "Checking every $INTERVAL seconds" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

while true; do
    {
        echo ""
        echo "=== STATUS CHECK $(date) ==="
        echo ""
        echo "=== Job Queue ==="
        echo "chge7185: $(bjobs -u chge7185 -o stat 2>/dev/null | grep -c RUN) RUN, $(bjobs -u chge7185 -o stat 2>/dev/null | grep -c PEND) PEND"
        echo "mima2416: $(bjobs -u mima2416 -o stat 2>/dev/null | grep -c RUN) RUN, $(bjobs -u mima2416 -o stat 2>/dev/null | grep -c PEND) PEND"
        echo ""
        echo "=== Active Locks ==="
        cd /home/chge7185/repositories/PROVE && python training_lock.py list 2>/dev/null || echo "Error checking locks"
        echo ""
        echo "=== Running IDD-AW/BDD10k Jobs ==="
        bjobs -u all -o "jobid user job_name stat" 2>/dev/null | grep -E "(idd-aw|retest_bdd10k).*RUN" | head -10 || echo "  None running"
        echo ""
        echo "=== Recent Activity (30 min) ==="
        echo "IDD-AW checkpoints: $(find /scratch/aaa_exchange/AWARE/WEIGHTS -path "*idd-aw*" -name "*.pth" -mmin -30 2>/dev/null | wc -l)"
        echo "BDD10k fixed results: $(find /scratch/aaa_exchange/AWARE/WEIGHTS -path "*bdd10k*test_results_detailed_fixed*" -name "results.json" -mmin -30 2>/dev/null | wc -l)"
        echo "======================================" 
    } >> "$LOG_FILE"
    
    sleep $INTERVAL
done
