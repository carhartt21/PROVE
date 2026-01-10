#!/bin/bash
# Monitor training progress for IDD-AW retrain and BDD10k retest jobs
# Usage: ./monitor_training.sh [interval_seconds]

INTERVAL=${1:-60}
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOCKS_DIR="/scratch/aaa_exchange/AWARE/training_locks"
LOGS_DIR="/home/chge7185/repositories/PROVE/logs/retrain"
RETEST_LOGS="/scratch/aaa_exchange/AWARE/LOGS/retest_bdd10k"

echo "=========================================="
echo "Training Progress Monitor"
echo "Interval: ${INTERVAL}s"
echo "=========================================="

while true; do
    clear
    echo "========================================================"
    echo "Training Progress Monitor - $(date)"
    echo "========================================================"
    
    # Job status summary
    echo ""
    echo "=== JOB STATUS ==="
    echo "Your (chge7185) jobs:"
    bjobs -u chge7185 -o "stat" 2>/dev/null | tail -n +2 | sort | uniq -c | tr '\n' '  '
    echo ""
    echo "mima2416 jobs:"
    bjobs -u mima2416 -o "stat" 2>/dev/null | tail -n +2 | sort | uniq -c | tr '\n' '  '
    echo ""
    
    # Active locks
    echo ""
    echo "=== ACTIVE TRAINING LOCKS ==="
    if [ -d "$LOCKS_DIR" ] && [ "$(ls -A $LOCKS_DIR 2>/dev/null)" ]; then
        for lock in "$LOCKS_DIR"/*.lock; do
            if [ -f "$lock" ]; then
                name=$(basename "$lock" .lock)
                job_id=$(cat "$lock" 2>/dev/null | grep -o '"job_id": "[^"]*"' | cut -d'"' -f4)
                host=$(cat "$lock" 2>/dev/null | grep -o '"hostname": "[^"]*"' | cut -d'"' -f4)
                echo "  $name (Job: $job_id on $host)"
            fi
        done
    else
        echo "  No active locks"
    fi
    
    # Recent IDD-AW checkpoints
    echo ""
    echo "=== RECENT IDD-AW CHECKPOINTS (last 30 min) ==="
    find "$WEIGHTS_ROOT" -path "*idd-aw*" -name "*.pth" -mmin -30 2>/dev/null | head -10 | while read f; do
        echo "  $(ls -lh "$f" 2>/dev/null | awk '{print $5, $6, $7, $8, $9}')"
    done
    if [ -z "$(find "$WEIGHTS_ROOT" -path "*idd-aw*" -name "*.pth" -mmin -30 2>/dev/null | head -1)" ]; then
        echo "  None found"
    fi
    
    # Recent BDD10k test results
    echo ""
    echo "=== RECENT BDD10K TEST RESULTS (last 30 min) ==="
    find "$WEIGHTS_ROOT" -path "*bdd10k*" -name "results.json" -mmin -30 2>/dev/null | head -10 | while read f; do
        parent=$(dirname "$f")
        echo "  $parent"
    done
    if [ -z "$(find "$WEIGHTS_ROOT" -path "*bdd10k*" -name "results.json" -mmin -30 2>/dev/null | head -1)" ]; then
        echo "  None found"
    fi
    
    # Recent log activity
    echo ""
    echo "=== RECENT LOG FILES ==="
    echo "IDD-AW retrain logs:"
    ls -lt "$LOGS_DIR"/*.out 2>/dev/null | head -5 | awk '{print "  " $9 " (" $6, $7, $8 ")"}'
    echo "BDD10k retest logs:"
    ls -lt "$RETEST_LOGS"/*.out 2>/dev/null | head -5 | awk '{print "  " $9 " (" $6, $7, $8 ")"}'
    
    # Check for errors in recent logs
    echo ""
    echo "=== ERRORS IN RECENT LOGS (last 5 min) ==="
    find "$LOGS_DIR" -name "*.err" -mmin -5 -size +0 2>/dev/null | head -5 | while read f; do
        echo "  ERROR in $f:"
        tail -3 "$f" | head -3 | sed 's/^/    /'
    done
    if [ -z "$(find "$LOGS_DIR" -name "*.err" -mmin -5 -size +0 2>/dev/null | head -1)" ]; then
        echo "  No recent errors"
    fi
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Next update in ${INTERVAL}s..."
    
    sleep $INTERVAL
done
