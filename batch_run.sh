#!/bin/bash

# Batch script to run RENO compression on multiple bag files
# Each entry in the list should specify: bag_path, posq, topic (optional)

# Define the list of bag files to process
# Format: "bag_path|posq|topic|max_messages"
# topic and max_messages are optional (use empty string to skip)
declare -a BAG_LIST=(
    # Example entries - modify these with your actual bag files
    "/datasets/collage/2021-04-07-13-52-31_1-math-easy.bag|16|/os_cloud_node/points|"
    "/datasets/vbr/colosseo_train0_11.bag|32|/ouster/points|"
    # Add more entries here as needed
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}RENO Batch Processing${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "Total number of bag files to process: ${#BAG_LIST[@]}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Counter for tracking progress
total=${#BAG_LIST[@]}
current=0
success=0
failed=0

# Log file for tracking results
LOG_FILE="batch_run_$(date +%Y%m%d_%H%M%S).log"
echo "Batch run started at $(date)" > "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

# Process each bag file
for entry in "${BAG_LIST[@]}"; do
    current=$((current + 1))
    
    # Parse the entry
    IFS='|' read -r bag_path posq topic max_messages <<< "$entry"
    
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}Processing $current of $total${NC}"
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "Bag Path:       $bag_path"
    echo -e "PosQ:           $posq"
    echo -e "Topic:          ${topic:-all PointCloud2 topics}"
    echo -e "Max Messages:   ${max_messages:-all}"
    echo -e "${YELLOW}=========================================${NC}"
    echo ""
    
    # Log the current processing
    echo "[$current/$total] Processing: $bag_path (posq=$posq, topic=${topic:-all})" >> "$LOG_FILE"
    
    # Build the command
    if [ -z "$topic" ] && [ -z "$max_messages" ]; then
        # No topic, no max_messages
        cmd="./single_run.sh \"$bag_path\" \"$posq\""
    elif [ -z "$max_messages" ]; then
        # Topic but no max_messages
        cmd="./single_run.sh \"$bag_path\" \"$posq\" \"$topic\""
    else
        # Both topic and max_messages
        cmd="./single_run.sh \"$bag_path\" \"$posq\" \"$topic\" \"$max_messages\""
    fi
    
    # Execute the command
    echo "Executing: $cmd"
    echo ""
    
    start_time=$(date +%s)
    eval $cmd
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Check if successful
    if [ $exit_code -eq 0 ]; then
        success=$((success + 1))
        echo -e "${GREEN}✓ Successfully processed $bag_path (${duration}s)${NC}"
        echo "  SUCCESS (${duration}s)" >> "$LOG_FILE"
    else
        failed=$((failed + 1))
        echo -e "${RED}✗ Failed to process $bag_path (exit code: $exit_code)${NC}"
        echo "  FAILED (exit code: $exit_code, ${duration}s)" >> "$LOG_FILE"
    fi
    
    echo ""
    echo ""
done

# Summary
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Batch Processing Complete${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "Total:      $total"
echo -e "${GREEN}Success:    $success${NC}"
echo -e "${RED}Failed:     $failed${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "Log file: $LOG_FILE"
echo ""

# Log summary
echo "----------------------------------------" >> "$LOG_FILE"
echo "Batch run completed at $(date)" >> "$LOG_FILE"
echo "Total: $total | Success: $success | Failed: $failed" >> "$LOG_FILE"

exit 0
