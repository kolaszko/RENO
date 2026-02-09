#!/bin/bash

# Helper script to run RENO compression in Docker
# Usage: ./run_docker.sh <bag_path> <posq> [topic] [max_messages]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <bag_path> <posq> [topic] [max_messages]"
    echo ""
    echo "Arguments:"
    echo "  bag_path        Path to input bag file (inside /datasets mount)"
    echo "  posq            Position quantization parameter (e.g., 16, 32, 64)"
    echo "  topic           (Optional) Specific topic to compress"
    echo "  max_messages    (Optional) Maximum number of messages to process"
    echo ""
    echo "Examples:"
    echo "  $0 /datasets/my_bag.bag 16"
    echo "  $0 /datasets/my_bag.bag 32 /velodyne_points"
    echo "  $0 /datasets/my_bag.bag 16 /velodyne_points 100"
    exit 1
fi

BAG_PATH="$1"
POSQ="$2"
TOPIC="${3:-}"
MAX_MESSAGES="${4:-}"

echo "========================================="
echo "RENO Point Cloud Compression"
echo "========================================="
echo "Bag Path:       $BAG_PATH"
echo "PosQ:           $POSQ"
echo "Topic:          ${TOPIC:-all PointCloud2 topics}"
echo "Max Messages:   ${MAX_MESSAGES:-all}"
echo "========================================="
echo ""

export BAG_PATH
export POSQ
export TOPIC
export MAX_MESSAGES

# Cleanup function to ensure Docker containers are removed
cleanup() {
    echo ""
    echo "========================================="
    echo "Cleaning up Docker containers..."
    echo "========================================="
    docker compose down --remove-orphans
    echo "Cleanup complete."
}

# Set trap to ensure cleanup on exit (normal or interrupted)
trap cleanup EXIT INT TERM

# Run docker-compose (without --build to speed up repeated runs)
# Docker will use cached layers unless Dockerfile or dependencies changed
echo "Starting Docker containers..."
docker compose up

# Store exit code
EXIT_CODE=$?

# Exit with the same code as docker-compose
exit $EXIT_CODE
