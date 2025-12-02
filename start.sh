#!/bin/bash

# Exit on error
set -e

# MasterStat Startup Script

echo "========================================="
echo "  MasterStat - Statistical Analysis Tool"
echo "========================================="
echo ""

# Get the script's directory and change to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verify docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found!"
    echo "Please run this script from the MasterStat project root directory."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo ""
    echo "Please start Docker Desktop and try again:"
    echo "  1. Open Docker Desktop application"
    echo "  2. Wait for it to fully start"
    echo "  3. Run this script again"
    echo ""
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check if docker compose command exists (V2), fallback to docker-compose (V1)
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
    echo "⚠️  Using legacy docker-compose. Consider upgrading to Docker Compose V2"
    echo ""
else
    echo "❌ Neither 'docker compose' nor 'docker-compose' found!"
    echo "Please install Docker Compose"
    exit 1
fi

# Build and start containers
echo "Building and starting containers..."
echo "This may take a few minutes on first run..."
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run docker compose in foreground - it handles Ctrl+C gracefully
$DOCKER_COMPOSE up --build

# This runs if docker compose exits normally (shouldn't happen without -d flag)
echo ""
echo "========================================="
echo "Application stopped."
echo "========================================="
