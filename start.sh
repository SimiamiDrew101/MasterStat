#!/bin/bash

# Exit on error
set -e

# MasterStat Startup Script

echo "========================================="
echo "  MasterStat - Statistical Analysis Tool"
echo "========================================="
echo ""

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

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "========================================="
    echo "Shutting down containers..."
    echo "========================================="
    $DOCKER_COMPOSE down
    exit 0
}

# Trap SIGINT and SIGTERM for graceful shutdown
trap cleanup INT TERM

# Build and start containers
echo "Building and starting containers..."
echo "This may take a few minutes on first run..."
echo ""

$DOCKER_COMPOSE up --build

# This line runs after docker-compose exits
echo ""
echo "========================================="
echo "Application stopped."
echo "========================================="
