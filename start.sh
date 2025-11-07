#!/bin/bash

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

# Build and start containers
echo "Building and starting containers..."
echo "This may take a few minutes on first run..."
echo ""

docker-compose up --build

echo ""
echo "========================================="
echo "Application stopped."
echo "========================================="
