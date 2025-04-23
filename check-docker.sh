#!/bin/bash
# check-docker.sh - Check if Docker is running
# Make sure to run: chmod +x ./check-docker.sh

if docker info > /dev/null 2>&1; then
  echo "Docker is running"
  exit 0
else
  echo "Docker is not running. Please start it first."
  exit 1
fi