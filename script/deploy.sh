#!/bin/bash

# Exit on error
set -e

# Configuration
DEPLOY_ENV=${1:-production}
DOCKER_COMPOSE_FILE="Docker/docker-compose.yml"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

echo "Starting deployment for environment: $DEPLOY_ENV"

# Create necessary directories
mkdir -p logs
mkdir -p $BACKUP_DIR
mkdir -p data/models
mkdir -p nginx/ssl

# Backup current state
echo "Creating backup..."
if [ -f "data/models/best_model.pt" ]; then
    cp data/models/best_model.pt $BACKUP_DIR/
fi
if [ -f "configs/config.json" ]; then
    cp configs/config.json $BACKUP_DIR/
fi

# Generate SSL certificates if they don't exist
if [ ! -f "nginx/ssl/cert.pem" ] || [ ! -f "nginx/ssl/key.pem" ]; then
    echo "Generating SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=VN/ST=HoChiMinh/L=HoChiMinh/O=Medical/CN=localhost"
fi

# Copy appropriate configuration
echo "Setting up configuration..."
cp configs/$DEPLOY_ENV.json configs/config.json

# Build and deploy
echo "Building and deploying containers..."
~/.local/bin/docker-compose -f $DOCKER_COMPOSE_FILE down
~/.local/bin/docker-compose -f $DOCKER_COMPOSE_FILE build --no-cache
~/.local/bin/docker-compose -f $DOCKER_COMPOSE_FILE up -d

# Wait for services to be up
echo "Waiting for services to start..."
sleep 10

# Health check
echo "Performing health check..."
if curl -f http://localhost/health; then
    echo "Deployment successful!"
else
    echo "Health check failed! Rolling back..."
    docker-compose -f $DOCKER_COMPOSE_FILE down
    cp $BACKUP_DIR/config.json configs/
    if [ -f "$BACKUP_DIR/best_model.pt" ]; then
        cp $BACKUP_DIR/best_model.pt data/models/
    fi
    docker-compose -f $DOCKER_COMPOSE_FILE up -d
    exit 1
fi

# Setup monitoring
if [ "$DEPLOY_ENV" = "production" ]; then
    echo "Setting up monitoring..."
    docker-compose -f docker-compose.monitoring.yml up -d
fi

# Cleanup
echo "Cleaning up old data..."
find backups -mtime +7 -type d -exec rm -rf {} +

echo "Deployment completed successfully!"