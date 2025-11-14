#!/usr/bin/env bash
# Demo startup script with checkpoint-8000
# Usage: ./start_demo.sh

set -e

cd "$(dirname "$0")"/..

# Activate venv
source .venv/bin/activate

# Configuration
export BASE_MODEL_DIR=models/indictrans2-indic-en-1B
export ADAPTER_DIR=outputs/hi_en_r8_v5_full/checkpoint-8000
export DEMO_ENFORCE=on

echo "=========================================="
echo "IndicTrans LoRA Demo - Starting..."
echo "=========================================="
echo "Base model:  $BASE_MODEL_DIR"
echo "Adapter:     $ADAPTER_DIR"
echo "Enforcement: $DEMO_ENFORCE"
echo ""
echo "The demo will start on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Start the demo
uvicorn demo.app:app --port 8000
