#!/bin/bash
# JurisFinanceAI - Build Script (Linux/macOS for testing)

echo "========================================="
echo "  JurisFinanceAI - Build Script"
echo "========================================="

set -e

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo "[2/3] Building application..."
pyinstaller build.spec --noconfirm

echo "[3/3] Build complete!"
echo "Output: dist/JurisFinanceAI/"
