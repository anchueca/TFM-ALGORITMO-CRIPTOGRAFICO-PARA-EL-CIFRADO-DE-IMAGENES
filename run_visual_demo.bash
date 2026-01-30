#!/bin/bash

# Script para ejecutar demostraciones visuales del sistema de cifrado de imágenes
# Visual demo runner for image encryption cipher system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "   IMAGE CIPHER - VISUAL DEMONSTRATIONS"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if binary exists
if [ ! -f "cuda/test/visual_demo.out" ]; then
    echo "✗ Visual demo binary not found!"
    echo "Building..."
    cd cuda
    make -f Makefile.visual clean
    make -f Makefile.visual
    cd ..
fi

echo "────────────────────────────────────────────────────────────"
echo "Running Visual Demonstrations..."
echo "────────────────────────────────────────────────────────────"
echo ""
echo "This program will display:"
echo "  • Sierpinski Triangle (Rule 90 Cellular Automata)"
echo "  • Various Cellular Automata Rules (30, 110, 150, 184, 225)"
echo "  • 2D Cellular Automata Pattern"
echo "  • Encryption Phase Visualizations"
echo "  • Block Grid Overlay"
echo "  • Histogram Analysis"
echo ""
echo "All output is displayed in console or OpenCV windows"
echo "No data is written to disk"
echo ""
echo "Press 'q' or click X to close windows"
echo "Press any key in window prompts to continue"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Run the visual demo
./cuda/test/visual_demo.out

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ Visual demonstrations completed successfully!"
echo "════════════════════════════════════════════════════════════"
echo ""
