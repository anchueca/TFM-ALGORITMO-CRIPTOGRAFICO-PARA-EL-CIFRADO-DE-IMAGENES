#!/bin/bash

###############################################################################
# Script to run visual encryption phase tests
# Tests: padding, column permutation, block permutation, diffusion, edge cases
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$SCRIPT_DIR/cuda"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                  VISUAL ENCRYPTION PHASE TESTS                           ║"
echo "║                                                                          ║"
echo "║  This program demonstrates individual encryption phases:                 ║"
echo "║    • TEST 1: Padding with various image sizes                           ║"
echo "║    • TEST 2: Column permutation effects                                 ║"
echo "║    • TEST 3: Block permutation with different block sizes               ║"
echo "║    • TEST 4: Obfuscation/Diffusion visualization                        ║"
echo "║    • TEST 5: Combined phase progression                                 ║"
echo "║    • TEST 6: Edge cases (unusual image dimensions)                      ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if binary exists
if [ ! -f "$CUDA_DIR/test/visual_demo.out" ]; then
    echo "❌ Error: visual_demo.out not found at $CUDA_DIR/test/visual_demo.out"
    echo "   Please run: cd $CUDA_DIR && make -f Makefile.visual"
    exit 1
fi

echo "✓ Binary found: visual_demo.out ($(du -h $CUDA_DIR/test/visual_demo.out | cut -f1))"
echo ""
echo "Starting visual phase tests..."
echo ""

# Run the tests
cd "$CUDA_DIR"
./test/visual_demo.out

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                          TESTS COMPLETED                                 ║"
echo "║                                                                          ║"
echo "║  What you saw:                                                          ║"
echo "║    • Padding applied to non-standard image sizes                         ║"
echo "║    • Column permutation scrambling patterns                              ║"
echo "║    • Block permutation with 4×4 to 64×64 block sizes                   ║"
echo "║    • Diffusion converting clear patterns to noise                       ║"
echo "║    • Combined effect of all phases                                       ║"
echo "║    • Robustness test with 1px to 1024px images                          ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
