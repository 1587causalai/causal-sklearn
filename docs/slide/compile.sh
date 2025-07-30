#!/bin/bash

# CausalEngine Presentation Compilation Script

echo "Starting CausalEngine presentation compilation..."

# Check for pdflatex (standard LaTeX compiler)
if ! command -v xelatex &> /dev/null; then
    echo "Error: xelatex not found. Please install TeX Live or MacTeX."
    exit 1
fi

# Clean old files
echo "Cleaning old files..."
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.pdf *.fls *.fdb_latexmk

# Compile document (run twice for proper TOC and references)
echo "First compilation..."
xelatex -interaction=nonstopmode causal_engine_presentation.tex

echo "Second compilation (generating TOC)..."
xelatex -interaction=nonstopmode causal_engine_presentation.tex

# Check if PDF was successfully generated
if [ -f "causal_engine_presentation.pdf" ]; then
    echo "✅ Compilation successful!"
    echo "📄 Output file: causal_engine_presentation.pdf"
    
    # Clean intermediate files
    echo "Cleaning intermediate files..."
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk
    
    # Auto-open PDF on macOS (optional)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Opening PDF in Preview..."
        open causal_engine_presentation.pdf
    fi
else
    echo "❌ Compilation failed! Please check error log."
    echo "View causal_engine_presentation.log for details."
    echo ""
    echo "Common issues and solutions:"
    echo "1. Install required packages: sudo tlmgr install pgfplots"
    echo "2. Update TeX distribution: sudo tlmgr update --all"
    echo "3. For font issues, use pdflatex instead of xelatex"
    exit 1
fi