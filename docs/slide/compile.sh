#!/bin/bash

# CausalEngine Presentation Compilation Script

# --- Configuration ---
DEFAULT_TARGET="causal_engine_presentation.tex"
TARGET_FILE="${1:-$DEFAULT_TARGET}"
BASE_NAME="${TARGET_FILE%.*}"

# --- Pre-flight Checks ---
echo "Starting compilation for: $TARGET_FILE"

if ! command -v xelatex &> /dev/null; then
    echo "Error: xelatex not found. Please install TeX Live or MacTeX."
    exit 1
fi

if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: Target file '$TARGET_FILE' not found."
    exit 1
fi

# --- Compilation ---
echo "Cleaning old files for $BASE_NAME..."
rm -f "${BASE_NAME}".aux "${BASE_NAME}".log "${BASE_NAME}".nav "${BASE_NAME}".out "${BASE_NAME}".snm "${BASE_NAME}".toc "${BASE_NAME}".vrb "${BASE_NAME}".pdf "${BASE_NAME}".fls "${BASE_NAME}".fdb_latexmk

echo "First compilation..."
xelatex -interaction=nonstopmode "$TARGET_FILE"

echo "Second compilation (generating TOC)..."
xelatex -interaction=nonstopmode "$TARGET_FILE"

# --- Post-compilation ---
if [ -f "${BASE_NAME}.pdf" ]; then
    echo "✅ Compilation successful!"
    echo "📄 Output file: ${BASE_NAME}.pdf"
    
    echo "Cleaning intermediate files..."
    rm -f "${BASE_NAME}".aux "${BASE_NAME}".log "${BASE_NAME}".nav "${BASE_NAME}".out "${BASE_NAME}".snm "${BASE_NAME}".toc "${BASE_NAME}".vrb "${BASE_NAME}".fls "${BASE_NAME}".fdb_latexmk
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Opening PDF in Preview..."
        open "${BASE_NAME}.pdf"
    fi
else
    echo "❌ Compilation failed! Please check error log."
    echo "View ${BASE_NAME}.log for details."
    echo ""
    echo "Common issues and solutions:"
    echo "1. Install required packages: sudo tlmgr install <package>"
    echo "2. Update TeX distribution: sudo tlmgr update --all"
    echo "3. Ensure all fonts are installed on your system (e.g., PingFang SC for Chinese)."
    exit 1
fi
