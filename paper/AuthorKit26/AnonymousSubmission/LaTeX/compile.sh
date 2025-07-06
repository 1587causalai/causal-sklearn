#!/bin/bash

# Compile LaTeX document with bibliography
# Usage: ./compile.sh

echo "Compiling LaTeX document..."

# First compilation to process basic LaTeX commands
echo "Step 1: First pdflatex run..."
pdflatex main.tex

# Process bibliography
echo "Step 2: Processing bibliography..."
bibtex main

# Second compilation to include bibliography
echo "Step 3: Second pdflatex run (for bibliography)..."
pdflatex main.tex

# Final compilation to resolve all cross-references
echo "Step 4: Final pdflatex run (for cross-references)..."
pdflatex main.tex

echo "Compilation complete!"
echo "Generated file: main.pdf"

# Optional: Open PDF if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening PDF..."
    open main.pdf
fi