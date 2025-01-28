#!/bin/bash

# Define Python files to run
PYTHON_FILE1="mjc-2-3bar.py"
PYTHON_FILE2="ti_3bar.py"

# Temporary files to store outputs
OUTPUT1=$(mktemp)
OUTPUT2=$(mktemp)

# Run the Python files and redirect outputs to temporary files
python3 "$PYTHON_FILE1" > "$OUTPUT1" 2>&1
python3 "$PYTHON_FILE2" > "$OUTPUT2" 2>&1

# Compare the outputs
if cmp -s "$OUTPUT1" "$OUTPUT2"; then
    echo "The outputs are the same."
else
    echo "The outputs are different."
    echo "Differences:"
    diff "$OUTPUT1" "$OUTPUT2"
fi

# Clean up temporary files
rm "$OUTPUT1" "$OUTPUT2"
