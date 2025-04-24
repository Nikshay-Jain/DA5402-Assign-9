#!/bin/bash

# Help function
show_help() {
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE    Path to input file (required)"
    echo "  -o, --output DIR    Output directory (default: output)"
    echo "  -s, --sample        Use a sample of data for testing"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run.sh --input data/amazon_reviews.csv --output results"
}

# Default values
INPUT_FILE=""
OUTPUT_DIR="output"
USE_SAMPLE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--sample)
            USE_SAMPLE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required"
    show_help
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the script
echo "Starting sentiment analysis pipeline..."
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

if [ "$USE_SAMPLE" = true ]; then
    echo "Using data sample for testing"
    python -c "
import pandas as pd
df = pd.read_csv('$INPUT_FILE')
sample = df.sample(n=min(1000, len(df)), random_state=42)
sample.to_csv('${INPUT_FILE}_sample.csv', index=False)
print(f'Created sample with {len(sample)} records')
"
    INPUT_FILE="${INPUT_FILE}_sample.csv"
fi

python main.py --input "$INPUT_FILE" --output "$OUTPUT_DIR"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Sentiment analysis completed successfully"
    echo "Results saved to '$OUTPUT_DIR'"
else
    echo "Error: Sentiment analysis failed"
    exit 1
fi
