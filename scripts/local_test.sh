#!/bin/bash
# local_test.sh - Script to test SLURM scripts locally

# Usage: ./local_test.sh your_slurm_script.sh [test_type]
# test_type options:
#   - run: Run the script without SBATCH directives (default)
#   - dryrun: Just print what would be run
#   - mini: Run a reduced version (fewer iterations)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <slurm_script> [test_type]"
    exit 1
fi

SCRIPT="$1"
TEST_TYPE="${2:-run}"

if [ ! -f "$SCRIPT" ]; then
    echo "Script not found: $SCRIPT"
    exit 1
fi

# Create a temporary directory for modified script
TEMP_DIR=$(mktemp -d)
TEMP_SCRIPT="$TEMP_DIR/temp_script.sh"

echo "Testing SLURM script: $SCRIPT"
echo "Test type: $TEST_TYPE"

# Copy the script but exclude SBATCH directives
grep -v "^#SBATCH" "$SCRIPT" > "$TEMP_SCRIPT"

if [ "$TEST_TYPE" = "mini" ]; then
    # Create a mini version for faster testing - modify for loops
    sed -i '' 's/for n in {3..10}/for n in {3..4}/' "$TEMP_SCRIPT"
    # Add a test flag to python commands to limit execution
    sed -i '' 's/python scripts\/\([^\.]*\)\.py/python scripts\/\1.py /' "$TEMP_SCRIPT"
fi

# Make the script executable
chmod +x "$TEMP_SCRIPT"

if [ "$TEST_TYPE" = "dryrun" ]; then
    # Just show what would be executed
    echo "----- Script that would be executed: -----"
    cat "$TEMP_SCRIPT"
    echo "----------------------------------------"
else
    # Run the script
    export SLURM_JOB_ID=local_test
    echo "Running script..."
    "$TEMP_SCRIPT"
fi

# Clean up
rm -rf "$TEMP_DIR"
echo "Test completed" 