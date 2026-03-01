#!/bin/bash

# Convert all notebooks and execute them recursively with error handling
# This script continues even if a notebook fails, reporting all errors at the end.

FAILED_NOTEBOOKS=()
SUCCESS_COUNT=0
TOTAL_COUNT=0

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Starting notebook execution in docs/examples...${NC}"

# Find all notebooks and iterate over them
# Using process substitution < <(...) to keep variable scope intact
while IFS= read -r -d '' notebook; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo -e "\n${BLUE}--------------------------------------------------${NC}"
    echo -e "🚀 Executing ($TOTAL_COUNT): ${BLUE}$notebook${NC}"
    
    # Execute the notebook
    # uv run jupyter nbconvert returns non-zero if execution fails
    if uv run jupyter nbconvert --to notebook --execute --inplace "$notebook"; then
        echo -e "${GREEN}✅ Success: $notebook${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}❌ Error: Failed to execute $notebook${NC}" >&2
        FAILED_NOTEBOOKS+=("$notebook")
    fi
done < <(find docs/examples docs/learn -name "*.ipynb" -print0)

echo -e "\n${BLUE}==================================================${NC}"
echo -e "📊 Execution Summary:"
echo -e "Total:    $TOTAL_COUNT"
echo -e "${GREEN}Success:  $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed:   ${#FAILED_NOTEBOOKS[@]}${NC}"
echo -e "${BLUE}==================================================${NC}"

if [ ${#FAILED_NOTEBOOKS[@]} -gt 0 ]; then
    echo -e "${RED}❌ The following notebooks had errors:${NC}"
    for failed in "${FAILED_NOTEBOOKS[@]}"; do
        echo -e "  - $failed"
    done
    exit 1
else
    echo -e "${GREEN}🎉 All notebooks executed successfully!${NC}"
    exit 0
fi
