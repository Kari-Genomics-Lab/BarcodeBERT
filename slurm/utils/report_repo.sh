#!/bin/bash

echo "========================================================================"
echo "-------- Reporting git repo configuration ------------------------------"
date
echo ""
echo "pwd: $(pwd)"
echo "commit ref: $(git rev-parse HEAD)" || echo "Not a git repo"
echo ""
git status || echo "Not a git repo"
echo "========================================================================"
echo ""
