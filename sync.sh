#!/bin/bash

# Check permissions
echo "Checking SSH permissions for GitHub..."
if ! ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo -e "\033[0;31mGitHub SSH permission denied. Please check your SSH keys.\033[0m"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the path to the notes repository
NOTES_REPO_PATH="$SCRIPT_DIR"

# Print the path for verification
echo "Notes repository path: $NOTES_REPO_PATH"
PERSONAL_NOTES_REPO_PATH="$NOTES_REPO_PATH/personal-notes"

# Sync the notes repository
echo "Syncing notes repository..."
cd "$NOTES_REPO_PATH" || exit
git pull origin main

# Sync the personal-notes repository
echo "Syncing personal-notes repository..."
cd "$PERSONAL_NOTES_REPO_PATH" || exit
git pull origin main

# Add, commit, and push any changes in the personal-notes repository
echo "Committing and pushing changes in personal-notes repository..."
git add .
git commit -m "Sync personal-notes repository"
git push origin main

# Go back to the notes repository
cd "$NOTES_REPO_PATH" || exit

# Add, commit, and push any changes in the notes repository
echo "Committing and pushing changes in notes repository..."
git add .
git commit -m "Sync notes repository"
git push origin main

echo "Sync complete."