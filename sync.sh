#!/bin/bash

# Check if the notes repository path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <notes_repo_path>"
  exit 1
fi

# Define the paths to the repositories
NOTES_REPO_PATH="$1"
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