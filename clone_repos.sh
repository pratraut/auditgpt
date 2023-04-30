#!/bin/sh

# Set your GitHub username and personal access token
GITHUB_USER="martin.kondov@gmail.com"
GITHUB_TOKEN="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"
ORG_NAME="code-423n4"

# Set the API URL for the organization repositories
API_URL="https://api.github.com/orgs/${ORG_NAME}/repos"

# Function to clone repositories if they don't exist
clone_repo() {
  repo_url="$1"
  repo_name=$(basename "${repo_url}" .git)

  if [ ! -d "${repo_name}" ]; then
    git clone "${repo_url}"
  else
    echo "Repository '${repo_name}' already exists, skipping."
  fi
}

# Get the list of repositories and clone them
page=1
while true; do
  repos=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" "${API_URL}?per_page=100&page=${page}" | jq -r '.[].ssh_url')
  
  # Break the loop if no more repositories are returned
  if [ -z "${repos}" ]; then
    break
  fi

  for repo in $repos; do
    clone_repo "${repo}"
  done

  # Increment the page number
  page=$((page + 1))
done

