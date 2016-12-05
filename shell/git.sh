#!/bin/bash

echo "First push to Repo, then deployment of the code"
echo " "
echo "Check if code has changed"

if [[ `git status --porcelain` ]]; then
    echo -n "Add Commit Message > "
    read commit
    git commit -m "$commit"
    git push origin master
else
  echo "No changes were made"
fi



echo -n "Input ssh adress for git Update > "
    read text
    echo "The Code will be deployed to: "$text

ssh $text "
    echo 'update Repository'
    cd final
    git pull
    git status
    echo 'Updated Code'
"