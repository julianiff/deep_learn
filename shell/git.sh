#!/bin/bash

source shell/config.yml


echo "First push to Repo, then deployment of the code"
echo " "


if [[ `git status --porcelain` ]]; then
    echo "Code has changed, please add commitmessage"
    echo -n "Add Commit Message > "
    read commit
    git add -A && git commit -m "$commit"
    git push origin master
else
  echo "No changes were made to the codebase"
fi


echo "The Code will be deployed to: "$RATTLESSH

ssh $RATTLESSH "
    echo 'update Repository'
    cd final
    git pull
    git status
    echo 'Updated Code'
"