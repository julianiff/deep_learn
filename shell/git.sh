#!/bin/bash

echo "First push to Repo, then deployment of the code"
echo " "
echo -n "Commit message eingeben > "
read commit
git commit -m "$commit"
git push origin master

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