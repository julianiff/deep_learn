#!/bin/bash

echo "deployment of the code"

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