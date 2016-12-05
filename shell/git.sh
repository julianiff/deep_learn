#!/bin/bash

echo "deploy on Rattle"

ssh jiff@rattle.ifi.uzh.ch "
    echo 'update Repository'
    cd final
    git pull
    git status
    echo 'Updated Code on rattle'
"