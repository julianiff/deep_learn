#!/bin/bash


echo "Watch processes"

echo -n "Input ssh adress for git Update > "
    read text
    echo "Starting to watch at: "$text

ssh $text "
    watch nvidia-smi
"