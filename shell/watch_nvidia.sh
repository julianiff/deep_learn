#!/bin/bash
source shell/config.yml

echo "Watch running processes: "

ssh -t $RATTLESSH "
    watch nvidia-smi
"