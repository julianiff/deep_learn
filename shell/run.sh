#If we want to have a runscript, at the moment a bit buggy
#!/bin/bash

source shell/config.yml

echo "Lets run it (run.sh is a bit buggy sometimes, määh): "
echo " "
echo -n "Input filename >"
read filename

ssh $RATTLESSH .zshrc
cd final
python $filename
