#!/bin/bash
END=$1
i=0
while [ "$i" -lt "$END" ]; do
    python3 run_client.py --dataset "$2" --summary "$3" --host "$4" --centralip "$5" &
    i=$((i + 1))
done
wait

# Sample command
# sh run.sh #clients dataset scheduler myip serverip
# sh run.sh 25 CIFAR10 py 1.2.3.4 1.2.3.5