#!/bin/sh
n_runs=8

for i in $(seq $n_runs)
do
     python3 delayedXOR_task.py --save_data=True --learning_rate=0.0005 --custom_mode=-2.5 --comment="ModProp_Wab_lr0.0005"
     python3 delayedXOR_task.py --save_data=True --learning_rate=0.001 --custom_mode=-2.5 --comment="ModProp_Wab_lr0.001"  
done
