#!/bin/bash
for ((trial=1;trial<=50;trial++))
do
    echo $trial
    CUDA_VISIBLE_DEVICES=2 python cli.py train-model --trial=$trial
done
