#!/bin/bash

for dset in {boston,power,energy,concrete,winered,winewhite}
do
    for m in {0.1,0.25,0.5,0.75,0.9}
    do
        for ntrain in {100,200,300,400,500,600,700}
        do
            python exact_runner.py --ntrain=${ntrain} --m=${m} --losstype=cmll --dataset=${dset}
        done
    done

    for ntrain in {100,200,300,400,500,600,700}
    do
        python exact_runner.py --ntrain=${ntrain} --losstype=mll --dataset=${dset}
    done
done
echo All done