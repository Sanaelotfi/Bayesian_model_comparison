#!/bin/bash

for ntrain in {100,200,300,400}
do
    python exact_runner.py --ntrain=${ntrain} --losstype=mll
done
echo All done