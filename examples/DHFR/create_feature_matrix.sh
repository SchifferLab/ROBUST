#!/bin/bash

# Example of how to create full feature matrix from a set of ROBUST descriptors

DATA="./data/dataset.csv"

source /home/leidnerf/schrodinger.ve/bin/activate

descriptors=(vdw elec rms hbond)

for var in "${descriptors[@]}"
do
	echo "create ${var} feature matrix"
	python ../../transformers/utils/preprocess.py $var -i $DATA --prefix $1 --keep_names

done


for var in "${descriptors[@]}"
do
	echo "merge ${var} feature matrices"
	python ../../transformers/utils/merge.py "${1}_${var}.csv" -i $DATA -t $var --prefix $1 --merge_replicates
done
