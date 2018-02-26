#!/bin/bash

# Definitely comment out extraneous prints before running!
# $(seq 0 0.1 1)
# for alphab in 0.2 0.5 0.8
# do
# 	for alphat in 0.2 0.5 0.8
# 	do
# 		echo "params are $alphab $alphat"
# 		echo "params are $alphab $alphat" >> trigramgrid.txt
# 		python3 trigram.py --alphab $alphab --alphat $alphat >> trigramgrid.txt
# 	done
# done
# exit 0
# for eps in 0.0001 0.001 0.01 0.1
# do
# 	echo "eps is $eps"
# 	echo "eps is $eps" >> trigramepsgrid.txt
# 	python3 trigram.py -e $eps >> trigramepsgrid.txt
# done
python3 main.py -e 8 -hd 4 -hs 1000 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3
python3 main.py -e 8 -hd 4 -hs 1000 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -w
python3 main.py -e 8 -hd 4 -hs 1000 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -w -wt

python3 main.py -m 1 -mf '../../models/HW3/s2s.pkl' -e 8 -hd 4 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3