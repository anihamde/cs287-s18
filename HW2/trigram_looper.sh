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
for eps in 0.0001 0.001 0.01 0.1
do
	echo "eps is $eps"
	echo "eps is $eps" >> trigramepsgrid.txt
	python3 trigram.py -e $eps >> trigramepsgrid.txt
done