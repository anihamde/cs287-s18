#!/bin/bash

# Definitely comment out extraneous prints before running!
# $(seq 0 0.1 1)
for alphab in 0.2 0.5 0.8
do
	for alphat in 0.2 0.5 0.8
	do
		echo "params are $alphab $alphat" >> trigramgrid.txt
		python trigram.py --alphab $alphab --alphat $alphat >> trigramgrid.txt
	done
done
exit 0