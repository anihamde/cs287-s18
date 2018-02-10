#!/bin/bash

# Definitely comment out extraneous prints before running!
for alphab in $(seq 0.2 0.1 0.8)
do
	for alphat in $(seq 0.2 0.1 0.8)
	do
		echo "params are $alphab $alphat" >> trigramgrid.txt
		python trigram.py --alphab $alphab --alphat $alphat >> trigramgrid.txt
	done
done
exit 0