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

# First overnight
# python3 main.py -e 8 -hd 4 -hs 1000 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 # 10.01/9.85 train/val ppl after 8 epochs
# python3 main.py -e 8 -hd 4 -hs 1000 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -w # 11.39/10.41 train/val ppl after 8 epochs
# python3 main.py -mf '../../models/HW3/wwt.pkl' -e 8 -hd 4 -hs 1000 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -w -wt 
# 12.33/10.22 train/valid ppl after 8 epochs

# python3 main.py -m 1 -mf '../../models/HW3/s2s.pkl' -e 8 -hd 4 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3
# s2s 16.74 val ppl

# second overnight
# these all failed on predict2 but the models saved. predict2 should work now, if you want to rerun it
python3 main.py -e 8 -hd 2 -hs 300 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -acc -mf '../../models/HW3/baseline.pkl' # 12.67/10.77 train/val ppl
python3 main.py -e 8 -hd 2 -hs 300 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -w -mf '../../models/HW3/w.pkl' # 10.28/9.31 train/val ppl
python3 main.py -e 8 -hd 2 -hs 300 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -wt -mf '../../models/HW3/wt.pkl' # 19.02/13.57 train/val ppl
python3 main.py -e 8 -hd 1 -hs 300 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -m 1 -mf '../../models/HW3/s2s.pkl' # 22.32/23.17 train/val ppl
python3 main.py -e 8 -hd 3 -hs 300 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -m 1 -mf '../../models/HW3/s2s3.pkl' # 19.95/16.80 train/val ppl
python3 main.py -e 8 -hd 3 -hs 300 -vs 300 -ada -lr 1.0 -vd 0.3 -ld 0.3 -m 1 -b -mf '../../models/HW3/s2s3b.pkl' # 21.00/17.62 train/val ppl
