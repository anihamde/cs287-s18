#!/bin/bash

python3 vae.py -m 0 -mf '../../models/HW4/baseline.pkl'
python3 vae.py -m 0 -ld 10 -mf '../../models/HW4/m0ld10.pkl'
python3 vae.py -m 1 -mf '../../models/HW4/m1.pkl'
python3 vae.py -m 2 -mf '../../models/HW4/m2.pkl'
python3 vae.py -m 2 -ld 10 -mf '../../models/HW4/m2ld10.pkl'
