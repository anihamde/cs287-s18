#!/bin/bash

# python3 vae.py -m 0 -mf '../../models/HW4/baseline.pkl' # 101.1507, 2.2716, 103.4223
# python3 vae.py -m 0 -ld 10 -mf '../../models/HW4/m0ld10.pkl' # 102.6618, 2.7239, 105.3858
python3 vae.py -m 1 -mf '../../models/HW4/m1.pkl' # 92.4910, 4.3581, 96.8491
python3 vae.py -m 2 -mf '../../models/HW4/m2.pkl'
python3 vae.py -m 2 -ld 10 -mf '../../models/HW4/m2ld10.pkl'
