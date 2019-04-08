#!/bin/bash
# These results in firstvae.0
# python3 vae.py -m 0 -mf '../../models/HW4/baseline.pkl' # 101.1507, 2.2716, 103.4223
# python3 vae.py -m 0 -ld 10 -mf '../../models/HW4/m0ld10.pkl' # 102.6618, 2.7239, 105.3858
# python3 vae.py -m 1 -mf '../../models/HW4/m1.pkl' # 92.4910, 4.3581, 96.8491
# python3 vae.py -m 2 -mf '../../models/HW4/m2.pkl' # 102.6618, 0.0052, 102.6670
# python3 vae.py -m 2 -ld 10 -mf '../../models/HW4/m2ld10.pkl' # 102.6618, 0.0138, 102.6756
python gan.py -m 1 -ld 100 -ne 1 # small G, small D
python gan.py -m 2 -ld 100 -ne 1 # small G, large D
python gan.py -m 4 -ld 100 -ne 1 # small G, larger D
python gan.py -m 5 -ld 100 -ne 1 # us