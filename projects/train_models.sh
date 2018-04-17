python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 2 -optim 0 -lr 1.0 -ne 12 -mf classic_00.pkl -l classic_00.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 2 -optim 0 -lr 0.0001 -ne 12 -mf classic_01.pkl -l classic_01.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 2 -optim 1 -lr 0.0001 -ne 12 -mf classic_02.pkl -l classic_02.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 2 -optim 1 -lr 0.002 -ne 12 -mf classic_03.pkl -l classic_03.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 2 -optim 2 -lr 0.01 -ne 12 -mf classic_04.pkl -l classic_04.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 2 -optim 2 -lr 0.002 -ne 12 -mf classic_05.pkl -l classic_05.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 0 -optim 0 -lr 1.0 -ne 12 -mf basset_00.pkl -l basset_00.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 0 -optim 0 -lr 0.0001 -ne 12 -mf basset_01.pkl -l basset_01.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 0 -optim 1 -lr 0.0001 -ne 12 -mf basset_02.pkl -l basset_02.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 0 -optim 1 -lr 0.002 -ne 12 -mf basset_03.pkl -l basset_03.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 0 -optim 2 -lr 0.01 -ne 12 -mf basset_04.pkl -l basset_04.log &&
python baseline_train.py -d /n/data_02/Basset/data/encode_roadmap.h5 -mt 0 -optim 2 -lr 0.002 -ne 12 -mf basset_05.pkl -l basset_05.log
