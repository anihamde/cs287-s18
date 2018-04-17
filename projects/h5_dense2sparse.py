import h5py
import numpy as np

data = h5py.File("encode_roadmap.h5",'r+')

train_in = np.argmax(data['train_in'], axis=1).astype('int8')
del data['train_in']
dset0 = data.create_dataset('train_in',data=train_in)

valid_in = np.argmax(data['valid_in'], axis=1).astype('int8')
del data['valid_in']
dset1 = data.create_dataset('valid_in',data=valid_in)

test_in = np.argmax(data['test_in'], axis=1).astype('int8')
del data['test_in']
dset2 = data.create_dataset('test_in',data=test_in)

data.close()
