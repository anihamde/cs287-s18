import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import h5py

class model_maker:
    def __init__(self,in_chn,in_width):
        self.in_chn   = in_chn
        self.in_width = in_width
        self.Modules = []
