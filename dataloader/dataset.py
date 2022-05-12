from torch.utils.data import Dataset
from config import *

class Dataset_base(Dataset):
    def __init__(self, datainfo):
        self.datainfo = datainfo
    
    def __len__(self):
        return len(self.datainfo)
    
    def __getitem__(self, idx):
        raise NotImplementedError()

class 

