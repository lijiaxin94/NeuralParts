from torch.utils.data import Dataset
from config import *

class Dataset_base(Dataset):
    def __init__(self, datainfo):
        self.datainfo = datainfo
    
    def __len__(self):
        return len(self.datainfo)
    
    def __getitem__(self, idx):
        raise NotImplementedError()

class Dataset_surface_samples(Dataset_base):
    def __init__(self, datainfo):
        super(Dataset_dfaust, self).__init__(datainfo)

        def __getitem__(self, index):
            return self.datainfo[index].get_surface_samples()

class Dataset_image(Dataset_base):
    



