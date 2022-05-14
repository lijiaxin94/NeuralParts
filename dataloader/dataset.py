from torch.utils.data import Dataset
from config import *

class MyDataset(Dataset):
    def __init__(self, datainfo):
        self.datainfo = datainfo
    
    def __len__(self):
        return len(self.datainfo)
    
    def __getitem__(self, index):
        image = self.datainfo[index].get_image()
        surface_samples = self.datainfo[index].get_surface_samples()
        volume_samples = self.datainfo[index].get_volume_samples()
        return [image, surface_samples, volume_samples]

