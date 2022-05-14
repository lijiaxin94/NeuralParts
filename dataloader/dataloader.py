from torch.utils.data import DataLoader
from dataloader.datainfo import Datainfo_dfaust
from dataloader.dataset import dataset_dfaust
from config import *

def build_dataloader(datatype, splits):
    if (datatype=='dfaust'):
        datainfo = Datainfo_dfaust(splits)
        dataset = dataset_dfaust(datainfo)
        batch_size = dfaust_batch_size
    else:
        print('no dataset named ' + datatype)
        exit(0)
    return DataLoader(dataset, batch_size = dfaust_batch_size, 
            num_workers = dfaust_num_workers, shuffle = True)

