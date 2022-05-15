from torch.utils.data import DataLoader
from dataloader.datainfo import Datainfo_dfaust
from dataloader.dataset import MyDataset
from config import *

# iterations in MyDataloader gives list of tensors
# first element of list is image data, batch * 3 * 224 * 224
# second element is surface sample data, batch * n_sampled_pt * (position, normal)
# third element is volume sample data, batch * n_sampled_pt * (position, occupancy, weight)

def build_dataloader(datatype, splits):
    if (datatype=='dfaust'):
        datainfo = Datainfo_dfaust(splits)
        batch_size = dfaust_batch_size
        num_workers = dfaust_num_workers
    else:
        print('no dataset named ' + datatype)
        exit(0)
    return MyDataloader(MyDataset(datainfo), batch_size = batch_size, 
            num_workers = num_workers, shuffle = True)

class MyDataloader(DataLoader):
    def infinite_iterator(self):
        while True:
            for d in self:
                yield d

