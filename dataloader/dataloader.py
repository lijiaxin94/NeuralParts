from torch.utils.data import DataLoader
from config import *

def build_dataloader(datatype, splits):
    if (datatype=='dfaust'):
        dataset = Dataset_dfaust(splits)
        batch_size = dfaust_batch_size
    else:
        print('no dataset named ' + datatype)
        exit(0)
    return DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True)

if __name__=='__main__':
    build_dataloader('dfaust',['train'])