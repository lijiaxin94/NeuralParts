import os
import csv
import numpy as np
from torch.utils.data import Dataset

from config import *

class dfaust_data():
    def __init__(self, datainfo):
        self.data_dir = datainfo[0]
        self.data_num = datainfo[1]


class dfaust_dataset():
    def __init__(self, splits):
        self.datainfo_list = []

        # get all datainfo in computer
        datainfo_local = []
        for d in sorted(os.listdir(dfaust_dataset_directory)):
            for l in sorted(os.listdir(os.path.join(
                dfaust_dataset_directory, d, dfaust_mesh_folder)))[20:]:
                datainfo_local.append((d + ':' +l[:-4]))
        print("found " + str(len(datainfo_local)) + " D-FAUST dataset from local")

        datainfo_split = []
        with open(dfaust_split_file, "r") as f:
            data = np.array([row for row in csv.reader(f)])
            for s in splits:
                s_data = data[data[:,2]==s]
                for d, l in zip(s_data[:, 0], s_data[:, 1]):
                    datainfo_split.append(d + ':' + l)
        datainfo_split = set(datainfo_split)

        for info in datainfo_local:
            if info in datainfo_split:
                self.datainfo_list.append(info.split(":"))   

        print("split data into " + str(len(self.datainfo_list)) + " data")     


    def __len__(self):
        return len(self.datainfo_list)

    def get(self, index):
        return dfaust_data(self.datainfo_list[i])


if __name__=='__main__':
    print(len(dfaust_dataset(['val'])))
