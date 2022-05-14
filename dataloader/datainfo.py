import os
import csv
import numpy as np
from config import *

class Datainfo_base():
    def __init__(self):
        self.datainfo_list = []

    def __len__(self):
        return len(self.datainfo_list)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        return self.get(index)

    def get(self, index):
        raise NotImplementedError()


class Datainfo_dfaust(Datainfo_base):
    def __init__(self, splits):
        super(Datainfo_dfaust, self).__init__()

        # get all datainfo in computer
        datainfo_local = []
        for d in sorted(os.listdir(dfaust_dataset_directory)):
            if '.hdf5' in d: continue
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

    def get(self, index):
        return Data_dfaust(self.datainfo_list[index])  
