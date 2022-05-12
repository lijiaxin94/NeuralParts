from config import *

class Data_base():
    def path_to_mesh_file(self):
        raise NotImplementedError()
    
    

class Data_dfaust(Data_base):
    def __init__(self, datainfo):
        self.data_dir = datainfo[0]
        self.data_num = datainfo[1]