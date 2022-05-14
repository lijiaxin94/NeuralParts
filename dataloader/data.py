from config import *
from os.path import join, exists
import numpy as np
from PIL import Image

class Data_base():
    def path_to_mesh_file(self):
        raise NotImplementedError()

    def path_to_surface_samples(self):
        raise NotImplementedError()

    def path_to_image_file(self):
        raise NotImplementedError()

    def get_surface_samples(self):
        n_pre, n_sample = get_n_surface_samples(self)
        pre_sampled = np.load(self.path_to_surfaace_samples, mmap_mode="r")
        rand = int(np.random.rand() * (n_pre - n_sample))
        return np.array(pre_sampled[rand:rand+n_sample]).astype(np.float32)
    
    def get_n_surface_samples(self):
        raise NotImplementedError()
    


class Data_dfaust(Data_base):
    def __init__(self, datainfo):
        self.data_dir = datainfo[0]
        self.data_num = datainfo[1]

    def path_to_mesh_file(self):
        return join(dfaust_dataset_directory, self.data_dir, 
                dfaust_mesh_folder, self.data_num + '.obj')

    def path_to_surface_samples(self):
        return join(dfaust_dataset_directory, self.data_dir, 
                dfaust_surface_samples_folder, self.data_num + '.npy')

    def path_to_image_file(self):
        return join(dfaust_dataset_directory, self.data_dir, 
                dfaust_image_folder, self.data_num + '.png')

    def get_n_surface_samples(self):
        return (dfaust_n_preprocessed_surface_samples, dfaust_n_surface_samples)

    def get_image(self):
        return np.transpose(np.array(Image.open(self.path_to_image_file())).convert("RGB"))


    
