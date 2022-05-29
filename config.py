n_preprocessed_surface_samples = 100000
n_surface_samples = 2000
n_preprocessed_volume_samples = 100000
n_volume_samples = 5000
loss_weight = [1.0,0.1,0.01,0.1,0.01]
occupancy_loss_temperature = 4e-3

dfaust_dataset_directory = '/data/D-FAUST'
dfaust_split_ratio = {'train': 0.7, 'test':0.2, 'val': 0.1}
dfaust_split_file = '/data/dfaust_split.csv'
dfaust_mesh_folder = 'mesh'
dfaust_image_folder = 'image'
dfaust_surface_samples_folder = 'surface_samples'
dfaust_volume_samples_folder = 'volume_samples'
dfaust_batch_size = 2
dfaust_num_workers = 0


