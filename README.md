# NeuralParts
code reconstruction of NeuralParts for CS492

To see mesh produced by trained network follow this instruction:

1. Select one of trained model from foler 'models'.
2. Open 'config.py' and set variable 'n_primitive' to be number of primitive of your selected model. (number of primitive of trained_model_n_prim is n)
3. On terminal, run 'python3 utils/result_visualization.py {path to selected model}'. for example, set n_primitive=5 and run 'python3 utils/result_visualization.py ./models/trained_model_5_prim.pth'.
4. Now open 'result' foler and you would see .obj file of overall mesh and each primitives, and .png file of rendered mesh.


Preprocessed data is about 300GB, so we cannot provide preprocessed data. If you want to test IOU and Chamfer-L1 loss of the model, or if you want to train the model yourself, follow this instruction:
1. Make sure you have more than 300GB of disk space.
2. Go to https://dfaust.is.tue.mpg.de/ and download 'MALE REGISTRATIONS' and 'FEMALE REGISTRATIONS'.
3. Move 'registrations_m.hdf5' and 'registrations_f.hdf5' into folder you want, and set variable 'dfaust_dataset_directory' of 'config.py' to be path of that folder.
4. Run 'python3 preprocess/preprocess_dfaust.py' on terimanl. Wait for perprocssing to finish. Preprocessing takes a long time.
5. For evaluation, select model, adjust 'n_primitives' of 'config.py', and run 'python3 test.py {path to selected model}'
6. for training, adjust parameters in 'config.py' and run 'python3 train.py'. The model will be saved as 'model.pth' at each step, and model with best validation score will be saved as 'best_model.pth'. 

