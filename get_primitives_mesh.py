import torch
import trimesh
from model import Model, Model_overall
from dataloader.dataloader import build_dataloader
from loss.loss_function import loss_function
from metric.metric_function import metric_function
import utils.get_points_from_sphere as gs
from dataloader.dataset import build_dataset
import os, sys

def get_primitives_mesh():
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
    model = torch.load("model.pth").to(device)
    target = build_dataset('dfaust', [0.7, 0.2, 0.1]).get(0)
    for i in range(len(target)):
        target[i] = target[i].to(device)
    prediction = model(target)
        # --------------- Only for debugging ----------------
    r = True 
    if r:
        p_p = model.points_primitives #batch * 222 *n_primitives * 3
        print(str(p_p == None))
        r = False
        # --------------- Only for debugging ----------------
    B, n_points, n_primitives, D = p_p.shape
    assert (D == 3)
    face = (gs.fx_sample_face(B, n_points, n_primitives, d=3, randperm=False))
    print("number of primitives : " + str(n_primitives))
    for j in range(3):
        r = trimesh.Trimesh(p_p[0, :, j, :], face)
        r.export(file_obj=".", file_type='obj')

get_primitives_mesh()

#class GPM(torch.nn.Module):
#    def __init__(self):
#        get_primitives_mesh()



    