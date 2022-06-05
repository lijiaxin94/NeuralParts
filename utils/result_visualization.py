import torch
import trimesh
import os, sys
sys.path.insert(0, os.getcwd()) 
from model import Model, Model_overall
from dataloader.dataloader import build_dataloader
from loss.loss_function import loss_function
from metric.metric_function import metric_function
import utils.get_points_from_sphere as gs
from dataloader.dataset import build_dataset
import matplotlib.pyplot as plt
from simple_3dviz import Mesh
from simple_3dviz.scenes import Scene
from simple_3dviz.utils import save_frame

def visualize_result(index, mesh_file_name, original_mesh_rendered, mesh_rendered):
    device =  torch.device('cpu')
    model = Model(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.set_new_device(device)
    target, mesh_path = build_dataset('dfaust', ['train']).get(index) #6000
    print("result for", mesh_path)
    for i in range(len(target)):
        target[i] = target[i].to(device)
    num_points = 60 ** 2 - 60 * 2 + 2
    prediction = model.primitive_points(target, num_points)
    p_p = prediction
    B, n_points, n_primitives, D = p_p.shape
    assert (D == 3)
    face = (gs.fx_sample_face(B, num_points, n_points, d=3, randperm=False))
    for j in range(n_primitives):
        c = p_p[0, :, j, :].cpu().detach().numpy()
        r = trimesh.Trimesh(c, face)
        r.export(file_obj=("./result/mesh_00" + str(j) + ".obj"), file_type='obj')
    Listmesh = []
    for j in range(n_primitives):
        c = p_p[0, :, j, :].cpu().detach().numpy()
        #print("shape of c is : " + str(len(c)) + " * " + str(len(c[0])))
        Listmesh.append(trimesh.Trimesh(c, face))
    trimesh.util.concatenate(Listmesh).export(file_obj=(mesh_file_name), file_type='obj')
    render_mesh(mesh_file_name, mesh_rendered)
    render_mesh(mesh_path, original_mesh_rendered)
    printf("done")

def get_scene():
    scene = Scene((1024, 1024))
    scene.camera_position = (0.7, 1.2, 2.1)
    scene.camera_target = (0, 0.5, 0)
    scene.light = (1, 1.5, 3)
    scene.up_vector = (0, 1, 0)

    return scene

def render_mesh(file_name, save_path):
    scene = get_scene()
    mesh = Mesh.from_file(file_name)
    scene.add(mesh)
    scene.render()
    save_frame(save_path, scene.frame)

if __name__=='__main__':
    visualize_result(6000, "./result/mesh_overall.obj", "./result/gt_mesh_rendered.png", "./result/mesh_rendered.png")
