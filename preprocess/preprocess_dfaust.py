# here we refer the code that is given as default code to process dfaust
# the code is from https://dfaust.is.tue.mpg.de/download.php 

from argparse import ArgumentParser
import shutil
import os
from os import mkdir
from os.path import join, exists
import h5py
import sys
import trimesh
import numpy as np
import csv
import random
from simple_3dviz import Mesh
from simple_3dviz.scenes import Scene
from simple_3dviz.utils import save_frame

sys.path.insert(0, os.getcwd()) 
from config import *
from utils.mesh_containment_check import check_mesh_contains

sids = ['50002', '50004', '50007', '50009', '50020',
        '50021', '50022', '50025', '50026', '50027']
mesh_paths = ['registrations_m.hdf5','registrations_f.hdf5',
        'registrations_m.hdf5','registrations_m.hdf5',
        '.registrations_f.hdf5','registrations_f.hdf5',
        'registrations_f.hdf5','registrations_f.hdf5',
        'registrations_m.hdf5','egistrations_m.hdf5']
seqs = ['chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees',
        'light_hopping_loose', 'light_hopping_stiff', 'one_leg_jump', 
        'one_leg_loose', 'punching', 'running_on_spot', 'shake_arms', 'shake_hips', 'shake_shoulders']


def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1: 
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def delete_folders(folder):
    for i in range(len(sids)):
        sid = sids[i]
        for seq in seqs:
            sidseq = sid + '_' + seq
            tdir = join(dfaust_dataset_directory, sidseq)
            if not exists(tdir): continue
            tdir = join(tdir, folder)
            if not exists(tdir): continue
            shutil.rmtree(tdir)
            print("deleted ", tdir)

def save_mesh():
    for i in range(len(sids)):
        sid = sids[i]
        path = join(dfaust_dataset_directory, mesh_paths[i])
        for seq in seqs:
            dir = dfaust_dataset_directory

            sidseq = sid + '_' + seq
            with h5py.File(path, 'r') as f:
                if sidseq not in f:
                    print('Sequence %s from subject %s not in %s' %
                        (seq, sid, path))
                    f.close()
                    continue
                verts = f[sidseq][()].transpose([2, 0, 1])
                faces = f['faces'][()]

            tdir = join(dir, sidseq)
            if not exists(tdir):
                mkdir(tdir)
            tdir = join(tdir, defaust_mesh_folder)
            if not exists(tdir):
                mkdir(tdir)

            for iv, v in enumerate(verts):
                fname = join(tdir, '%05d.obj' % iv)
                write_mesh_as_obj(fname, v, faces)
            print("saved on ", tdir)

def get_train_datainfo():
    split = ["train","val"]
    datainfo_split = []
    with open(dfaust_split_file, "r") as f:
        data = np.array([row for row in csv.reader(f)])
        for s in split:
            s_data = data[data[:,2]==s]
            for d, l in zip(s_data[:, 0], s_data[:, 1]):
                datainfo_split.append(d + ':' + l)
    return set(datainfo_split)

def save_surface_samples():
    datainfo_split = get_train_datainfo()
    for i in range(len(sids)):
        sid = sids[i]
        for seq in seqs:
            sidseq = sid + '_' + seq
            tdir = join(dfaust_dataset_directory, sidseq)
            if not exists(tdir): continue
            mdir = join(tdir, dfaust_mesh_folder)
            tdir = join(tdir, dfaust_surface_samples_folder)
            if not exists(tdir):
                mkdir(tdir)
            for mesh_file_name in os.listdir(mdir):
                mesh_file_path = join(mdir, mesh_file_name)
                if '.obj' not in mesh_file_path:
                    print('delete file that is not mesh ' + mesh_file_path)
                    os.remove(mesh_file_path)
                    continue
                if (sidseq + ':' + mesh_file_name[:-4]) not in datainfo_split:
                #     if os.path.exists(targetpath):
                #         os.remove(targetpath)
                    continue
                # we don't normalize D-FAUST data
                mesh = trimesh.load(mesh_file_path, process=False)
                p, f = trimesh.sample.sample_surface(mesh,
                        n_preprocessed_surface_samples)
                face_normals = np.array(mesh.face_normals)
                samples = np.hstack([p, face_normals[f,:]])
                np.random.shuffle(samples)
                np.save(join(tdir,mesh_file_name[:-4]+'.npy'),samples)

            print("saved on ", tdir)

# for rendering we refer to code of https://github.com/paschalidoud/hierarchical_primitives
def render_dfaust(scene, prev_renderable, seq, target):
    new_renderable = Mesh.from_file(seq)
    scene.remove(prev_renderable)
    scene.add(new_renderable)
    scene.render()
    save_frame(target, scene.frame)
    return new_renderable

def get_scene():
    scene = Scene((224, 224))
    scene.camera_position = (1, 1.5, 3)
    scene.camera_target = (0, 0.5, 0)
    scene.light = (1, 1.5, 3)
    scene.up_vector = (0, 1, 0)

    return scene

def save_images():
    scene = get_scene()
    renderable = None
    for i in range(len(sids)):
        sid = sids[i]
        for seq in seqs:
            sidseq = sid + '_' + seq
            tdir = join(dfaust_dataset_directory, sidseq)
            if not exists(tdir): continue
            mdir = join(tdir, dfaust_mesh_folder)
            tdir = join(tdir, dfaust_image_folder)
            if not exists(tdir):
                mkdir(tdir)
            for mesh_file_name in os.listdir(mdir):
                meshpath = join(mdir, mesh_file_name)
                imagepath = join(tdir,mesh_file_name[:-4]+'.png')
                renderable = render_dfaust(scene, renderable, meshpath, imagepath)
            print("saved images on ", tdir)

def save_volume_samples():
    datainfo_split = get_train_datainfo()
    for i in range(len(sids)):
        sid = sids[i]
        for seq in seqs:
            sidseq = sid + '_' + seq
            tdir = join(dfaust_dataset_directory, sidseq)
            if not exists(tdir): continue
            mdir = join(tdir, dfaust_mesh_folder)
            tdir = join(tdir, dfaust_volume_samples_folder)
            if not exists(tdir):
                mkdir(tdir)
            for mesh_file_name in os.listdir(mdir):
                targetpath = join(tdir,mesh_file_name[:-4]+'.npz')
                if (sidseq + ':' + mesh_file_name[:-4]) not in datainfo_split:
                #     if os.path.exists(targetpath):
                #         os.remove(targetpath)
                    continue
                meshpath = join(mdir, mesh_file_name)
                mesh = trimesh.load(meshpath, process=False)
                points = (np.random.rand(n_preprocessed_volume_samples, 
                        3)*np.array([1.,2.,1.]) - np.array([0.5, 0.75, 0.5])).astype(np.float32)
                labels = check_mesh_contains(mesh, points).astype(np.float32)[:, None]
                np.savez(targetpath, points=points, occupancies=labels[:,0])
                sys.stdout.write("%.5f \r"%(np.sum(labels)))
            print("saved volume samples on ", tdir)


if __name__ == '__main__':
    #delete_folders("volume_samples")
    #delete_folders("surface_samples")
    #save_mesh()
    save_surface_samples()
    #save_images()
    #save_volume_samples()
