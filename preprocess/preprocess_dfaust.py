from argparse import ArgumentParser
import shutil
import os
from os import mkdir
from os.path import join, exists
import h5py
import sys
from config import *


def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1: 
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

sids = ['50002', '50004', '50007', '50009', '50020',
        '50021', '50022', '50025', '50026', '50027']
mesh_paths = ['../registrations_m.hdf5','../registrations_f.hdf5',
        '../registrations_m.hdf5','../registrations_m.hdf5',
        '../registrations_f.hdf5','../registrations_f.hdf5',
        '../registrations_f.hdf5','../registrations_f.hdf5',
        '../registrations_m.hdf5','../registrations_m.hdf5']
seqs = ['chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees',
        'light_hopping_loose', 'light_hopping_stiff', 'one_leg_jump', 
        'one_leg_loose', 'punching', 'running_on_spot', 'shake_arms', 'shake_hips', 'shake_shoulders']

def save_mesh():
    for i in range(len(sids)):
        sid = sids[i]
        path = mesh_paths[i]
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

if __name__ == '__main__':
    save_mesh()
