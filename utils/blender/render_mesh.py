import argparse
import os
import numpy as np
import torch
import shutil
from tqdm import tqdm
from smplx import SMPL
from trimesh import Trimesh
import bpy

class npy2obj:
    def __init__(self, npy_path, smpl_path):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True).item()['rotation'] # [frame_num, 24, 3]
        self.positions = np.load(self.npy_path, allow_pickle=True).item()['position'] # [frame_num, 24, 3]
        self.smpl = SMPL(model_path=smpl_path, gender='MALE').eval()
        self.faces = self.smpl.faces
        self.nframes, self.njoints, self.nfeats = self.motions.shape

        self.rotations = torch.from_numpy(self.motions)
        self.global_orient = self.rotations[:, 0:1]
        self.rotations = self.rotations[:, 1:]
        betas = torch.zeros([self.rotations.shape[0], self.smpl.num_betas], dtype=self.rotations.dtype, device=self.rotations.device)

        self.out = self.smpl.forward(global_orient=self.global_orient, body_pose=self.rotations, transl=torch.zeros((self.nframes, 3)), betas=betas)
        self.joints = self.out.joints.detach().cpu().numpy()[:, 0:24, :]
        self.vertices = self.out.vertices.detach().cpu().numpy()
        self.order = [0, 2, 1]
        # self.vertices += np.tile(self.positions[:, 0:1, :], (1, self.vertices.shape[1], 1))
    
    def get_trimesh(self, frame_i):
        return Trimesh(vertices=self.vertices[frame_i], faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def get_root(self, i):
        return self.positions[i, 0, self.order].copy()

    def load_in_blender(self, results_dir, i, mat):
        obj_path = os.path.join(results_dir, 'frame{:03d}.obj'.format(i))
        bpy.ops.import_scene.obj(filepath=obj_path)
        obj = bpy.context.selected_objects[0]
        print(f"Object {obj.name} imported")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        obj.location = self.get_root(i)
        obj.active_material = mat
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        bpy.ops.object.select_all(action='DESELECT')
        return obj.name

    def get_boader(self):
        shifted_vertices = (self.vertices + np.tile(self.positions[:, 0:1, :], (1, self.vertices.shape[1], 1)))[:, :, self.order]
        mins, maxs = np.min(shifted_vertices, axis=(0, 1)), np.max(shifted_vertices, axis=(0, 1))
        mins[2] = np.quantile(np.min(shifted_vertices, axis=1)[:, 2], 0.5)
        return (mins, maxs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True)
    config = parser.parse_args()

    # config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config.cuda = True if torch.cuda.is_available() else False
    config.smpl = 'smpl'

    assert config.npy_path.endswith('.npy')
    assert os.path.exists(config.npy_path)
    parsed_name = os.path.basename(config.npy_path).replace('.npy', '')
    results_dir_name = parsed_name + '_obj'
    results_dir = os.path.join(os.path.dirname(os.path.dirname(config.npy_path)), results_dir_name)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    npy2obj = npy2obj(config.npy_path, smpl_path=config.smpl)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.nframes)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)