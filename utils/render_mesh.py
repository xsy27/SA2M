import argparse
import os
import numpy as np
import torch
import shutil
from tqdm import tqdm
from smplx import SMPL
from trimesh import Trimesh

class npy2obj:
    def __init__(self, npy_path, smpl_path):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True).item()['rotation'] # [frame_num, 24, 3]
        self.smpl = SMPL(model_path=smpl_path, gender='MALE').eval()
        self.faces = self.smpl.faces
        self.nframes, self.njoints, self.nfeats = self.motions.shape

        self.rotations = torch.from_numpy(self.motions)
        self.global_orient = self.rotations[:, 0:1]
        self.rotations = self.rotations[:, 1:]
        betas = torch.zeros([self.rotations.shape[0], self.smpl.num_betas], dtype=self.rotations.dtype, device=self.rotations.device)

        self.out = self.smpl.forward(body_pose=self.rotations, global_orient=self.global_orient, betas=betas)
        self.joints = self.out.joints.detach().cpu().numpy()
        self.global_shift = np.expand_dims(self.joints[0, 0, :].copy(), axis=(0, 1))
        self.vertices = self.out.vertices.detach().cpu().numpy() - self.global_shift

    def get_trimesh(self, frame_i):
        return Trimesh(vertices=self.vertices[frame_i],
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
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