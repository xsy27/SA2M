import os
import argparse
from argparse import Namespace
import torch
from tqdm import tqdm
import json
from network.models import MotionDiffusion
import numpy as np
import blobfile as bf
from pytorch3d import transforms
from utils.nn_transforms import repr6d2quat
from smplx import SMPL
from utils.visualizer import export
import shutil
from diffusion.create_diffusion import create_gaussian_diffusion

def get_input_dims(config, fname, dtype=np.float32):
    path = os.path.join(config.test_data, fname)
    with open(path) as f:
        sample_dict = json.loads(f.read())
        np_music = np.array(sample_dict['music_array'], dtype=dtype)
        np_motion = np.array(sample_dict['dance_array'], dtype=dtype)
        audio_dim, vec_len = np_music.shape[-1], np_motion.shape[-1]
    return vec_len, audio_dim

def make_gt_data(config, fnames, dtype=np.float32):
    gt = []

    smpl = SMPL(model_path=config.smpl_dir, gender='MALE', batch_size=1)

    for fname in tqdm(fnames):
        path = os.path.join(config.test_data, fname)
        fname = fname[:-5]
        
        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_motion = np.array(sample_dict['dance_array'], dtype=dtype)[np.newaxis, ...] # [1, nframes, vec_len]

            bs, nframes, vec_len = np_motion.shape

            x_start_pos = np.concatenate((np_motion[..., 3:6], np_motion[..., 12:81]), axis=-1).reshape(nframes, 24, 3)
            global_shift = np.expand_dims(x_start_pos[0, 0, :].copy(), axis=(0, 1))
            x_start_pos = x_start_pos - np.tile(global_shift, (nframes, 24, 1)) # [nframes, 24, 3]
            gt.append(x_start_pos)
    
    fnames = [fname[:-5] for fname in fnames]
    export(gt, fnames, '%s/%s/' % (config.save_dir, 'gt'), prefix='gt')

def make_pred_data(config, model, fnames, dtype=np.float32):
    tot = 0
    bs = 1
    pred = []
    interval = 240

    smpl = SMPL(model_path=config.smpl_dir, gender='MALE', batch_size=1)
    diffusion = create_gaussian_diffusion(config)

    for fname in tqdm(fnames):
        path = os.path.join(config.test_data, fname)
        fname = fname[:-5]
        
        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'], dtype=dtype)[:interval][np.newaxis, ...] # [1, nframes_music, audio_dim]
            np_motion = np.array(sample_dict['dance_array'], dtype=dtype)[:interval][np.newaxis, ...] # [1, nframes_motion, vec_len]
            np_smpl_scaling = np.array(sample_dict['smpl_config'][0], dtype=dtype)
            np_smpl_trans = np.array(sample_dict['smpl_config'][1], dtype=dtype)[:interval]
        
        with torch.no_grad():
            np_motion = np.transpose(np.tile(np_motion, (bs, 1, 1)), (0, 2, 1)) # [bs, vec_len, nframes]
            np_music = np.transpose(np.tile(np_music, (bs, 1, 1)), (0, 2, 1)) # [bs, audio_dim, nframes]

            w = np.ones([800])
            p = w / np.sum(w)
            t = torch.from_numpy(np.random.choice(len(p), size=(bs,), p=p)).long().to(config.device)

            x_start = torch.randn_like(torch.from_numpy(np_motion)).to(config.device) # [bs, vec_len, nframes]
            # x_start = torch.from_numpy(np_motion).to(config.device)
            x_t = diffusion.q_sample(x_start, t) # [bs, vec_len, nframes]

            cond = {
                'music': torch.from_numpy(np_music).to(config.device) # [bs, nframes, audio_dim]
            }

            model_output = model.interface(x_t, diffusion._scale_timesteps(t), cond).permute(0, 2, 1) # (bs, nframes, vec_len)

            bs, nframes, vec_len = model_output.shape

            model_output_rot = transforms.quaternion_to_axis_angle(
                repr6d2quat(torch.cat((model_output[..., 6:12], model_output[..., 150:288]), dim=-1).view(bs, nframes, 24, 6)))\
                    .cpu().detach().float()[0] # [nframes, 24, 3]
            model_output_pos = smpl.forward(
                global_orient=model_output_rot[:, 0:1].float(),
                body_pose=model_output_rot[:, 1:].float(),
                transl=torch.from_numpy(np_smpl_trans / np_smpl_scaling).float(),
                ).joints.cpu().detach().numpy()[:, 0:24, :]
            global_shift = np.expand_dims(model_output_pos[0, 0, :].copy(), axis=(0, 1))
            model_output_pos = model_output_pos - np.tile(global_shift, (nframes, 24, 1))
            pred.append(model_output_pos)
        tot += 1
        if tot >= 5:
            fnames = fnames[:tot]
            break
    
    fnames = [fname[:-5] for fname in fnames]
    export(pred, fnames, '%s/%s/' % (config.save_dir, 'pred'), prefix='pred')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_data', type=str, default='data/aistpp_test_wav', help='input local dictionary for AIST++ evaluation dataset.')
    parser.add_argument('--smpl_dir', type=str, default='smpl', help='input local dictionary that stores SMPL data.')
    parser.add_argument('--best', type=str, default='save/a2m2_aistpp_train_wav/weights_50.pt', help='input local dictionary that stores the best model parameters.')
    parser.add_argument('--save_dir', type=str, default='data/aistpp_eval_data', help='input local dictionary that stores the output data.')
    config = parser.parse_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.diff = Namespace()
    config.diff.diffusion_steps = 800
    config.diff.noise_schedule = 'cosine'
    config.diff.sigma_small = True
    
    # if os.path.exists(config.save_dir):
    #     shutil.rmtree(config.save_dir, ignore_errors=True)
    
    os.makedirs(config.save_dir, exist_ok=True)
    fnames = sorted(os.listdir(config.test_data))

    vec_len, audio_dim = get_input_dims(config, fnames[0])

    model = MotionDiffusion('complex', vec_len, audio_dim).to(config.device)
    model.eval()
    
    if bf.exists(config.best):
        best_model = torch.load(config.best)
        model.load_state_dict(best_model['state_dict'])
    
    # make_gt_data(config, fnames)
    make_pred_data(config, model, fnames)