import os
import argparse
from argparse import Namespace
import torch
from tqdm import tqdm
from network.models import MotionDiffusion
import numpy as np
import blobfile as bf
from pytorch3d import transforms
from utils.nn_transforms import repr6d2quat
from smplx import SMPL
from utils.visualizer import export
import shutil
from data.dataloader import prepare_test_dataloader
from diffusion.create_diffusion import create_gaussian_diffusion

def predict(config, dataloader, model, diffusion):
    bs = 1
    pred = []
    fnames = []

    smpl = SMPL(model_path=config.smpl, gender='MALE', batch_size=bs).eval().to(config.device)

    for datas in tqdm(dataloader):
        datas = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas.items()}
        cond = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
        data_config = {key: val.to(config.device) if torch.is_tensor(val) else val for key, val in datas['configs'].items()}
        x_start = datas['data'].permute(0, 2, 1) # [bs, vec_len, frame_num]
        cond['music'] = cond['music'].permute(0, 2, 1) # [bs, feature_num, frame_num]
        conditions = {'y': cond}
        data_shape = x_start.shape

        # x_t = torch.randn_like(x_start).to(config.device).permute(0, 2, 1) # [bs, vec_len, nframes]

        # w = np.ones([config.diff.diffusion_steps])
        # p = w / np.sum(w)
        # t = torch.from_numpy(np.random.choice(len(p), size=(bs,), p=p)).long().to(config.device)

        # model_output = diffusion.q_sample(x_start, t)

        with torch.no_grad():
            # model_output = model.interface(x_t, diffusion._scale_timesteps(t), cond).permute(0, 2, 1) # [bs, nframes, vec_len]
            model_output = diffusion.p_sample_loop(model, data_shape, clip_denoised=False, model_kwargs=conditions, skip_timesteps=0,
                                        init_image=None, progress=True, dump_steps=None, noise=None, const_noise=False)
            model_output = model_output.permute(0, 2, 1)
            bs, nframes, vec_len = model_output.shape
            assert (nframes == config.dataset.clip_len)

            model_output_rot = transforms.quaternion_to_axis_angle(
                repr6d2quat(torch.cat((model_output[..., 6:12], model_output[..., 150:288]), dim=-1).view(bs, nframes, 24, 6))).float() # [bs, nframes, 24, 3]
            for i in range(bs):
                fname = data_config['name'][i]
                model_output_pos = smpl.forward(
                    global_orient=model_output_rot[i][:, 0:1].float(),
                    body_pose=model_output_rot[i][:, 1:].float(),
                    transl=(data_config['smpl_trans'][i] / data_config['smpl_scaling'][i]).float(),
                    ).joints.cpu().detach().numpy()[:, 0:24, :]
                global_shift = np.expand_dims(model_output_pos[0, 0, :].copy(), axis=(0, 1))
                model_output_pos = model_output_pos - np.tile(global_shift, (nframes, 24, 1))
                pred.append(model_output_pos)
                fnames.append(fname)

    export(pred, fnames, '%s/%s/' % (config.save, 'pred'), prefix='pred')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_data', type=str, default='data/aistpp_test_wav', help='input local dictionary for AIST++ evaluation dataset.')
    parser.add_argument('--smpl', type=str, default='smpl', help='input local dictionary that stores SMPL data.')
    parser.add_argument('--best', type=str, default='save/gamma_aistpp_train_wav/best.pt', help='input local dictionary that stores the best model parameters.')
    parser.add_argument('--save', type=str, default='data/aistpp_evaluation', help='input local dictionary that stores the output data.')
    config = parser.parse_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.workers = 4

    config.dataset = Namespace()
    config.dataset.clip_len = 240

    config.diff = Namespace()
    config.diff.diffusion_steps = 1000
    config.diff.noise_schedule = 'cosine'
    config.diff.sigma_small = True

    if os.path.exists(config.save):
        shutil.rmtree(config.save, ignore_errors=True)
    os.makedirs(config.save, exist_ok=True)

    test_data_loader, vec_len, audio_dim = prepare_test_dataloader(config)

    model = MotionDiffusion('complex', vec_len, audio_dim).to(config.device)
    if bf.exists(config.best):
        best_model = torch.load(config.best)
        model.load_state_dict(best_model['state_dict'])
    model.eval()

    diffusion = create_gaussian_diffusion(config)
    
    predict(config, test_data_loader, model, diffusion)