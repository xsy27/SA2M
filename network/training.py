import numpy as np
import blobfile as bf
import utils.common as common
from tqdm import tqdm
import utils.nn_transforms as nn_transforms
import itertools

import torch
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader
from torch_ema import ExponentialMovingAverage

from diffusion.resample import create_named_schedule_sampler
from diffusion.gaussian_diffusion import *

from utils.visualizer import export
from utils.nn_transforms import repr6d2quat
from pytorch3d import transforms

from smplx import SMPL

import multiprocessing
from multiprocessing import Pool
from functools import partial

class BaseTrainingPortal:
    def __init__(self, config, model, diffusion, train_dataloader, val_dataloader, logger, tb_writer, prior_loader=None):
        
        self.model = model
        self.diffusion = diffusion
        self.dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.tb_writer = tb_writer
        self.config = config
        self.batch_size = config.trainer.batch_size
        self.lr = config.trainer.lr
        self.lr_anneal_steps = config.trainer.lr_anneal_steps

        self.epoch = 0
        self.num_epochs = config.trainer.epoch
        self.save_freq = config.trainer.save_freq
        self.best_loss = 1e10
        
        print('Train with %d epoches, %d batches by %d batch_size' % (self.num_epochs, len(self.dataloader), self.batch_size))

        self.save_dir = config.save

        self.smpl = SMPL(model_path=self.config.dataset.smpl_dir, gender='MALE', batch_size=1).eval().to(config.device)
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=config.trainer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs, eta_min=self.lr * 0.1)
        
        if config.trainer.ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        
        self.device = config.device

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.use_ddp = False
        
        self.prior_loader = prior_loader

        self.pose_vec = config.dataset.pose_vec
        
    def diffuse(self, x_start, t, cond, data_config, noise=None, return_loss=False):
        raise NotImplementedError('diffuse function must be implemented')

    def evaluate_sampling(self, dataloader, save_folder_name):
        raise NotImplementedError('evaluate_sampling function must be implemented')
    
    def evaluate(self, dataloader, save_folder_name):
        raise NotImplementedError('evaluate function must be implemented')
    
        
    def run_loop(self):
        self.evaluate(self.val_dataloader, save_folder_name='init_validation')
        
        epoch_process_bar = tqdm(range(self.epoch, self.num_epochs), desc=f'Epoch {self.epoch}')
        for epoch_idx in epoch_process_bar:
            self.model.train()
            self.model.training = True
            self.epoch = epoch_idx + 1
            epoch_losses = {}
            
            for datas in self.dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                data_config = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['configs'].items()}
                x_start = datas['data']

                self.opt.zero_grad()
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                
                _, losses = self.diffuse(x_start, t, cond, data_config, noise=None, return_loss=True)
                total_loss = (losses["loss"] * weights).mean()
                total_loss.backward()
                self.opt.step()
            
                if self.config.trainer.ema:
                    self.ema.update()
                
                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in epoch_losses.keys():
                            epoch_losses[key_name] = []
                        epoch_losses[key_name].append(losses[key_name].mean().item())
            
            loss_str = ''
            for key in epoch_losses.keys():
                loss_str += f'{key}: {np.mean(epoch_losses[key]):.6f}, '
            
            epoch_avg_loss = np.mean(epoch_losses['loss'])
            
            if self.epoch > 10 and epoch_avg_loss < self.best_loss:                
                self.save_checkpoint(filename='best')
            
            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss
            
            epoch_process_bar.set_description(f'Epoch {self.epoch}/{self.config.trainer.epoch} | loss: {epoch_avg_loss:.6f} | best_loss: {self.best_loss:.6f}')
            self.logger.info(f'Epoch {self.epoch}/{self.config.trainer.epoch} | {loss_str} | best_loss: {self.best_loss:.6f}')

            self.evaluate(self.val_dataloader, save_folder_name=f'validation_{self.epoch}')
                        
            if self.epoch > 0 and self.epoch % self.config.trainer.save_freq == 0:
                self.save_checkpoint(filename=f'weights_{self.epoch}')
            
            for key_name in epoch_losses.keys():
                if 'loss' in key_name:
                    self.tb_writer.add_scalar(f'train/{key_name}', np.mean(epoch_losses[key_name]), self.epoch)

            self.scheduler.step()
        
        best_path = '%s/best.pt' % (self.config.save)
        self.load_checkpoint(best_path)
        self.evaluate(self.val_dataloader, save_folder_name='best')


    def state_dict(self):
        model_state = self.model.state_dict()
        opt_state = self.opt.state_dict()
            
        return {
            'epoch': self.epoch,
            'state_dict': model_state,
            'opt_state_dict': opt_state,
            'config': self.config,
            'loss': self.best_loss,
        }

    def save_checkpoint(self, filename='weights'):
        save_path = '%s/%s.pt' % (self.config.save, filename)
        with bf.BlobFile(bf.join(save_path), "wb") as f:
            torch.save(self.state_dict(), f)
        self.logger.info(f'Saved checkpoint: {save_path}')


    def load_checkpoint(self, resume_checkpoint, load_hyper=True):
        if bf.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            if load_hyper:
                self.epoch = checkpoint['epoch'] + 1
                self.best_loss = checkpoint['loss']
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
            self.logger.info('\nLoad checkpoint from %s, start at epoch %d, loss: %.4f' % (resume_checkpoint, self.epoch, checkpoint['loss']))
        else:
            raise FileNotFoundError(f'No checkpoint found at {resume_checkpoint}')


class MotionTrainingPortal(BaseTrainingPortal):
    def __init__(self, config, model, diffusion, train_dataloader, val_dataloader, logger, tb_writer, finetune_loader=None):
        super().__init__(config, model, diffusion, train_dataloader, val_dataloader, logger, tb_writer, finetune_loader)

    def diffuse(self, x_start, t, cond, data_config, noise=None, return_loss=False, evaluate=False):
        batch_size, frame_num, vec_len = x_start.shape
        x_start = x_start.permute(0, 2, 1) # [bs, vec_len, frame_num]
        
        if noise is None:
            noise = th.randn_like(x_start)
        
        cond['music'] = cond['music'].permute((0, 2, 1)) # [bs, feature_num, frame_num]
        
        if not evaluate:
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            model_output = self.model.interface(x_t, self.diffusion._scale_timesteps(t), cond)
        else:
            x_t = th.randn_like(x_start)
            conditions = {'y': cond}
            data_shape = x_start.shape
            model_output = self.diffusion.p_sample_loop(self.model, data_shape, clip_denoised=False, model_kwargs=conditions, skip_timesteps=0,
                                                        init_image=None, dump_steps=None, noise=None, const_noise=False)
        assert ((batch_size, vec_len, frame_num) == model_output.shape)
        
        if return_loss:
            loss_terms = {}
            
            if self.diffusion.model_var_type in [ModelVarType.LEARNED,  ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                loss_terms["vb"] = self.diffusion._vb_terms_bpd(model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t, clip_denoised=False)["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0
            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.diffusion.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            mask = data_config['mask'].view(batch_size, 1, -1)
            
            if self.config.trainer.use_loss_mse:
                loss_terms['loss_data'] = self.diffusion.masked_l2(target, model_output, mask)
                
            if self.config.trainer.use_loss_delta:
                model_output_vel = model_output[..., 1:] - model_output[..., :-1]
                target_vel = target[..., 1:] - target[..., :-1]
                loss_terms['loss_data_delta'] = self.diffusion.masked_l2(target_vel, model_output_vel, mask[..., 1:])

            target = target.permute(0, 2, 1) # [batch_size, frame_num, vec_len]
            model_output = model_output.permute(0, 2, 1) # [batch_size, frame_num, vec_len]
            mask = mask.view(batch_size, 1, 1, -1) # [batch_size, 1, 1, frame_num]
            target_rot_6d = torch.cat((target[..., 6:12], target[..., 150:288]), dim=-1).view(batch_size, 24, 6, frame_num)
            model_output_rot_6d = torch.cat((model_output[..., 6:12], model_output[..., 150:288]), dim=-1).view(batch_size, 24, 6, frame_num)

            # if self.config.trainer.use_loss_mse:
            #     loss_terms['loss_data'] = self.diffusion.masked_l2(target_rot_6d, model_output_rot_6d, mask)
            
            # if self.config.trainer.use_loss_delta:
            #     target_rot_6d_vel = target_rot_6d[..., 1:] - target_rot_6d[..., :-1]
            #     model_output_rot_6d_vel = model_output_rot_6d[..., 1:] - model_output_rot_6d[..., :-1]
            #     loss_terms['loss_data_delta'] = self.diffusion.masked_l2(target_rot_6d_vel, model_output_rot_6d_vel, mask[..., 1:])
            
            if self.config.trainer.use_loss_contact:
                target_xyz = torch.cat((target[..., 3:6], target[..., 12:81]), dim=-1).reshape(batch_size, frame_num, 24, 3)

                model_output_rot = transforms.quaternion_to_axis_angle(
                    repr6d2quat(torch.cat((model_output[..., 6:12], model_output[..., 150:288]), dim=-1).view(batch_size, frame_num, 24, 6))).float() # [bs, nframes, 24, 3]
                
                pred_xyz = []
                for i in range(batch_size):
                    model_output_pos = self.smpl.forward(
                        global_orient=model_output_rot[i][:, 0:1],
                        body_pose=model_output_rot[i][:, 1:],
                        transl=(data_config['smpl_trans'][i] / data_config['smpl_scaling'][i]).float(),
                        ).joints.cpu().detach()[:, 0:24, :].unsqueeze(0)
                    pred_xyz.append(model_output_pos)
                pred_xyz = torch.cat(pred_xyz, dim=0).to(target_xyz.device)

                assert pred_xyz.shape == target_xyz.shape

                loss_terms["loss_geo_xyz"] = self.diffusion.masked_l2(target_xyz.permute(0, 2, 3, 1), pred_xyz.permute(0, 2, 3, 1), mask)

                target_xyz_vel = target_xyz[:, 1:] - target_xyz[:, :-1]
                pred_xyz_vel = pred_xyz[:, 1:] - pred_xyz[:, :-1]
                loss_terms["loss_geo_xyz_vel"] = self.diffusion.masked_l2(target_xyz_vel.permute(0, 2, 3, 1), pred_xyz_vel.permute(0, 2, 3, 1), mask[..., 1:])

                # loss_terms["loss_self"] = self.diffusion.masked_l2(
                #     torch.cat((model_output[..., 3:6], model_output[..., 12:81]), dim=-1).reshape(batch_size, frame_num, 24, 3).permute(0, 2, 3, 1),
                #     pred_xyz.permute(0, 2, 3, 1),
                #     mask
                # ) + \
                # self.diffusion.masked_l2(
                #     model_output[..., 81:150].reshape(batch_size, frame_num, 23, 3).permute(0, 2, 3, 1)[..., :-1],
                #     pred_xyz_vel[:, :, 1:, :].permute(0, 2, 3, 1),
                #     mask[..., 1:]
                # )

                l_foot_idx, r_foot_idx = 10, 11
                l_ankle_idx, r_ankle_idx = 7, 8
                relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
                target_xyz_reshape = target_xyz.permute(0, 2, 3, 1)
                pred_xyz_reshape = pred_xyz.permute(0, 2, 3, 1)
                gt_joint_xyz = target_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
                fc_mask = torch.unsqueeze((gt_joint_vel <= 5e-3), dim=2).repeat(1, 1, 3, 1)
                pred_joint_xyz = pred_xyz_reshape[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                pred_vel[~fc_mask] = 0
                loss_terms["loss_foot_contact"] = self.diffusion.masked_l2(pred_vel,
                                            torch.zeros(pred_vel.shape, device=pred_vel.device),
                                            mask[:, :, :, 1:])
                
                target = target.permute(0, 2, 1)
                model_output = model_output.permute(0, 2, 1)
            
            loss_terms["loss"] = loss_terms.get('vb', 0.) + \
                            loss_terms.get('loss_data', 0.) + \
                            loss_terms.get('loss_data_delta', 0.) + \
                            loss_terms.get('loss_geo_xyz', 0) + \
                            loss_terms.get('loss_geo_xyz_vel', 0) + \
                            loss_terms.get('loss_foot_contact', 0.) + \
                            loss_terms.get('loss_self', 0.)
            
            return model_output.permute(0, 2, 1), loss_terms
        
        return model_output.permute(0, 2, 1)

    def evaluate(self, dataloader, save_folder_name):
        print("Evaluation...")
        self.model.eval()
        self.model.training = False
        with torch.no_grad():
            pred = []
            pred_rot = []
            fnames = []
            eval_losses = {}

            for datas in dataloader:
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['conditions'].items()}
                data_config = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas['configs'].items()}

                x_start = datas['data'] # [bs, nframes, vec_len]
                bs, nframes, vec_len = x_start.shape
                t, _ = self.schedule_sampler.sample(dataloader.batch_size, self.device)
                model_output, losses = self.diffuse(x_start, t, cond, data_config, noise=None, return_loss=True, evaluate=True)
                assert model_output.shape == x_start.shape

                if self.epoch % self.config.trainer.eval_freq == 0:
                    model_output_rot = transforms.quaternion_to_axis_angle(repr6d2quat(torch.cat((model_output[..., 6:12], model_output[..., 150:288]), dim=-1).view(bs, nframes, 24, 6))).float() # [bs, nframes, 24, 3]
                    
                    for i in range(bs):
                        fname = data_config['name'][i]
                        model_output_pos = self.smpl.forward(
                            global_orient=model_output_rot[i][:, 0:1].float(),
                            body_pose=model_output_rot[i][:, 1:].float(),
                            transl=(data_config['smpl_trans'][i] / data_config['smpl_scaling'][i]).float(),
                            ).joints.cpu().detach().numpy()[:, 0:24, :]
                        # global_shift = np.expand_dims(model_output_pos[0, 0, :].copy(), axis=(0, 1))
                        # model_output_pos = model_output_pos - np.tile(global_shift, (nframes, 24, 1))

                        pred.append(model_output_pos)
                        pred_rot.append(model_output_rot[i].detach().cpu().numpy())
                        fnames.append(fname)
                
                for key_name in losses.keys():
                    if 'loss' in key_name:
                        if key_name not in eval_losses.keys():
                            eval_losses[key_name] = []
                        eval_losses[key_name].append(losses[key_name].mean().item())
                
            for key_name in eval_losses.keys():
                if 'loss' in key_name:
                    self.tb_writer.add_scalar(f'eval/{key_name}', np.mean(eval_losses[key_name]), self.epoch)
            
            # if save_folder_name == 'init_validation':
            #     export(gt, fnames, '%s/%s/' % (self.save_dir, save_folder_name), 'gt')
            if self.epoch % self.config.trainer.eval_freq == 0:
                common.mkdir('%s/%s' % (self.save_dir, save_folder_name))
                export(pred, pred_rot, fnames, '%s/%s/' % (self.save_dir, save_folder_name), prefix='pred')
        