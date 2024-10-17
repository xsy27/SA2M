### Bailando

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.visualizer import export

class AudMoTrainDataset(Dataset):
    def __init__(self, musics, motions, smpl_configs, interval):
        self.musics = musics
        self.motions = motions
        self.smpl_scalings = smpl_configs[0]
        self.smpl_transs = smpl_configs[1]
        self.mask = np.ones(interval, dtype=bool) if interval is not None else None

    def __len__(self):
        assert(len(self.musics) == len(self.motions))
        assert(len(self.smpl_scalings) == len(self.motions))
        assert(len(self.smpl_transs) == len(self.motions))
        return len(self.motions)

    def __getitem__(self, index):
        # print(self.motions[index].shape)
        return {
            'data': self.motions[index],
            'conditions': {
                'music': self.musics[index],
            },
            'configs': {
                'smpl_scaling': self.smpl_scalings[index],
                'smpl_trans': self.smpl_transs[index],
                'mask': self.mask
            }
        }
    
class AudMoTestDataset(Dataset):
    def __init__(self, musics, motions, names, smpl_configs, masks):
        self.musics = musics
        self.motions = motions
        self.names = names
        self.smpl_scalings = smpl_configs[0]
        self.smpl_transs = smpl_configs[1]
        self.masks = masks

    def __len__(self):
        assert(len(self.musics) == len(self.motions))
        assert(len(self.names) == len(self.motions))
        assert(len(self.smpl_scalings) == len(self.motions))
        assert(len(self.smpl_transs) == len(self.motions))
        assert(len(self.masks) == len(self.motions))
        return len(self.motions)
    
    def __getitem__(self, index):
        return {
            'data': self.motions[index],
            'conditions': {
                'music': self.musics[index],
            },
            'configs': {
                'name': self.names[index],
                'smpl_scaling': self.smpl_scalings[index],
                'smpl_trans': self.smpl_transs[index],
                'mask': self.masks[index]
            }
        }

def load_train_data_aist(data_dir, dtype=np.float32, interval=120, move=40, wav_padding=0):
    tot = 0
    music_data, motion_data, smpl_configs = [], [], []
    smpl_scalings, smpl_transs = [], []
    fnames = sorted(os.listdir(data_dir))
    # print(fnames)
    # fnames = fnames[:10]  # For debug
    
    if ".ipynb_checkpoints" in fnames:
        fnames.remove(".ipynb_checkpoints")
    print("Loading AIST++ data...")
    for fname in tqdm(fnames):
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            # print(path)
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'], dtype=dtype)
            np_motion = np.array(sample_dict['dance_array'], dtype=dtype)
            np_smpl_scaling = np.array(sample_dict['smpl_config'][0], dtype=dtype)
            np_smpl_trans = np.array(sample_dict['smpl_config'][1], dtype=dtype)

            music_sample_rate = 1
            if interval is not None:
                seq_len, dim = np_music.shape
                for i in range(0, seq_len, move):
                    i_sample = i // music_sample_rate
                    interval_sample = interval // music_sample_rate

                    music_sub_seq = np_music[i_sample: i_sample + interval_sample]
                    motion_sub_seq = np_motion[i: i + interval]
                    smpl_trans_sub_seq = np_smpl_trans[i: i + interval]
                    assert(len(motion_sub_seq) == len(smpl_trans_sub_seq))

                    if len(music_sub_seq) == interval_sample and len(motion_sub_seq) == interval:
                        padding_sample = wav_padding // music_sample_rate
                        # Add paddings/context of music
                        music_sub_seq_pad = np.zeros((interval_sample + padding_sample * 2, dim), dtype=music_sub_seq.dtype)
                        
                        if padding_sample > 0:
                            music_sub_seq_pad[padding_sample:-padding_sample] = music_sub_seq
                            start_sample = padding_sample if i_sample > padding_sample else i_sample
                            end_sample = padding_sample if i_sample + interval_sample + padding_sample < seq_len else seq_len - (i_sample + interval_sample)
                            # print(end_sample)
                            music_sub_seq_pad[padding_sample - start_sample:padding_sample] = np_music[i_sample - start_sample:i_sample]
                            if end_sample == padding_sample:
                                music_sub_seq_pad[-padding_sample:] = np_music[i_sample + interval_sample:i_sample + interval_sample + end_sample]
                            else:     
                                music_sub_seq_pad[-padding_sample:-padding_sample + end_sample] = np_music[i_sample + interval_sample:i_sample + interval_sample + end_sample]
                        else:
                            music_sub_seq_pad = music_sub_seq
                        music_data.append(music_sub_seq_pad)
                        motion_data.append(motion_sub_seq)
                        smpl_scalings.append(np_smpl_scaling)
                        smpl_transs.append(smpl_trans_sub_seq)
                        # tot += 1
                        # if tot > 1:
                        #     break
            else:
                music_data.append(np_music)
                motion_data.append(np_motion)
                smpl_scalings.append(np_smpl_scaling)
                smpl_transs.append(np_smpl_trans)

            # tot += 1
            # if tot > 1:
            #     break
            
            # tot += 1
            # if tot > 30:
            #     break

    # music_data = torch.from_numpy(np.array(music_data, dtype=dtype))
    # motion_data = torch.from_numpy(np.array(motion_data, dtype=dtype))
    smpl_configs = [smpl_scalings, smpl_transs]

    return music_data, motion_data, smpl_configs

def load_val_data_aist(data_dir, save_dir, interval, dtype=np.float32):
    tot = 0
    input_names = []

    music_data, motion_data, smpl_configs, masks = [], [], [], []
    smpl_scalings, smpl_transs = [], []
    fnames = sorted(os.listdir(data_dir))
    
    '''
    print("Loading AIST++ validation data...")
    for fname in tqdm(fnames):
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'], dtype=dtype)
            np_smpl_scaling = np.array(sample_dict['smpl_config'][0], dtype=dtype)
            np_smpl_trans = np.array(sample_dict['smpl_config'][1], dtype=dtype)

            if 'dance_array' in sample_dict:
                np_motion = np.array(sample_dict['dance_array'], dtype=dtype)

                assert(len(np_motion) == len(np_smpl_trans))
                for kk in range((len(np_motion) // move + 1) * move - len(np_motion)):
                    np_motion = np.append(np_motion, np_motion[-1:], axis=0)
                for kk in range((len(np_smpl_trans) // move + 1) * move - len(np_smpl_trans)):
                    np_smpl_trans = np.append(np_smpl_trans, np_smpl_trans[-1:], axis=0)

                motion_data.append(np_motion)
            else:
                np_motion = None
                motion_data = None
            
            music_move = move

            # zero padding left
            for kk in range(wav_padding):
                np_music = np.append(np.zeros_like(np_music[-1:]), np_music, axis=0)
            # fully devisable
            for kk in range((len(np_music) // music_move + 1) * music_move - len(np_music) ):
                np_music = np.append(np_music, np_music[-1:], axis=0)
            # zero padding right
            for kk in range(wav_padding):
                np_music = np.append(np_music, np.zeros_like(np_music[-1:]), axis=0)
            music_data.append(np_music)

            input_names.append(fname[:-5])
            smpl_scalings.append(np_smpl_scaling)
            smpl_transs.append(np_smpl_trans)
            masks.append(np.ones(len(np_motion), dtype=bool))
            
            tot += 1
            if tot > 5:
                break
    '''
    
    print("Loading AIST++ validation data...")
    for fname in tqdm(fnames):
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            # print(path)
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'], dtype=dtype)
            np_motion = np.array(sample_dict['dance_array'], dtype=dtype)
            np_smpl_scaling = np.array(sample_dict['smpl_config'][0], dtype=dtype)
            np_smpl_trans = np.array(sample_dict['smpl_config'][1], dtype=dtype)

            if interval is not None:
                music_data.append(np_music[:interval])
                motion_data.append(np_motion[:interval])
                smpl_transs.append(np_smpl_trans[:interval])
                masks.append(np.ones(interval, dtype=bool))
                        
            else:
                music_data.append(np_music)
                motion_data.append(np_motion)
                smpl_transs.append(np_smpl_trans)
                masks.append(np.ones(len(np_motion), dtype=bool))
            
            smpl_scalings.append(np_smpl_scaling)
            input_names.append(fname[:-5])

        # tot += 1
        # if tot > 1:
        #     break
    
    smpl_configs = [smpl_scalings, smpl_transs]

    save_gt(motion_data, input_names, '%s/%s/' % (save_dir, 'gt'))

    return music_data, motion_data, input_names, smpl_configs, masks

def load_test_data_aist(data_dir, save_dir, interval, dtype=np.float32):
    tot = 0
    input_names = []

    music_data, motion_data, smpl_configs, masks = [], [], [], []
    smpl_scalings, smpl_transs = [], []
    fnames = sorted(os.listdir(data_dir))

    print("Loading AIST++ test data...")
    for fname in tqdm(fnames):
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            # print(path)
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'], dtype=dtype)
            np_motion = np.array(sample_dict['dance_array'], dtype=dtype)
            np_smpl_scaling = np.array(sample_dict['smpl_config'][0], dtype=dtype)
            np_smpl_trans = np.array(sample_dict['smpl_config'][1], dtype=dtype)

            if interval is not None:
                music_data.append(np_music[:interval])
                motion_data.append(np_motion[:interval])
                smpl_transs.append(np_smpl_trans[:interval])
                masks.append(np.ones(interval, dtype=bool))
                        
            else:
                music_data.append(np_music)
                motion_data.append(np_motion)
                smpl_transs.append(np_smpl_trans)
                masks.append(np.ones(len(np_motion), dtype=bool))
            
            smpl_scalings.append(np_smpl_scaling)
            input_names.append(fname[:-5])

        # tot += 1
        # if tot > 1:
        #     break
    
    smpl_configs = [smpl_scalings, smpl_transs]

    save_gt(motion_data, input_names, '%s/%s/' % (save_dir, 'gt'))

    return music_data, motion_data, input_names, smpl_configs, masks

def prepare_train_dataloader(config, dtype=np.float32):
    train_music_data, train_motion_data, train_smpl_configs = load_train_data_aist(
        config.data, dtype=dtype, interval=config.dataset.clip_len, move=config.dataset.move)
    vec_len, audio_dim = train_motion_data[0].shape[-1], train_music_data[0].shape[-1]
    data = AudMoTrainDataset(train_music_data, train_motion_data, train_smpl_configs, config.dataset.clip_len)
    sampler = torch.utils.data.RandomSampler(data, replacement=True)
    data_loader = torch.utils.data.DataLoader(
        data,
        num_workers=config.trainer.workers,
        batch_size=config.trainer.batch_size,
        sampler=sampler,
        pin_memory=True
    )
    return data_loader, vec_len, audio_dim

def prepare_val_dataloader(config, dtype=np.float32):
    val_music_data, val_motion_data, val_names, val_smpl_configs, val_masks = load_val_data_aist(
        config.val_data, config.save, config.dataset.clip_len, dtype=dtype)
    data = AudMoTestDataset(val_music_data, val_motion_data, val_names, val_smpl_configs, val_masks)
    data_loader = torch.utils.data.DataLoader(
        data,
        num_workers=config.trainer.workers,
        batch_size=1,
        shuffle=False
    )
    return data_loader

def prepare_test_dataloader(config, dtype=np.float32):
    test_music_data, test_motion_data, test_names, test_smpl_configs, test_masks = load_test_data_aist(
        config.test_data, config.save, config.dataset.clip_len, dtype=dtype)
    vec_len, audio_dim = test_motion_data[0].shape[-1], test_music_data[0].shape[-1]
    data = AudMoTestDataset(test_music_data, test_motion_data, test_names, test_smpl_configs, test_masks)
    data_loader = torch.utils.data.DataLoader(
        data,
        num_workers=config.workers,
        batch_size=1,
        shuffle=False
    )
    return data_loader, vec_len, audio_dim

def save_gt(motions, input_names, save_dir):
    x_start_poss = []
    for motion in motions:
        interval = motion.shape[0]
        x_start_pos = np.concatenate((motion[..., 3:6], motion[..., 12:81]), axis=-1).reshape(interval, 24, 3)
        global_shift = np.expand_dims(x_start_pos[0, 0, :].copy(), axis=(0, 1))
        x_start_pos = x_start_pos - np.tile(global_shift, (interval, 24, 1)) # [bs, nframes, 24, 3]
        x_start_poss.append(x_start_pos)

    export(x_start_poss, input_names, save_dir, prefix='gt')

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_music_data, train_motion_data, train_smpl_configs = load_train_data_aist('aistpp_train_wav', interval=240, move=8)
    train_dataloader = prepare_train_dataloader(4, train_music_data, train_motion_data, train_smpl_configs, batch_size=128, interval=240)
    for batch_i, batch in enumerate(train_dataloader):
        if batch_i==0:
            print(f"{batch_i}: motion:{batch['data'].shape}\n\
                | music:{batch['conditions']['music'].shape}\n\
                | smpl_scaling:{batch['configs']['smpl_scaling'].shape}\n\
                | smpl_trans:{batch['configs']['smpl_trans'].shape}\n\
                | mask:{batch['configs']['mask'].shape}")
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # val_music_data, val_motion_data, val_names, val_smpl_configs = load_val_data_aist('aistpp_val_wav', move=8)
    # train_dataloader = prepare_val_dataloader(4, val_music_data, val_motion_data, val_names, val_smpl_configs)
    # for batch_i, batch in enumerate(train_dataloader):
    #     if batch_i==0:
    #         print(f"{batch_i}: motion:{batch['data'].shape}\n\
    #                | music:{batch['conditions']['music'].shape}\n\
    #                | name:{batch['configs']['name'][0]}\n\
    #                | smpl_scaling:{batch['configs']['smpl_scaling'].shape}\n\
    #                | smpl_trans:{batch['configs']['smpl_trans'].shape}")
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # val_music_data, val_motion_data, val_names, val_smpl_configs = load_val_data_aist('aistpp_test_wav', move=8)
    # train_dataloader = prepare_val_dataloader(4, val_music_data, val_motion_data, val_names, val_smpl_configs)
    # for batch_i, batch in enumerate(train_dataloader):
    #     if batch_i==0:
    #         print(f"{batch_i}: motion:{batch['data'].shape}\n\
    #                | music:{batch['conditions']['music'].shape}\n\
    #                | name:{batch['configs']['name'][0]}\n\
    #                | smpl_scaling:{batch['configs']['smpl_scaling'].shape}\n\
    #                | smpl_trans:{batch['configs']['smpl_trans'].shape}")