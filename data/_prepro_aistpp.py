# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


### Bailando

import os
import json
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import numpy as np
from data.extractor import FeatureExtractor
from aistplusplus_api.aist_plusplus.loader import AISTDataset
from smplx import SMPL
import torch
from utils.nn_transforms import axis_angle_to_matrix, matrix_to_rotation_6d, y_axis_angular_vel_from_marix
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_dir', type=str, default='aist_plusplus_final/all_musics')
parser.add_argument('--input_annotation_dir', type=str, default='aist_plusplus_final')
parser.add_argument('--smpl_dir', type=str, default='smpl')

parser.add_argument('--train_dir', type=str, default='data/aistpp_train_wav')
parser.add_argument('--val_dir', type=str, default='data/aistpp_val_wav')
parser.add_argument('--test_dir', type=str, default='data/aistpp_test_wav')

parser.add_argument('--split_train_file', type=str, default='aist_plusplus_final/splits/crossmodal_train.txt')
parser.add_argument('--split_val_file', type=str, default='aist_plusplus_final/splits/crossmodal_val.txt')
parser.add_argument('--split_test_file', type=str, default='aist_plusplus_final/splits/crossmodal_test.txt')

parser.add_argument('--sampling_rate', type=int, default=15360*2)
args = parser.parse_args()

extractor = FeatureExtractor()

if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)
if not os.path.exists(args.val_dir):
    os.mkdir(args.val_dir)
if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)

split_train_file = args.split_train_file
split_val_file = args.split_val_file
split_test_file = args.split_test_file

def make_music_dance_set(video_dir, annotation_dir):
    print('---------- Extract features from raw audio ----------')
    # print(annotation_dir)
    aist_dataset = AISTDataset(annotation_dir) 

    # musics = []
    # dances = []
    fnames = []
    train = []
    val = []
    test = []
    # smpl_configs = []

    # music_dance_keys = []
    # onset_beats = []
    # audio_fnames = sorted(os.listdir(video_dir))
    # dance_fnames = sorted(os.listdir(dance_dir))
    # audio_fnames = audio_fnames[:20]  # for debug
    # print(f'audio_fnames: {audio_fnames}')
    num_musics, num_dances, num_smpl_configs = 0, 0, 0

    train_file = open(split_train_file, 'r')
    for fname in train_file.readlines():
        train.append(fname.strip())
    train_file.close()

    val_file = open(split_val_file, 'r')
    for fname in val_file.readlines():
        val.append(fname.strip())
    val_file.close()

    test_file = open(split_test_file, 'r')
    for fname in test_file.readlines():
        test.append(fname.strip())
    test_file.close()

    ii = 0
    all_names = train + val + test
    for audio_fname in tqdm(all_names):
        # if ii > 1:
        #     break
        # ii += 1
        video_file = os.path.join(video_dir, audio_fname.split('_')[4] + '.wav')
        # print(f'Process -> {video_file}')
        # print(audio_fname)
        seq_name, _ = AISTDataset.get_seq_name(audio_fname.replace('cAll', 'c02'))
        
        if (seq_name not in train) and (seq_name not in val) and (seq_name not in test):
            print(f'Not in set!')
            continue

        if seq_name in fnames:
            print(f'Already scaned!')
            continue

        sr = args.sampling_rate
        
        loader = None
        try:
            loader = essentia.standard.MonoLoader(filename=video_file, sampleRate=sr)
        except RuntimeError:
            continue

        fnames.append(seq_name)
        # print(seq_name)
        
        ### load audio features ###
        audio = loader()
        audio = np.array(audio).T

        feature = extract_acoustic_feature(audio, sr)
        num_musics += 1
        # musics.append(feature.tolist())

        ### load pose sequence ###
        # for seq_name in tqdm(seq_names):
        # print(f'Process -> {seq_name}')
        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, seq_name)

        smpl = None
        smpl = SMPL(model_path=args.smpl_dir, gender='MALE', batch_size=1)
        keypoints = smpl.forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
            transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
            ).joints.detach().numpy()[:, 0:24, :]
        nframes = keypoints.shape[0]

        root_rot_mat = np.array([
            axis_angle_to_matrix(torch.from_numpy(smpl_pose[0:3])).numpy() for smpl_pose in smpl_poses
        ]) #(nframes, 3, 3)
        root_angular_vel_y = np.concatenate([
            y_axis_angular_vel_from_marix(torch.from_numpy(root_rot_mat)).detach().cpu().numpy(),
            np.zeros(1)
        ], axis=0)[..., np.newaxis] #(nframes, 1)

        root_pos = keypoints[:, 0, :].reshape(nframes, -1) #(nframes, 3)

        root_rot_6d = np.array([
            matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(smpl_pose[:3]).unsqueeze(0))).flatten().numpy() for smpl_pose in smpl_poses
        ]) #(nframes, 6)
        root_vel_x = np.concatenate([root_pos[1:, 0] - root_pos[:-1, 0], np.zeros(1)], axis=0)[..., np.newaxis] #(nframes, 1)
        root_vel_z = np.concatenate([root_pos[1:, 2] - root_pos[:-1, 2], np.zeros(1)], axis=0)[..., np.newaxis] #(nframes, 1)
        root_pos_y = root_pos[:, 1][..., np.newaxis] #(nframes, 1)

        keypoints_pos = keypoints[:, 1:24, :].reshape(nframes, -1) #(nframes, 23*3)
        keypoints_vel = np.concatenate([keypoints_pos[1:] - keypoints_pos[:-1], np.zeros((1, 23*3))], axis=0) #(nframes, 23*3)
        keypoints_rot_6d = np.array([
            matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(smpl_pose[3:].reshape((23, 3))))).flatten().numpy() for smpl_pose in smpl_poses
        ]) #(nframes, 23*6)

        foot_keypoints = keypoints[:, [7, 8, 10, 11], :] #(nframes, 4, 3)
        foot_vels = np.concatenate([np.linalg.norm(foot_keypoints[1:] - foot_keypoints[:-1], axis=2), np.zeros((1, 4))], axis=0) #(nframes, 4)
        threshold = 5e-3
        foot_contacts = np.where(np.abs(foot_vels)<threshold, 1, 0)
        # print(np.count_nonzero(foot_contacts))

        pose_vec = np.concatenate([
            root_angular_vel_y, # 1
            root_vel_x, # 1
            root_vel_z, # 1
            root_pos, # 3
            root_rot_6d, # 6
            keypoints_pos, # 69
            keypoints_vel, # 69
            keypoints_rot_6d, # 138
            # foot_contacts, # 4
        ], axis=1) # 288
        # print(f'dance feature -> {pose_vec.shape}')

        # pose_vec = np.concatenate([
        #     root_rot_6d,
        #     keypoints_rot_6d
        # ], axis=1)
        num_dances += 1

        smpl_config = [smpl_scaling[0].item(), smpl_trans.tolist()]
        num_smpl_configs += 1

        data_class = "train" if audio_fname in train else "val" if audio_fname in val else "test"

        new_music, new_dance, music_raw = align_data(feature.tolist(), pose_vec.tolist())
        save_data(new_music, new_dance, music_raw, smpl_config, seq_name, data_class)
        
        # dances.append(pose_vec.tolist())
        # smpl_configs.append([smpl_scaling, smpl_trans])
        # print(np.shape(dances[-1]))

    # return None, None, None
    # return musics, dances, smpl_configs, fnames
    assert num_musics == num_dances
    assert num_smpl_configs == num_dances
    assert len(fnames) == num_dances
    return


def extract_acoustic_feature(audio, sr):

    melspe_db = extractor.get_melspectrogram(audio, sr)
    nframes = melspe_db.shape[1]
    
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = extractor.get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    harmonic_melspe_db = extractor.get_harmonic_melspe_db(audio_harmonic, sr)
    # percussive_melspe_db = extractor.get_percussive_melspe_db(audio_percussive, sr)
    # chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr, octave=7 if sr==15360*2 else 5)
    # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    tempogram = extractor.get_tempogram(onset_env, sr)
    beats_one_hot, peaks_one_hot = extractor.get_onset_beat(onset_env, sr)
    # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    onset_env = onset_env.reshape(1, -1)

    rms_energy = extractor.get_rms_energy(audio)
    source_poss = np.tile(np.array([0, 1, -1]), (nframes, 1)).reshape(3, -1)

    feature = np.concatenate([
        mfcc, # 20
        mfcc_delta, # 20
        onset_env, # 1
        tempogram,
        beats_one_hot, # 1
        # peaks_one_hot, # 1
        rms_energy, # 1
        # source_poss, # 3
    ], axis=0) # 427

    feature = feature.transpose(1, 0)
    # print(f'acoustic feature -> {feature.shape}')

    return feature

def align_data(music, dance):
    # print('---------- Align the frames of music and dance ----------')
    min_seq_len = min(len(music), len(dance))
    new_music = [music[i] for i in range(min_seq_len)]
    new_dance = [dance[i] for i in range(min_seq_len)]

    return new_music, new_dance, music


def save_data(music, dance, music_raw, smpl_config, fname, data_class):
    # print(fname)
    if data_class == "train":
        # print('---------- train data ----------')
        with open(os.path.join(args.train_dir, f'{fname}.json'), 'w') as f:
            simple_dict_train = {
                'id': fname,
                'music_array': music,
                'dance_array': dance,
                'smpl_config': smpl_config
            }
            json.dump(simple_dict_train, f)
    
    elif data_class == "val":
        # print('---------- val data ----------')
        with open(os.path.join(args.val_dir, f'{fname}.json'), 'w') as f:
            simple_dict_val = {
                'id': fname,
                'music_array': music_raw,
                'dance_array': dance,
                'smpl_config': smpl_config
            }
            json.dump(simple_dict_val, f)

    else:
        # print('---------- test data ----------')
        with open(os.path.join(args.test_dir, f'{fname}.json'), 'w') as f:
            simple_dict_test = {
                'id': fname,
                'music_array': music_raw,
                'dance_array': dance,
                'smpl_config': smpl_config
            }
            json.dump(simple_dict_test, f)

def align(musics, dances, smpl_configs):
    print('---------- Align the frames of music and dance ----------')
    assert len(musics) == len(dances), \
        'the number of audios should be equal to that of videos'
    assert len(musics) == len(smpl_configs)
    new_musics=[]
    new_dances=[]
    for i in range(len(musics)):
        min_seq_len = min(len(musics[i]), len(dances[i]))
        print(f'music -> {np.array(musics[i]).shape}, ' +
              f'dance -> {np.array(dances[i]).shape}, ' +
              f'min_seq_len -> {min_seq_len}')

        new_musics.append([musics[i][j] for j in range(min_seq_len)])
        new_dances.append([dances[i][j] for j in range(min_seq_len)])

    return new_musics, new_dances, musics

def split_data(fnames):
    train = []
    val = []
    test = []

    print('---------- Split data into train, val and test ----------')
    
    train_file = open(split_train_file, 'r')
    for fname in train_file.readlines():
        train.append(fnames.index(fname.strip()))
    train_file.close()

    val_file = open(split_val_file, 'r')
    for fname in val_file.readlines():
        val.append(fnames.index(fname.strip()))
    val_file.close()

    test_file = open(split_test_file, 'r')
    for fname in test_file.readlines():
        test.append(fnames.index(fname.strip()))
    test_file.close()

    train = np.array(train)
    val = np.array(val)
    test = np.array(test)

    return train, val, test


def save(args, musics, dances, smpl_configs, fnames, musics_raw):
    print('---------- Save to text file ----------')
    # fnames = sorted(os.listdir(os.path.join(args.input_dance_dir,inner_dir)))
    # # fnames = fnames[:20]  # for debug
    # assert len(fnames)*2 == len(musics) == len(dances), 'alignment'

    train_idx, val_idx, test_idx = split_data(fnames)
    print(f'train ids: {[fnames[idx] for idx in train_idx]}')
    print(f'val ids: {[fnames[idx] for idx in val_idx]}')
    print(f'test ids: {[fnames[idx] for idx in test_idx]}')

    print('---------- train data ----------')
    for idx in train_idx:
        with open(os.path.join(args.train_dir, f'{fnames[idx]}.json'), 'w') as f:
            sample_dict = {
                'id': fnames[idx],
                'music_array': musics[idx],
                'dance_array': dances[idx],
                'smpl_config': smpl_configs[idx]
            }
            json.dump(sample_dict, f)
    
    print('---------- val data ----------')
    for idx in val_idx:
        with open(os.path.join(args.val_dir, f'{fnames[idx]}.json'), 'w') as f:
            simple_dict = {
                'id': fnames[idx],
                'music_array': musics_raw[idx],
                'dance_array': dances[idx],
                'smpl_config': smpl_configs[idx]
            }
            json.dump(simple_dict, f)

    print('---------- test data ----------')
    for idx in test_idx:
        with open(os.path.join(args.test_dir, f'{fnames[idx]}.json'), 'w') as f:
            sample_dict = {
                'id': fnames[idx],
                'music_array': musics_raw[idx], # musics[idx+i],
                'dance_array': dances[idx],
                'smpl_config': smpl_configs[idx]
            }
            json.dump(sample_dict, f)



if __name__ == '__main__':
    make_music_dance_set(args.input_video_dir, args.input_annotation_dir) 
    # musics, dances, smpl_configs, fnames = make_music_dance_set(args.input_video_dir, args.input_annotation_dir)

    # musics, dances, musics_raw = align(musics, dances, smpl_configs)
    # save(args, musics, dances, smpl_configs, fnames, musics_raw)