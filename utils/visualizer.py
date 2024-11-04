### Bailando, MDM

import os
from tqdm import tqdm
from functools import partial
import json
from multiprocessing import Pool
from PIL import Image
import cv2
import numpy as np
from utils.keypoint2img import read_keypoints
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

pose_keypoints_num = 25
height = 540
width = 960

def export(motions, rot_motions, names, save_path, prefix=None):
    assert len(names) == len(motions)
    assert len(rot_motions) == len(motions)
    # motions: [nmotions, nframes, 24, 3]
    np_motions = []
    motion_datas = []

    for idx in range(len(motions)):
        np_motion = motions[idx]
        motion_datas.append(np_motion)

        global_shift = np.expand_dims(np_motion[0, 0, :].copy(), axis=(0, 1))
        np_motion -= np.tile(global_shift, (np_motion.shape[0], 24, 1))

        n, njoints, nfeats = np_motion.shape
        # np_motion2 = np_motion[:, :, :2] / 2 - 0.5
        # np_motion2[:, :, 1] = np_motion2[:, :, 1]
        np_motion2 = np_motion[:, :, :2] / 1.5
        np_motion2[:, :, 0] /= 2.2
        np_motion_trans = np.zeros([n, 25, 2]).copy()
        
        # head
        np_motion_trans[:, 0] = np_motion2[:, 12]
        
        #neck
        np_motion_trans[:, 1] = np_motion2[:, 9]
        
        # left up
        np_motion_trans[:, 2] = np_motion2[:, 16]
        np_motion_trans[:, 3] = np_motion2[:, 18]
        np_motion_trans[:, 4] = np_motion2[:, 20]

        # right up
        np_motion_trans[:, 5] = np_motion2[:, 17]
        np_motion_trans[:, 6] = np_motion2[:, 19]
        np_motion_trans[:, 7] = np_motion2[:, 21]

        
        np_motion_trans[:, 8] = np_motion2[:, 0]
        
        np_motion_trans[:, 9] = np_motion2[:, 1]
        np_motion_trans[:, 10] = np_motion2[:, 4]
        np_motion_trans[:, 11] = np_motion2[:, 7]

        np_motion_trans[:, 12] = np_motion2[:, 2]
        np_motion_trans[:, 13] = np_motion2[:, 5]
        np_motion_trans[:, 14] = np_motion2[:, 8]

        np_motion_trans[:, 15] = np_motion2[:, 15]
        np_motion_trans[:, 16] = np_motion2[:, 15]
        np_motion_trans[:, 17] = np_motion2[:, 15]
        np_motion_trans[:, 18] = np_motion2[:, 15]

        np_motion_trans[:, 19] = np_motion2[:, 11]
        np_motion_trans[:, 20] = np_motion2[:, 11]
        np_motion_trans[:, 21] = np_motion2[:, 8]

        np_motion_trans[:, 22] = np_motion2[:, 10]
        np_motion_trans[:, 23] = np_motion2[:, 10]
        np_motion_trans[:, 24] = np_motion2[:, 7]

        np_motions.append(np_motion_trans.reshape([n, 25*2]))

    write2npy(motion_datas, rot_motions, names, save_path)
    # write2json(np_motions, names, save_path)
    # visualize(names, save_path)
    # img2video(save_path, prefix)
    visualize3d(save_path, prefix, names, motion_datas)

def write2npy(motions, rot_motions, motion_names, expdir):
    # print(len(dances))
    # print(len(dance_names))
    assert len(motions) == len(motion_names),\
        "number of generated dance != number of dance_names"

    ep_path = os.path.join(expdir, "npy")
        
    if not os.path.exists(ep_path):
        os.makedirs(ep_path)

    # print("Writing Json...")
    for i in tqdm(range(len(motions)),desc='Generating npy'):
        np_motion = motions[i]
        np_rot_motion = rot_motions[i]
        npy_data = {"position": np_motion, "rotation": np_rot_motion}

        motion_path = os.path.join(ep_path, motion_names[i])
        np.save(motion_path, npy_data)

def write2json(dances, dance_names, expdir):
    assert len(dances) == len(dance_names),\
        "number of generated dance != number of dance_names"

    ep_path = os.path.join(expdir, "jsons")
        
    if not os.path.exists(ep_path):
        os.makedirs(ep_path)

    # print("Writing Json...")
    for i in tqdm(range(len(dances)),desc='Generating Jsons'):
        num_poses = dances[i].shape[0]
        dances[i] = dances[i].reshape(num_poses, pose_keypoints_num, 2)
        dance_path = os.path.join(ep_path, dance_names[i])
        if not os.path.exists(dance_path):
            os.makedirs(dance_path)

        for j in range(num_poses):
            frame_dict = {'version': 1.2}
            # 2-D key points
            pose_keypoints_2d = []

            keypoints = dances[i][j]
            for keypoint in keypoints:
                x = (keypoint[0] + 1) * 0.5 * width
                y = (keypoint[1] + 1) * 0.5 * height
                score = 0.8
                pose_keypoints_2d.extend([x, y, score])

            people_dicts = []
            people_dict = {'pose_keypoints_2d': pose_keypoints_2d}
            people_dicts.append(people_dict)
            frame_dict['people'] = people_dicts
            frame_json = json.dumps(frame_dict)
            with open(os.path.join(dance_path, f'frame{j:06d}_kps.json'), 'w') as f:
                f.write(frame_json)

def img2video(expdir, prefix, audio_path=None):
    video_dir = os.path.join(expdir, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    image_dir = os.path.join(expdir, "imgs")

    dance_names = sorted(os.listdir(image_dir))
    audio_dir = "aist_plusplus_final/all_musics"
    music_names = sorted(os.listdir(audio_dir))
    
    for dance in tqdm(dance_names, desc='Generating Videos'):
        #pdb.set_trace()
        name = dance.split(".")[0]
        # cmd = f"ffmpeg -r 60 -i {image_dir}/{dance}/frame%06d.png -vb 20M -vcodec mpeg4 -y {video_dir}/{name}.mp4 -loglevel quiet"
        # cmd = f"ffmpeg -r 60 -i {image_dir}/{dance}/%05d.png -vb 20M -vcodec qtrle -y {video_dir}/{name}.mov -loglevel quiet"

        if 'cAll' in name:
            music_name = name[-9:-5] + '.wav'
        else:
            music_name = name + '.mp3'
            audio_dir = 'extra/'
        
        cmd = f"ffmpeg -r 60 -i {image_dir}/{dance}/frame%06d.png -vb 20M -vcodec mpeg4 -y {video_dir}/{name}.{prefix}.mp4 -loglevel quiet"
        os.system(cmd)
        
        if music_name in music_names:
            # print('combining audio!')
            audio_dir_ = os.path.join(audio_dir, music_name)
            # print(audio_dir_)
            name_w_audio = name + "_audio"
            cmd_audio = f"ffmpeg -i {video_dir}/{name}.{prefix}.mp4 -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {video_dir}/{name_w_audio}.{prefix}.mp4 -loglevel quiet"
            os.system(cmd_audio)
            cmd_rm = f"rm {video_dir}/{name}.{prefix}.mp4"
            os.system(cmd_rm)

def visualize_json(fname_iter, image_dir, dance_path, dance_name, quant=None):
    j, fname = fname_iter
    json_file = os.path.join(dance_path, fname)
    img = Image.fromarray(read_keypoints(json_file, (width, height), remove_face_labels=False, basic_point_only=False))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = np.asarray(img)
    if quant is not None:
        cv2.putText(img, str(quant[j]), (width-400, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # img[np.all(img == [0, 0, 0, 255], axis=2)] = [255, 255, 255, 0]
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(f'{image_dir}/{dance_name}', f'frame{j:06d}.png'))

def visualize(names, expdir, quants=None, worker_num=4):
    json_dir = os.path.join(expdir, "jsons")

    image_dir = os.path.join(expdir, "imgs")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    dance_names = sorted(os.listdir(json_dir))
    quant_list = None

    # print("Visualizing")
    for i, dance_name in enumerate(tqdm(dance_names, desc='Generating Images')):
        dance_path = os.path.join(json_dir, dance_name)
        fnames = sorted(os.listdir(dance_path))
        if not os.path.exists(f'{image_dir}/{dance_name}'):
            os.makedirs(f'{image_dir}/{dance_name}')
        if quants is not None:
            if isinstance(quants[dance_name], tuple):
                quant_lists = []
                for qs in quants[dance_name]:   
                    downsample_rate = max(len(fnames) // len(qs), 1)
                    quant_lists.append(qs.repeat(downsample_rate).tolist())
                quant_list = [tuple(qlist[ii] for qlist in quant_lists) for ii in range(len(quant_lists[0]))]
            # while len(quant_list) < len(dance_names):
            # print(quants)
            # print(len(fnames), len(quants[dance_name]))
            else:                
                downsample_rate = max(len(fnames) // len(quants[dance_name]), 1)
                quant_list = quants[dance_name].repeat(downsample_rate).tolist()
            while len(quant_list) < len(dance_names):
                quant_list.append(quant_list[-1])        


        # Visualize json in parallel
        pool = Pool(worker_num)
        partial_func = partial(visualize_json, image_dir=image_dir, dance_path=dance_path, dance_name=dance_name, quant=quant_list)
        pool.map(partial_func, enumerate(fnames))
        pool.close()
        pool.join()
    
def visualize3d(save, prefix, names, motions, figsize=(9.6, 6.4), fps=60, radius=3):

    def init():
        fig.suptitle(title, fontsize=8)
    
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    
    def update(index):
        ax.clear()
        ax.set_xlim3d(-radius / 2, radius / 2)
        ax.set_ylim3d(0, radius)
        ax.set_zlim3d(-radius / 3., radius * 2 / 3.)
        ax.grid(b=False)
        ax.view_init(elev=120, azim=-90, roll=0)
        ax.dist = 7.5

        plot_xzPlane(2 * MINS[0], 2 * MAXS[0], 0, 2 * MINS[2], 2 * MAXS[2])
        used_colors = colors

        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    video3d_dir = os.path.join(save, 'videos3d')
    if not os.path.exists(video3d_dir):
        os.makedirs(video3d_dir)
    kinematic_tree = [
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20, 22],
        [9, 14, 17, 19, 21, 23]
    ]
    matplotlib.use('Agg')

    for i, motion in enumerate(tqdm(motions, desc='Generating 3D animation')):
        name = names[i].split('.')[0]
        title = title = '\n'.join(wrap(name, 40))
        data = motion.copy().reshape(len(motion), -1, 3) # [nframes, 24, 3]
        data *= 1.3
        assert (data.shape == motion.shape)

        fig = plt.figure(figsize=figsize)
        # plt.tight_layout()
        ax = fig.add_subplot(projection='3d')
        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)
        init()
        colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
        colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
        colors = colors_orange
        if(prefix == 'gt'):
            colors = colors_blue
        
        frame_num = data.shape[0]

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]

        ani = FuncAnimation(fig, update, frames=frame_num, interval=240, repeat=False)
        video3d_file_name = name + '.' + prefix + '.mp4'
        video3d_file_path = os.path.join(video3d_dir, video3d_file_name)
        ani.save(filename=video3d_file_path, writer="ffmpeg", fps=fps)

        music_name = name[-9:-5] + '.wav'
        audio_dir = "aist_plusplus_final/all_musics"
        music_names = sorted(os.listdir(audio_dir))
        
        if music_name in music_names:
            # print('combining audio!')
            audio_dir_ = os.path.join(audio_dir, music_name)
            # print(audio_dir_)
            name_w_audio = name + "_audio"
            cmd_audio = f"ffmpeg -i {video3d_file_path} -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {video3d_dir}/{name_w_audio}.{prefix}.mp4 -loglevel quiet"
            os.system(cmd_audio)
            cmd_rm = f"rm {video3d_dir}/{name}.{prefix}.mp4"
            os.system(cmd_rm)

        plt.close()