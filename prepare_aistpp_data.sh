#!/bin/sh
python _prepro_aistpp.py

# for actor critic
# python _prepro_aistpp_music.py --store_dir data/aistpp_music_feat --sampling_rate 30720

# for demo in the wild
# python _prepro_aistpp_music.py --store_dir data/demo_music_feat_60fps --sampling_rate 30720
# python _prepro_aistpp_music.py --input_video_dir data/demo_music --store_dir data/demo_music_feat_7.5fps

# remove bad dances; the list is from AIST++ project page
for ff in `cat aist_plusplus_final/ignore_list.txt`
do
    rm -rf data/aistpp_train_wav/$ff
done
