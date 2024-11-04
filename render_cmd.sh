#!/bin/bash

# /userhome/cs2/xsy27/blender/blender --background --python render.py -- --npy ./eval/aistpp_evaluation/gt/rotnpy/gBR_sBM_cAll_d04_mBR0_ch02.npy --mode video --gt True --fps 60 > render_process.out
# python -W ignore evaluate.py
/userhome/cs2/xsy27/blender/blender --background --python render.py -- --npy ./eval/aistpp_evaluation/gt/npy/gJB_sBM_cAll_d08_mJB5_ch02.npy --mode video --gt True --fps 60 > render_process.out