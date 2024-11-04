#!/bin/bash

/userhome/cs2/xsy27/blender/2.93/python/bin/python3.9 -m ensurepip --upgrade
/userhome/cs2/xsy27/blender/2.93/python/bin/python3.9 -m pip install torch==2.4.0 smplx trimesh tqdm moviepy scipy chumpy --target=/userhome/cs2/xsy27/blender/2.93/python/lib/python3.9/site-packages
/userhome/cs2/xsy27/blender/2.93/python/bin/python3.9 -m pip install -U numpy==1.23.1