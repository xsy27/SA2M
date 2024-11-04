import bpy
import math


class Camera:
    def __init__(self, first_root, mode):
        camera = bpy.data.objects['Camera']

        # initial position
        camera.location.x = 0
        camera.location.y = -10
        camera.location.z = 3

        camera.rotation_euler.x = 70 * (math.pi / 180.0)
        camera.rotation_euler.y = 0
        camera.rotation_euler.z = 0

        # wider point of view
        if mode == "sequence":
            camera.data.lens = 65
        elif mode == "frame":
            camera.data.lens = 130
        elif mode == "video":
            camera.data.lens = 110

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        self._root = first_root

    def update(self, new_root):
        delta_root = new_root - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = new_root
