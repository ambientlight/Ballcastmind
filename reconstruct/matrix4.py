import numpy as np
from numpy import ndarray
from pprint import pformat
from reconstruct.quaternion import Quaternion


class Matrix4:
    # shape (4, 4)
    value: ndarray

    def __init__(self):
        self.value = np.identity(4, dtype=np.float32)

    def __getitem__(self, index):
        return self.value[index]

    def __setitem__(self, index, value):
        self.value[index] = value

    def __repr__(self):
        return pformat(self.value)

    def compose(self, position: ndarray, quaternion: Quaternion, scale: ndarray):
        self.make_rotation_from_quaternion(quaternion)
        self.scale(scale)
        self.set_position(position)

    # taken from three.js/src/math/Matrix4.js
    def make_rotation_from_quaternion(self, quaternion: Quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        x2 = x + x
        y2 = y + y
        z2 = z + z

        xx = x * x2
        xy = x * y2
        xz = x * z2

        yy = y * y2
        yz = y * z2
        zz = z * z2

        wx = w * x2
        wy = w * y2
        wz = w * z2

        self.value[0][0] = 1 - (yy + zz)
        self.value[1][0] = xy - wz
        self.value[2][0] = xz + wy

        self.value[0][1] = xy + wz
        self.value[1][1] = 1 - (xx + zz)
        self.value[2][1] = yz - wx

        self.value[0][2] = xz - wy
        self.value[1][2] = yz + wx
        self.value[2][2] = 1 - (xx + yy)

        # last column
        self.value[0][3] = 0
        self.value[1][3] = 0
        self.value[2][3] = 0

        # bottom row
        self.value[3] = np.array([0, 0, 0, 1])

    def scale(self, vector: ndarray):
        self.value[0] *= vector[0]
        self.value[1] *= vector[1]
        self.value[2] *= vector[2]

    def set_position(self, vector: ndarray):
        self.value[3][0] = vector[0]
        self.value[3][1] = vector[1]
        self.value[3][2] = vector[2]

    def make_perspective(self,
                         left: float,
                         right: float,
                         top: float,
                         bottom: float,
                         near: float,
                         far: float):

        x = 2 * near / (right - left)
        y = 2 * near / (top - bottom)

        a = (right + left) / (right - left)
        b = (top + bottom) / (top - bottom)
        c = -(far + near) / (far - near)
        d = -2 * far * near / (far - near)

        self.value[0] = np.array([x, 0, 0, 0])
        self.value[1] = np.array([0, y, 0, 0])
        self.value[2] = np.array([a, b, c, -1])
        self.value[3] = np.array([0, 0, d, 0])
