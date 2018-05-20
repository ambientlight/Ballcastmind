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

    def multiple_matrices(self, a: ndarray, b: ndarray):
        ae = a
        be = b
        te = self.value

        a11 = ae[0][0]
        a12 = ae[1][0]
        a13 = ae[2][0]
        a14 = ae[3][0]

        a21 = ae[0][1]
        a22 = ae[1][1]
        a23 = ae[2][1]
        a24 = ae[3][1]

        a31 = ae[0][2]
        a32 = ae[1][2]
        a33 = ae[2][2]
        a34 = ae[3][2]

        a41 = ae[0][3]
        a42 = ae[1][3]
        a43 = ae[2][3]
        a44 = ae[3][3]

        b11 = be[0][0]
        b12 = be[1][0]
        b13 = be[2][0]
        b14 = be[3][0]

        b21 = be[0][1]
        b22 = be[1][1]
        b23 = be[2][1]
        b24 = be[3][1]

        b31 = be[0][2]
        b32 = be[1][2]
        b33 = be[2][2]
        b34 = be[3][2]

        b41 = be[0][3]
        b42 = be[1][3]
        b43 = be[2][3]
        b44 = be[3][3]

        te[0][0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41
        te[1][0] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42
        te[2][0] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43
        te[3][0] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44

        te[0][1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41
        te[1][1] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42
        te[2][1] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43
        te[3][1] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44

        te[0][2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41
        te[1][2] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42
        te[2][2] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43
        te[3][2] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44

        te[0][3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41
        te[1][3] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42
        te[2][3] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43
        te[3][3] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44

        self.value = te
