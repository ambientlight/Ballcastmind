import numpy as np
from numpy import ndarray
from numpy.linalg import inv
from reconstruct.perspective_camera import PerspectiveCamera
from reconstruct.matrix4 import Matrix4


def project(vector: ndarray, camera: PerspectiveCamera, width: int, height: int):
    matrix = Matrix4()
    # wtf is here: should be the following
    # np.dot(camera.projection_matrix.value, inv(camera.matrix.value))

    # camera.projection_matrix.value[0][0] = 8.624822915037795
    matrix.multiple_matrices(
        camera.projection_matrix.value,
        inv(camera.matrix.value)
    )

    target = vector.copy()
    x = target[0]
    y = target[1]
    z = target[2]
    e = matrix.value

    w = 1 / (e[0][3] * x + e[1][3] * y + e[2][3] * z + e[3][3])
    target[0] = (e[0][0] * x + e[1][0] * y + e[2][0] * z + e[3][0]) * w
    target[1] = (e[0][1] * x + e[1][1] * y + e[2][1] * z + e[3][1]) * w
    target[2] = (e[0][2] * x + e[1][2] * y + e[2][2] * z + e[3][2]) * w

    width_half = width / 2
    height_half = height / 2
    target[0] = (target[0] * width_half) + width_half
    target[1] = -(target[1] * height_half) + height_half

    return target

