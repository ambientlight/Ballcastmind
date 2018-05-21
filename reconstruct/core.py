import numpy as np
from numpy import ndarray
from numpy.linalg import inv
from reconstruct.perspective_camera import PerspectiveCamera
from reconstruct.matrix4 import Matrix4
from numpy import ones, vstack
from numpy.linalg import lstsq
from math import sqrt


def project(vector: ndarray, camera: PerspectiveCamera, width: int, height: int):
    matrix = Matrix4()
    matrix.value = np.dot(inv(camera.matrix.value), camera.projection_matrix.value)

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


def linear_parameters(p1: ndarray, p2: ndarray):

    x_coords = np.array([p1[0], p2[0]])
    y_coords = np.array([p1[1], p2[1]])

    a = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(a, y_coords, rcond=None)[0]
    return m, c


def cut_off_line(line: ndarray, cutoff: float):
    p1 = line[0]
    p2 = line[1]

    m, c = linear_parameters(p1, p2)
    x_dist: float = p2[0] - p1[0]

    # cut of the SEARCH_WINDOW_CORNER_CUTOFF from lines
    # to make sure corners are not included
    x1 = p1[0] + x_dist * cutoff
    x2 = p1[0] + x_dist * (1 - cutoff)
    y1 = m * x1 + c
    y2 = m * x2 + c

    # cut of points
    return np.array([
        [x1, y1],
        [x2, y2]
    ])


def get_perp_coord(line: ndarray, d: float):
    s = line[1] - line[0]

    vX = s[0]
    vY = s[1]

    if vX == 0 or vY == 0:
        return 0, 0, 0, 0

    mag = sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = line[0][0] + vX * d
    cY = line[0][1] + vY * d
    dX = line[0][0] - vX * d
    dY = line[0][1] - vY * d
    return np.array([
        [cX, cY],
        [dX, dY]
    ])


# https://math.stackexchange.com/questions/2043054/find-a-point-on-a-perpendicular-line-a-given-distance-from-another-point
def buffer(line: ndarray, d: float):
    m, c = linear_parameters(line[0], line[1])
    factor = sqrt((d * d) / (1 + 1 / (m * m)))
    x1 = line[0][0] + factor
    x2 = line[0][0] - factor

    # y = y2 - (1/m) * (x - x2)
    # y - y3 = - (1/m) * (x - x3)
