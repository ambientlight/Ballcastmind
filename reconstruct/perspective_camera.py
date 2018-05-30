from typing import Optional
from math import tan, radians
from numpy import ndarray
import numpy as np

from reconstruct.matrix4 import Matrix4
from reconstruct.quaternion import Quaternion


class PerspectiveCamera:
    _position: ndarray
    _rotation: ndarray
    _scale: ndarray

    fov: float
    zoom: float
    near: float
    far: float
    aspect: float

    focus: int
    film_gauge: int
    film_offset: int

    matrix: Matrix4
    projection_matrix: Matrix4

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: ndarray):
        self._position = value
        self.update_matrix()

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value: ndarray):
        self._rotation = value
        self.update_matrix()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value: ndarray):
        self._scale = value
        self.update_matrix()

    def __init__(self,
                 fov: Optional[float],
                 aspect: Optional[float],
                 near: Optional[float],
                 far: Optional[float]):

        self._position = np.array([0, 0, 0], dtype=float)
        self._rotation = np.array([0, 0, 0], dtype=float)
        self._scale = np.array([1, 1, 1], dtype=float)

        self.fov = fov if fov is not None else 50.0
        self.zoom = 1

        self.near = near if near is not None else 0.1
        self.far = far if far is not None else 2000
        self.focus = 10

        self.aspect = aspect if aspect is not None else 1
        self.film_gauge = 35
        self.film_offset = 0

        self.matrix = Matrix4()
        self.projection_matrix = Matrix4()

        self.update_projection_matrix()

    def copy(self):
        target = PerspectiveCamera(self.fov, self.aspect, self.near, self.far)
        target._position = self._rotation.copy()
        target._rotation = self._rotation.copy()
        target._scale = self._scale.copy()

        target.matrix.value = self.matrix.value.copy()
        target.projection_matrix.value = self.projection_matrix.value.copy()
        return target

    def with_new_fov(self, fov: float):
        target = self.copy()
        target.fov = fov
        target.update_projection_matrix()

    def get_film_width(self):
        return self.film_gauge * min(self.aspect, 1)

    def update_projection_matrix(self):
        near = self.near
        top = near * tan(radians(0.5 * self.fov) / self.zoom)
        height = 2 * top
        width = self.aspect * height
        left = -0.5 * width
        skew = self.film_offset
        if skew != 0:
            left += near * skew / self.get_film_width()

        self.projection_matrix.make_perspective(
            left,
            left + width,
            top,
            top - height,
            near,
            self.far
        )

    def update_matrix(self):
        quaternion = Quaternion()
        quaternion.set_from_euler(x=self.rotation[0], y=self.rotation[1], z=self.rotation[2])
        self.matrix.compose(
            self.position,
            quaternion,
            self.scale)









