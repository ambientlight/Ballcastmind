from typing import Optional
from math import cos, sin


class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __init__(self,
                 x: Optional[float] = None,
                 y: Optional[float] = None,
                 z: Optional[float] = None,
                 w: Optional[float] = None):

        self.x = x if x is not None else 0
        self.y = y if y is not None else 0
        self.z = z if z is not None else 0
        self.w = w if w is not None else 1

    # assuming default XYZ order
    def set_from_euler(self, x: float, y: float, z: float):
        c1 = cos(x / 2)
        c2 = cos(y / 2)
        c3 = cos(z / 2)

        s1 = sin(x / 2)
        s2 = sin(y / 2)
        s3 = sin(z / 2)

        self.x = s1 * c2 * c3 + c1 * s2 * s3
        self.y = c1 * s2 * c3 - s1 * c2 * s3
        self.z = c1 * c2 * s3 + s1 * s2 * c3
        self.w = c1 * c2 * c3 - s1 * s2 * s3
