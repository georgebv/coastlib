import numpy as np


class Vector():

    def __init__(self, **kwargs):

        self.u = kwargs.pop('u', None)
        self.v = kwargs.pop('v', None)

        self.mag = kwargs.pop('mag', None)
        self.dir = kwargs.pop('dir', None)

        if self.dir > 360 or self.dir < 0:
            raise IOError('direction outside the [0;360] range')

        if (not (self.mag is None) or not (self.dir is None)) and (not (self.u is None) or not (self.v is None)):
            raise IOError('mixed vector definition is not allowed')

        if (not (self.u is None) and self.v is None) or (not (self.v is None) and self.u is None):
            raise IOError('both vector components should be given')

        if (not (self.mag is None) and self.dir is None) or (not (self.dir is None) and self.mag is None):
            raise IOError('both vector components should be given')

        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        if self.u is None:
            self.__magdir2uv()
        elif self.mag is None:
            self.__uv2magdir()
        else:
            raise RuntimeError('unexpected error')

        if self.mag != 0:
            self.i = self.u / self.mag
            self.j = self.v / self.mag
        else:
            self.i = self.j = np.nan

    def __repr__(self):
        return f'Vector\n' \
               f'Magnitude    -> {self.mag}\n' \
               f'Direction to -> {self.dir}\n' \
               f'U-component  -> {self.u}\n' \
               f'V-component  -> {self.v}'

    def __add__(self, other):
        assert type(other).__name__ == 'Vector', 'type "Vector" expected, got "{}" instead'.format(type(other).__name__)
        return Vector(u=self.u + other.u, v=self.v + other.v)

    def __sub__(self, other):
        assert type(other).__name__ == 'Vector', 'type "Vector" expected, got "{}" instead'.format(type(other).__name__)
        return Vector(u=self.u - other.u, v=self.v - other.v)

    def __mul__(self, other):
        # Dot product
        assert type(other).__name__ == 'Vector', 'type "Vector" expected, got "{}" instead'.format(type(other).__name__)
        return np.sum(self.u * other.u + self.v * other.v)

    def __uv2magdir(self):
        self.mag = np.sqrt(self.u ** 2 + self.v ** 2)
        self.dir = np.rad2deg((np.arctan2(self.u, self.v)))
        if self.dir < 0:
            self.dir += 360
        if 0 <= self.dir < 90:
            self.quadrant = 'I'
        elif 90 <= self.dir < 180:
            self.quadrant = 'II'
        elif 180 <= self.dir < 270:
            self.quadrant = 'III'
        elif 270 <= self.dir <=360:
            self.quadrant = 'IV'

    def __magdir2uv(self):
        if 0 <= self.dir < 90:
            self.v = self.mag * np.cos(np.deg2rad(self.dir))
            self.u = self.mag * np.sin(np.deg2rad(self.dir))
        elif 90 <= self.dir < 180:
            self.u = self.mag * np.cos(np.deg2rad(self.dir - 90))
            self.v = -self.mag * np.sin(np.deg2rad(self.dir - 90))
        elif 180 <= self.dir < 270:
            self.v = -self.mag * np.cos(np.deg2rad(self.dir - 180))
            self.u = -self.mag * np.sin(np.deg2rad(self.dir - 180))
        elif 270 <= self.dir <= 360:
            self.u = -self.mag * np.cos(np.deg2rad(self.dir - 270))
            self.v = self.mag * np.sin(np.deg2rad(self.dir - 270))
        else:
            self.u, self.v = np.nan, np.nan


def angle(v1, v2):
    # Returns the smaller angle between two vectors [0;180]
    # For azimutal angle use "v1.dir - v2.dir"
    if v1.i == v2.i and v1.j == v2.j:
        # Collinear vectors
        return 0
    elif v1.i == -v2.i and v1.j == -v2.j:
        # Opposite direction vectors
        return 180
    else:
        return np.rad2deg(np.arccos(v1 * v2 / (v1.mag * v2.mag)))
