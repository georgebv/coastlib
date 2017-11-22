import numpy as np


class Vector():

    def __init__(self, **kwargs):

        self.u = kwargs.pop('u', None)
        self.v = kwargs.pop('v', None)

        if (self.u and not self.v) or (self.v and not self.u):
            raise IOError('both vector components should be given')

        if not self.u and not self.v :
            self.mag = kwargs.pop('mag', None)
            self.dir = kwargs.pop('dir', None)

        if (self.mag and not self.dir) or (self.dir and not self.mag):
            raise IOError('both vector components should be given')

        if (not self.u and not self.v) or (not self.mag and not self.v):
            raise IOError('not enough arguments passed')

        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        if self.mag:
            self.magdir2uv()
        else:
            self.uv2magdir()

        self.unit()

    def uv2magdir(self):
        self.mag = np.sqrt(self.u ** 2 + self.v ** 2)
        self.dir = np.rad2deg((np.arctan2(self.u, self.v)))
        if self.dir < 0:
            self.dir += 360
    
    def magdir2uv(self):
        if 0 <= self.dir < 90:
            self.v = self.mag * np.cos(np.deg2rad(self.dir))
            self.u = self.mag * np.sin(np.deg2rad(self.dir))
        elif 90 <= self.dir < 180:
            self.u = self.mag * np.cos(np.deg2rad(self.dir - 90))
            self.v = -self.mag * np.sin(np.deg2rad(self.dir - 90))
        elif 180 <= self.dir < 270:
            self.v = -self.mag * np.cos(np.deg2rad(self.dir - 180))
            self.u = -self.mag * np.sin(np.deg2rad(self.dir - 180))
        elif 270 <= self.dir < 360:
            self.u = -self.mag * np.cos(np.deg2rad(self.dir - 270))
            self.v = self.mag * np.sin(np.deg2rad(self.dir - 270))

    def unit(self):
        self.i = self.u / self.mag
        self.j = self.v / self.mag

def dotproduct(v1, v2):
    return np.sum(v1.u * v2.u + v1.v * v2.v)

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
        return np.rad2deg(
            np.arccos(
                dotproduct(v1, v2) / (v1.mag * v2.mag)
            )
        )
