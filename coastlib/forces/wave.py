import numpy as np
import scipy.constants
from coastlib.wavemodels.fenton import FentonWave
import pandas as pd


def drag_force(velocity, drag_coefficient, face_area, rho=1025):
    # https://en.wikipedia.org/wiki/Drag_equation
    # https://en.wikipedia.org/wiki/Drag_coefficient
    # <face_area> - unit area normal to flow
    return (1 / 2) * rho * velocity * np.abs(velocity) * drag_coefficient * face_area


def inertia_force(acceleration, volume, inertia_coefficient, rho=1025):
    # https://en.wikipedia.org/wiki/Froude%E2%80%93Krylov_force
    # hydrodynamic mass force
    return rho * inertia_coefficient * volume * acceleration


class Morrison:
    # https://en.wikipedia.org/wiki/Morison_equation

    def __init__(self, wave_height, wave_period, depth, **kwargs):

        # Parse general inputs
        self.wave_height = wave_height  # up to user to provide the 1.8Hs value
        self.wave_period = wave_period
        self.depth = depth
        self.current_velocity = kwargs.pop('current_velocity', 0)
        self._rho = kwargs.pop('rho', 1025)
        self._g = kwargs.pop('g', scipy.constants.g)
        self.fenton_wave = FentonWave(
            wave_heigh=wave_height, wave_period=wave_period, depth=depth,
            current_criterion=1, current_velocity=self.current_velocity,
            rho=self._rho, g=self._g, points=dict(m=100, ua=100, vert=1000)
        )  # 1000 points per vertical profile for extra accuracy
        self.type = kwargs.pop('element_type', 'vertical cylinder')

        # Parse phases
        self._x = np.unique(self.fenton_wave.flowfield['X (m)'].values)

        if self.type == 'vertical cylinder':
            # Parse type-specific inputs
            self.cylinder_diameter = kwargs.pop('cylinder_diameter', None)
            self.cylinder_top = kwargs.pop('cylinder_top', None)  # Relative to seabed
            self.cylinder_bottom = kwargs.pop('cylinder bottom', None)  # Relative to seabed
            self.drag_coefficient = kwargs.pop('drag_coefficient', 1.2)
            self.inertia_coefficient = kwargs.pop('inertia_coefficient', 2)
            # Calculate wave force on the cylinder and store results
            self.__vertical_cylinder()
        elif self.type == 'sloped cylinder':
            # Parse type-specific inputs
            self.cylinder_diameter = kwargs.pop('cylinder_diameter', None)
            self.cylinder_top = kwargs.pop('cylinder_top', None)  # Relative to seabed
            self.cylinder_bottom = kwargs.pop('cylinder bottom', None)  # Relative to seabed
            self.drag_coefficient = kwargs.pop('drag_coefficient', 1.2)
            self.inertia_coefficient = kwargs.pop('inertia_coefficient', 2)
            self.slope = kwargs.pop('slope', 3)  # Slope V:H
            # TODO - inclined cylinder
            pass
        else:
            raise ValueError('Type {} is not supported'.format(self.type))

        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    def __vertical_cylinder(self):
        forces, moments, centroids = [], [], []
        for i, x in enumerate(self._x):
            flowfield = self.fenton_wave.flowfield[
                (self.fenton_wave.flowfield['X (m)'] == x) &
                (self.fenton_wave.flowfield['Y (m)'] >= self.cylinder_bottom) &
                (self.fenton_wave.flowfield['Y (m)'] <= self.cylinder_top)
            ]
            # integration slice heights
            dz = flowfield['Y (m)'].values[1:] - flowfield['Y (m)'].values[:-1]
            # average <u> for each slice
            u = (flowfield['u (m/s)'].values[1:] + flowfield['u (m/s)'].values[:-1]) / 2
            # average <ua> for each slice
            ua = (flowfield['ua (m/s)'].values[1:] + flowfield['ua (m/s)'].values[:-1]) / 2
            # center coordinates for each slice (from seabed to slice center)
            y = (flowfield['Y (m)'].values[1:] + flowfield['Y (m)'].values[:-1]) / 2

            # Wave force
            force = sum(drag_force(
                velocity=u, drag_coefficient=self.drag_coefficient,
                face_area=self.cylinder_diameter*dz, rho=self._rho
                ),
                inertia_force(
                    acceleration=ua, volume=np.pi*self.cylinder_diameter*dz,
                    inertia_coefficient=self.inertia_coefficient, rho=self._rho
                    )
            )
            forces.extend([force.sum()])
            moment = force * y
            moments.extend([moment.sum()])
            centroids.extend(([moment.sum() / force.sum()]))
        self.force = pd.DataFrame(data=self._x, columns=['X (m)'])
        self.force['F (N)'] = forces
        self.force['M (N-m)'] = moments
        self.force['Centroid (m from seabed)'] = centroids
