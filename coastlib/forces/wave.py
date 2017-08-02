import numpy as np
import scipy.constants
from coastlib.wavemodels.fenton import FentonWave


def drag_force(velocity, drag_coefficient, section_area, rho=1025):
    # https://en.wikipedia.org/wiki/Drag_equation
    return (1 / 2) * rho * (velocity ** 2) * drag_coefficient * section_area


def drag_coefficient():
    # https://en.wikipedia.org/wiki/Drag_coefficient
    # TODO - function of Reynolds number
    pass


def inertia_force(acceleration, volume, added_mass_coefficient, rho=1025):
    # https://en.wikipedia.org/wiki/Froude%E2%80%93Krylov_force
    # hydrodynamic mass force
    return rho * (1 + added_mass_coefficient) * volume * acceleration


def added_mass_coefficient():
    # https://en.wikipedia.org/wiki/Added_mass
    # TODO - added mass coefficient
    pass


class Morrison:
    # https://en.wikipedia.org/wiki/Morison_equation

    def __init__(self, wave_height, wave_period, depth, **kwargs):

        # Parse and verify inputs
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

        if self.type == 'vertical cylinder':
            self.cylinder_diameter = kwargs.pop('cylinder_diameter', None)
            self.cylinder_top = kwargs.pop('cylinder_top', None)
            self.cylinder_bottom = kwargs.pop('cylinder bottom', None)

            # TODO - finish the vertical cylinder type
            self.drag_coefficient = kwargs.pop('drag_coefficient', )
        else:
            raise ValueError('Type {} is not supported'.format(self.type))

        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        # Calculate force and moment against seabed over wave phase and save internally
        # TODO
