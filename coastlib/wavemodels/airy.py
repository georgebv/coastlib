import copy
import warnings

import numpy as np
import pandas as pd
import scipy.constants
import scipy.optimize


def solve_dispersion_relation(t, h, g=scipy.constants.g):
    """Solves dispersion relation for wavelength, given period and depth.

    Find wavelength by solving the dispersion relation given period 't' and depth 'h'.
    Solves the dispersion relation using the Newton-Raphson method.

    Parameters
    ----------
    g : float
        Gravity constant (m/s^2)
    t : float
        Wave period (sec) for which the dispersion relation is solved.
    h : float
        Water depth (m) for which the dispersion relation is solved.

    Returns
    -------
    l : float
        Estimated wavelength (m) for parameters entered.
    """

    def disprel(var):
        return (2 * np.pi / t) ** 2 - g * var * np.tanh(var * h)

    def disprel_prime1(var):
        return (-g) * (np.tanh(var * h) + var * h * (1 - np.tanh(var * h) ** 2))

    l0 = g * (t ** 2) / (2 * np.pi)
    k = scipy.optimize.newton(disprel, l0 / 2, fprime=disprel_prime1)
    return (2 * np.pi) / k


class AiryWave:
    """
    A linear wave with the following properties:

    Attributes:
        wave_period : float
            peak wave period (sec)
        wave_height : float
            significant spectral wave height (m)
        angle : float (optional)
            angle of the wave front to the depth contour (deg) (0 for normal to shore - default)
        depth : float or string (optional)
            depth associated with the wave (m) ('deep' for deepwater - default)
    """

    def __init__(self, wave_height, wave_period, depth='deep', angle=0, rho=1025):
        """
        Return a linear wave with:
            wave_period *wave_period* (sec),
            depth *depth* or 'deep' for deepwater (m),
            significant spectral wave height *wave_height* (m),
            wavelength *L* (m),
            phase speed *c* (m/sec),
            wave approach angle *angle* (deg),
            wave energy *E* (J/m^2),
            wave number *k* (m^-1),
            group velocity *cg* (m/sec),
            angular frequency *w* (Hz),
            wave amplitude *a* (m).
        """

        self.wave_period = wave_period
        if depth == 'deep':
            self.depth = 'deep'
            self.L = scipy.constants.g * (wave_period ** 2) / (2 * np.pi)
        elif isinstance(depth, float) or isinstance(depth, int):
            self.depth = depth
            self.L = solve_dispersion_relation(wave_period, depth)
        else:
            raise ValueError('Depth should be float/int or \'deep\'. {} was given'.format(depth))
        self.wave_height = wave_height
        self.c = self.L / wave_period
        self.angle = angle
        self.E = rho * scipy.constants.g * (wave_height ** 2) / 8
        self.k = 2 * np.pi / self.L
        if depth == 'deep':
            self.cg = 0.5 * self.c
        else:
            self.cg = 0.5 * self.c * (1 + (2 * self.k * depth) / np.sinh(2 * self.k * depth))
        self.w = 2 * np.pi / wave_period
        self.a = wave_height / 2
        self.rho = rho
        self.__test_wave__()

    def __repr__(self):
        print('AiryWave class object')
        print('=' * 29)
        _report = str(self.report()).split('\n')
        _report = ['Parameter               Value', ' '] + _report[1:]
        return '\n'.join(_report)

    def __test_wave__(self):
        if self.wave_height / self.L > 1 / 7:
            warnings.warn('WARNING: Critical steepness of 1/7 has been exceeded', UserWarning)
        if isinstance(self.depth, float) or isinstance(self.depth, int):
            if self.depth / self.wave_height <= 1.28:
                warnings.warn('WARNING: Depth limited breaking is occurring', UserWarning)

    def report(self):
        if isinstance(self.depth, float) or isinstance(self.depth, int):
            depth = round(self.depth, 2)
        else:
            depth = self.depth
        return pd.DataFrame(
            data={
                'Value': [
                    round(self.L, 2),
                    round(self.wave_height, 2),
                    round(self.wave_period, 2),
                    depth,
                    round(self.angle, 2),
                    round(self.k, 3),
                    round(self.E, 2),
                    round(self.c, 2)
                ]
            },
            index=[
                'Wave length [m]',
                'Wave height [m]',
                'Wave period [s]',
                'Water depth [m]',
                'Approach angle [deg]',
                'Wave number',
                'Wave energy [J]',
                'Phase speed [m/s]'
            ]
        )

    def dynprop(self, z, t, x=0):
        """
        For a specified vertical co-ordinate *z* (m) (positive upward, origin at still water level)
        caclulates the following dynamic properties at time *t* (sec) (t can take values in interval
        [0;*wave wave_period*]) for a fixed position *x* (m) [0;*L*], or vice versa:
            dynamic pressure *pd* (Pa),
            horizontal particle acceleration *ua* (m/s^2),
            horizontal particle velocity *u* (m/s),
            vertical particle acceleration *va* (m/s^2),
            vertical particle velocity *v* (m/s),
            wave profile (free surface elevation) *S* (m, above still water surface).
        """
        if z > 0:
            raise ValueError('ERROR: Value <z> should be negative')
        elif self.depth != 'deep':
            if z + self.depth < 0:
                raise ValueError('ERROR: Value <z> should be less or equal to negative depth')
        self.z = z
        self.x = x
        self.t = t
        self.S = self.a * np.sin(self.w * t - self.k * x)
        if self.depth == 'deep':
            self.pd = self.rho * scipy.constants.g * self.a * np.exp(self.k * z) \
                * np.sin(self.w * t - self.k * x)
            self.ua = (self.w ** 2) * self.a * np.exp(self.k * z) * np.cos(self.w * t - self.k * x)
            self.u = self.w * self.a * np.exp(self.k * z) * np.sin(self.w * t - self.k * x)
            self.va = -(self.w ** 2) * self.a * np.exp(self.k * z) * np.sin(self.w * t - self.k * x)
            self.v = self.w * self.a * np.exp(self.k * z) * np.cos(self.w * t - self.k * x)
        elif isinstance(self.depth, float) or isinstance(self.depth, int):
            self.pd = self.rho * scipy.constants.g * self.a * np.cosh(self.k * (z + self.depth)) \
                * np.sin(self.w * t - self.k * x) / np.cosh(self.k * self.depth)
            self.ua = (self.w ** 2) * self.a * np.cosh(self.k * (z + self.depth)) \
                * np.cos(self.w * t - self.k * x) / np.sinh(self.k * self.depth)
            self.u = self.w * self.a * np.cosh(self.k * (z + self.depth)) * np.sin(self.w * t - self.k * x) \
                / np.sinh(self.k * self.depth)
            self.va = -(self.w ** 2) * self.a * np.sinh(self.k * (z + self.depth)) \
                * np.sin(self.w * t - self.k * x) / np.sinh(self.k * self.depth)
            self.v = self.w * self.a * np.sinh(self.k * (z + self.depth)) * np.cos(self.w * t - self.k * x) \
                / np.sinh(self.k * self.depth)

    def propagate(self, ndepth):
        """
        Using the linear wave theory propagate wave to the new depth *ndepth*
        and update all linear wave paramters.

        Parameters
        ----------
        ndepth : float
            Depth at the location to which the linear wave is propagated (m).

        Returns
        -------
        Updated linear wave parameters at the new depth *ndepth*.
        """
        nl = solve_dispersion_relation(self.wave_period, ndepth)
        nc = nl / self.wave_period
        # Shoaling
        k = 2 * np.pi / nl
        ncg = 0.5 * nc * (1 + 2 * k * ndepth / np.sinh(2 * k * ndepth))
        ks = np.sqrt(self.cg / ncg)
        # Refraction
        ac = (nl / self.L) * np.sin(np.deg2rad(self.angle))
        a = np.rad2deg(np.arcsin(ac))
        kr = np.sqrt(np.cos(np.deg2rad(self.angle)) / np.cos(np.deg2rad(a)))
        if ndepth - self.wave_height * (ks * kr) * 1.28 < 0:
            warnings.warn('WARNING : The wave was propagated beyond breaking point', UserWarning)
        self.angle = a
        self.wave_height *= (ks * kr)
        self.c = nc
        self.depth = ndepth
        self.L = nl
        self.E = self.rho * scipy.constants.g * (self.wave_height ** 2) / 8
        self.k = 2 * np.pi / self.L
        self.cg = 0.5 * self.c * (1 + 2 * self.k * self.depth / np.sinh(2 * self.k * self.depth))
        self.w = 2 * np.pi / self.wave_period
        self.a = self.wave_height / 2
        if self.z is not None:
            if self.z + self.depth >= 0:
                self.dynprop(self.z, self.x, self.t)
            else:
                self.x = None
                self.t = None
                self.S = None
                self.z = None
                self.u = None
                self.v = None
                self.ua = None
                self.va = None
                self.pd = None
        self.__test_wave__()

    def wavebreak(self, precision=0.01, echo=True):
        """
        Propagates the wave until it breaks (*depth* = 1.28*wave_height* breaking condition).
        Updates wave parameters at the moment of breaking.
        """
        if self.depth == 'deep':
            self.depth = 0.6 * self.L
        depth = self.depth
        while True:
            b = copy.deepcopy(self)
            depth -= precision
            b.propagate(depth)
            kr = np.sqrt(np.cos(np.deg2rad(self.angle)) / np.cos(np.deg2rad(b.angle)))
            ks = np.sqrt(self.cg / b.cg)
            crt1 = b.depth - b.wave_height * 1.28
            crt2 = kr * ks - b.wave_height / self.wave_height
            if crt1 < 0 and crt2 < 0:
                if echo:
                    print('Solution reached with:'
                          '    <depth> - <wave height> * 1.28 = {0:.2f}'
                          '    <kr> * <ks> - <wave height 0> / <wave height 1> = {1:.2f}'.format(crt1, crt2))
                depth += precision
                break
        self.propagate(depth)
        self.__test_wave__()
