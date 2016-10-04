from copy import deepcopy
from math import sinh, pi, sqrt, sin, asin, cos, cosh, exp, tanh

from coastlib.coreutils.design_tools import g
from coastlib.coreutils.design_tools import sea_water_density as swd
from scipy.optimize import newton


def solve_dispersion_relation(t, h):
    """Solves dispersion relation for wavelength, given period and depth.

    Find wavelength by solving the dispersion relation given period 't' and depth 'h'.
    Solves the dispersion relation using the secant method.

    Parameters
    ----------

    t : float
        Wave period (sec) for which the dispersion relation is solved.
    h : float
        Water depth (m) for which the dispersion relation is solved.

    Returns
    -------
    l : float
        Estimated wavelength (m) for parameters entered.
    """

    def disprel(var_l):
        k = 2 * pi / var_l
        omega = 2 * pi / t
        return omega ** 2 - g * k * tanh(k * h)

    return newton(disprel, 1)


class LinearWave:
    """
    A linear wave with the following properties:

    Attributes:
        period : float
            peak wave period (sec)
        Hm0 : float
            significant spectral wave height (m)
        angle : float (optional)
            angle of the wave front to the depth contour (deg) (0 for normal to shore - default)
        depth : float or string (optional)
            depth associated with the wave (m) ('deep' for deepwater - default)
    """

    def __init__(self, period, Hm0, angle=0, depth='deep'):
        """
        Return a linear wave with:
            period *period* (sec),
            depth *depth* or 'deep' for deepwater (m),
            significant spectral wave height *Hm0* (m),
            wavelength *L* (m),
            phase speed *c* (m/sec),
            wave approach angle *angle* (deg),
            wave energy *E* (J/m^2),
            wave number *k* (m^-1),
            group velocity *cg* (m/sec),
            angular frequency *w* (Hz),
            wave amplitude *a* (m).
        """
        self.period = float(period)
        if depth == 'deep':
            self.depth = 'deep'
            self.L = g * (self.period ** 2) / (2 * pi)
        else:
            self.depth = float(depth)
            self.L = solve_dispersion_relation(self.period, self.depth)
        self.Hm0 = float(Hm0)
        self.c = self.L / self.period
        self.angle = float(angle)
        self.E = swd * g * (self.Hm0 ** 2) / 8
        self.k = 2 * pi / self.L
        if depth == 'deep':
            self.cg = 0.5 * self.c
        else:
            self.cg = 0.5 * self.c * (1 + (2 * self.k * self.depth) / sinh(2 * self.k * self.depth))
        self.w = 2 * pi / self.period
        self.a = self.Hm0 / 2
        self.S = None
        self.z = None
        self.u = None
        self.v = None
        self.ua = None
        self.va = None
        self.pd = None
        self.x = None
        self.t = None

    def dynprop(self, z, t, x=0):
        """
        For a specified vertical co-ordinate *z* (m) (positive upward, origin at still water level)
        caclulates the following dynamic properties at time *t* (sec) (t can take values in interval
        [0;*wave period*]) for a fixed position *x* (m) [0;*L*], or vice versa:
            dynamic pressure *pd* (Pa),
            horizontal particle acceleration *ua* (m/s^2),
            horizontal particle velocity *u* (m/s),
            vertical particle acceleration *va* (m/s^2),
            vertical particle velocity *v* (m/s),
            wave profile (free surface elevation) *S* (m, above still water surface).
        """
        if z > 0:
            raise ValueError('ERROR: Value *z* should be negative')
        elif self.depth != 'deep':
            if z + self.depth < 0:
                raise ValueError('ERROR: Value *z* should be less or equal to negative depth')
        self.z = z
        self.x = x
        self.t = t
        self.S = self.a * sin(self.w * t - self.k * x)
        if self.depth == 'deep':
            self.pd = swd * g * self.a * exp(self.k * z) * sin(self.w * t - self.k * x)
            self.ua = (self.w ** 2) * self.a * exp(self.k * z) * cos(self.w * t - self.k * x)
            self.u = self.w * self.a * exp(self.k * z) * sin(self.w * t - self.k * x)
            self.va = -(self.w ** 2) * self.a * exp(self.k * z) * sin(self.w * t - self.k * x)
            self.v = self.w * self.a * exp(self.k * z) * cos(self.w * t - self.k * x)
        else:
            self.pd = swd * g * self.a * cosh(self.k * (z + self.depth)) * sin(self.w * t - self.k * x) / cosh(
                self.k * self.depth)
            self.ua = (self.w ** 2) * self.a * cosh(self.k * (z + self.depth)) * cos(self.w * t - self.k * x) / sinh(
                self.k * self.depth)
            self.u = self.w * self.a * cosh(self.k * (z + self.depth)) * sin(self.w * t - self.k * x) / sinh(
                self.k * self.depth)
            self.va = -(self.w ** 2) * self.a * sinh(self.k * (z + self.depth)) * sin(self.w * t - self.k * x) / sinh(
                self.k * self.depth)
            self.v = self.w * self.a * sinh(self.k * (z + self.depth)) * cos(self.w * t - self.k * x) / sinh(
                self.k * self.depth)

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
        nL = solve_dispersion_relation(self.period, ndepth)
        nc = nL / self.period
        # Shoaling
        k = 2 * pi / nL
        ncg = 0.5 * nc * (1 + 2 * k * ndepth / sinh(2 * k * ndepth))
        Ks = sqrt(self.cg / ncg)
        # Refraction
        Ac = (nL / self.L) * sin(self.angle * pi / 180)
        A = asin(Ac) * (180 / pi)
        Kr = sqrt(cos(self.angle * pi / 180) / cos(A * pi / 180))
        self.angle = A
        self.Hm0 *= (Ks * Kr)
        self.c = nc
        self.depth = ndepth
        self.L = nL
        self.E = swd * g * (self.Hm0 ** 2) / 8
        self.k = 2 * pi / self.L
        self.cg = 0.5 * self.c * (1 + 2 * self.k * self.depth / sinh(2 * self.k * self.depth))
        self.w = 2 * pi / self.period
        self.a = self.Hm0 / 2
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

    def wavebreak(self):
        """
        Propagates the wave until it breaks (*depth* = 1.28*Hm0* breaking condition).
        Updates wave parameters at the moment of breaking.
        """
        if self.depth == 'deep':
            self.depth = 0.6 * self.L
        depth = self.depth
        while True:
            b = deepcopy(self)
            depth -= 0.01
            b.propagate(depth)
            Kr = sqrt(cos(self.angle * pi / 180) / cos(b.angle * pi / 180))
            Ks = sqrt(self.cg / b.cg)
            crt1 = b.depth - b.Hm0 * 1.28
            crt2 = Kr / Ks - b.Hm0 / self.Hm0
            if crt1 < 0 and crt2 < 0:
                depth += 0.01
                break
        self.propagate(depth)
