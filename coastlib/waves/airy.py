import numpy as np
import scipy.constants
import scipy.optimize

import coastlib.waves.support


def solve_dispersion_relation(wave_period, depth, g=scipy.constants.g):
    """
    Solves dispersion relation for given wave period and water depth.
    Solves the dispersion relation using the Newton-Raphson method.

    Parameters
    ----------
    wave_period : float
        Wave period in seconds.
    depth : float
        Water depth in meters.
    g : float, optional
        Gravity constant (m/s^2).

    Returns
    -------
    wave_length : float
        Estimated wavelength (m) for parameters entered.
    """

    omega = 2 * np.pi / wave_period

    def disp_rel(k):
        return omega ** 2 - g * k * np.tanh(k * depth)

    def disp_rel_prime(k):
        return -g * (
                np.tanh(k * depth) + k * depth * (1 - np.tanh(k * depth) ** 2)
        )

    k_0 = 4 * np.pi ** 2 / (g * wave_period ** 2)
    solution = scipy.optimize.fsolve(
        func=disp_rel, x0=k_0, fprime=disp_rel_prime, full_output=True
    )

    return 2 * np.pi / solution[0][0]


class AiryWave:
    """
    Airy (linear) wave theory class.
    See https://en.wikipedia.org/wiki/Airy_wave_theory
    See https://folk.ntnu.no/oivarn/hercules_ntnu/LWTcourse/lwt_new_2000_Part_A.pdf

    """

    def __init__(self, wave_height, wave_period, angle=0, depth=None, **kwargs):
        """
        Initializes the AiryWave class instance object.
        See https://en.wikipedia.org/wiki/Airy_wave_theory
        See https://folk.ntnu.no/oivarn/hercules_ntnu/LWTcourse/lwt_new_2000_Part_A.pdf

        Parameters
        ----------
        wave_height : float
            Wave height in meters.
        wave_period : float, optional
            Wave period in seconds.
        depth : float
            Water depth in meters. Considered deep, if None is passed (default=None).
        angle : float, optional
            Angle of the wave front to the depth contour in degrees (default=0, perpendicular to shore).
        """

        self.wave_height = wave_height
        self.wave_period = wave_period
        self.depth = depth
        self.angle = angle

        self.g = kwargs.pop('g', scipy.constants.g)
        self.rho = kwargs.pop('rho', 1025)

        self.amplitude = None
        self.wave_length = None
        self.k = None
        self.omega = None
        self.condition = None
        self.phase_speed = None
        self.group_speed = None
        self.wave_energy = None
        self.radiation_stress = None
        self.energy_flux = None

        self.broken = False

        self.z = None
        self.x = None
        self.t = None
        self.phase = None
        self.u = None
        self.v = None
        self.ua = None
        self.va = None
        self.pressure = None

        self.__update()

    def __update(self, echo=True):
        """
        Updates wave parameters such as wave length, phase speed, and energy. Checks if wave was broken.

        Parameters
        ----------
        echo : bool, optional
            If True, prints out information about wave being broken (default=True).
        """

        self.amplitude = self.wave_height / 2

        if self.depth is None:
            self.wave_length = self.g * self.wave_period ** 2 / (2 * np.pi)
        else:
            self.wave_length = solve_dispersion_relation(wave_period=self.wave_period, depth=self.depth, g=self.g)

        self.k = 2 * np.pi / self.wave_length
        self.omega = 2 * np.pi / self.wave_period
        self.phase_speed = self.omega / self.k

        if self.depth is None or self.depth > .5 * self.wave_length:
            self.condition = 'deep'
        elif self.depth < .05 * self.wave_length:
            self.condition = 'shallow'
        else:
            self.condition = 'intermediate'

        if self.condition == 'deep':
            self.group_speed = .5 * self.phase_speed
        elif self.condition == 'shallow':
            self.group_speed = self.phase_speed
        else:
            coeff = 1 + self.k * self.depth * (1 - np.tanh(self.k * self.depth) ** 2) / np.tanh(self.k * self.depth)
            self.group_speed = .5 * self.phase_speed * coeff

        self.wave_energy = self.rho * self.g * self.amplitude ** 2 / 2
        self.radiation_stress = self.wave_energy * (2 * self.group_speed / self.phase_speed - .5)
        self.energy_flux = self.wave_energy * self.group_speed

        if echo:
            if self.wave_height / self.wave_length >= 1 / 7:
                print('WARNING : Critical steepness of 1/7 has been exceeded')
            if self.depth is not None:
                if self.depth / self.wave_height <= 1.28:
                    print('WARNING : Depth limited breaking has occurred : depth/Hs <= 1.28')

        self.z = None
        self.x = None
        self.t = None
        self.phase = None
        self.u = None
        self.v = None
        self.ua = None
        self.va = None
        self.pressure = None

    def __repr__(self):
        """
        Generates a string with a summary of the AiryWave class instance object.
        """

        summary = str(
            f'{" "*15}Airy Wave\n'
            f'{"=" * 39}\n'
        )
        summary += str(
            f'Wave height{self.wave_height:26.2f} m\n'
            f'Wave period{self.wave_period:26.2f} s\n'
        )
        if self.depth is None:
            summary += str(
                f'Water depth{"N/A":>26} m\n'
            )
        else:
            summary += str(
                f'Water depth{self.depth:26.2f} m\n'
            )
        summary += str(
            f'Wave length{self.wave_length:26.2f} m\n'
            f'Wave angle{self.angle:21.2f} degrees\n'
            f'Wave energy{self.wave_energy:22.2f} J/m^2\n'
            f'{"=" * 39}'
        )

        return summary

    def get_surface(self, t, x=0):
        """
        Calculates linear wave surface elevation for given phase.

        Parameters
        ----------
        t : float
            Phase in seconds.
        x : float, optional
            Phase in meters (default=0).

        Returns
        -------
        eta : float or array_like
            Surface elevation in meters.
        """

        if not np.isscalar(t) and not np.isscalar(x):
            raise ValueError('t and x cannot both be arrays, one must be a scalar')

        phase = self.omega * t - self.k * x
        eta = self.amplitude * np.sin(phase)
        return eta

    def get_property(self, z, t, x=0):
        """
        Calculates dynamic wave properties for given parameters.

        Parameters
        ----------
        z : float
            Vertical coordinate in meters. Starts at surface elevation (0) and goes to negative depth.
        t : float
            Phase in seconds.
        x : float, optional
            Phase in meters (default=0).
        """

        if z > 0:
            raise ValueError(f'z must be negative, {z} was passed')
        if self.depth is not None and z + self.depth < 0:
            raise ValueError(f'|z|={z:.2f} should not be larger than water depth {self.depth:.2f}')

        self.z = z
        self.x = x
        self.t = t
        self.phase = self.omega * t - self.k * x

        if self.condition == 'deep':
            part = self.amplitude * np.exp(self.k * self.z)
            self.u = self.omega * part * np.sin(self.phase)
            self.v = self.omega * part * np.cos(self.phase)
            self.ua = self.omega ** 2 * part * np.cos(self.phase)
            self.va = -self.omega ** 2 * part * np.sin(self.phase)
            self.pressure = self.rho * self.g * part * np.cos(self.phase)
        elif self.condition == 'shallow':
            if self.depth is None:
                raise RuntimeError
            self.u = self.omega * self.amplitude / (self.k * self.depth) * np.sin(self.phase)
            self.v = self.omega * self.amplitude * (z + self.depth) / self.depth * np.cos(self.phase)
            self.ua = self.omega ** 2 * self.amplitude / (self.k * self.depth) * np.cos(self.phase)
            self.va = -self.omega ** 2 * self.amplitude * (z + self.depth) / self.depth * np.sin(self.phase)
            self.pressure = self.rho * self.g * self.amplitude * np.sin(self.phase)
        else:
            if self.depth is None:
                raise RuntimeError
            kzd = self.k * (z + self.depth)
            kd = self.k * self.depth
            self.u = self.omega * self.amplitude * np.cosh(kzd) / np.sinh(kd) * np.sin(self.phase)
            self.v = self.omega * self.amplitude * np.sinh(kzd) / np.sinh(kd) * np.cos(self.phase)
            self.ua = self.omega ** 2 * self.amplitude * np.cosh(kzd) / np.sinh(kd) * np.cos(self.phase)
            self.va = -self.omega ** 2 * self.amplitude * np.sinh(kzd) / np.sinh(kd) * np.sin(self.phase)
            self.pressure = self.rho * self.g * self.amplitude * np.cosh(kzd) / np.cosh(kd) * np.sin(self.phase)

    def propagate(self, new_depth, echo=False):
        """
        Propagates the wave to new depth and updates wave parameters.
        Wave can only be propagated towards shallower water.

        Parameters
        ----------
        new_depth : float
            New depth in meters to which the wave is propagated.
        echo : bool, optional
            If True, prints out information about wave being broken (default=False).
        """

        if self.depth is not None and new_depth >= self.depth:
            raise ValueError(f'New depth {new_depth:.2f} must be shallower than existing {self.depth:.2f}')

        new_wave_length = solve_dispersion_relation(wave_period=self.wave_period, depth=new_depth, g=self.g)
        new_phase_speed = new_wave_length / self.wave_period

        # Shoaling
        k = 2 * np.pi / new_wave_length
        new_group_speed = .5 * new_phase_speed * (1 + 2 * k * new_depth / np.sinh(2 * k * new_depth))
        ks = np.sqrt(self.group_speed / new_group_speed)

        # Refraction
        ac = new_wave_length / self.wave_length * np.sin(np.deg2rad(self.angle))
        a = np.rad2deg(np.arcsin(ac))
        # preserves angle of approach sign
        if self.angle * a < 0:
            a *= -1
        kr = np.sqrt(np.cos(np.deg2rad(self.angle)) / np.cos(np.deg2rad(a)))

        self.depth = new_depth
        self.wave_length = new_wave_length
        self.phase_speed = new_phase_speed
        self.group_speed = new_group_speed
        self.angle = a
        self.wave_height *= ks * kr

        # Check if wave was broken
        check_1 = self.depth / self.wave_height <= 1.28
        check_2 = self.wave_height / self.wave_length >= 1 / 7
        if check_1 or check_2:
            self.broken = True
            if echo:
                if check_1:
                    print('Depth limited breaking has occurred : depth/Hs <= 1.28')
                elif check_2:
                    print('Critical steepness of 1/7 has been exceeded')

        self.__update(echo=False)

    def break_wave(self, precision=.01, echo=False):
        """
        Propagates wave until it breaks due to steepness or depth.

        Parameters
        ----------
        precision : float, optional
            Calculation precision (default=.01).
        echo : bool, optional
            If True, prints out information about wave being broken (default=False).
        """

        if self.broken:
            raise ValueError('Wave is already broken')

        if self.depth is None:
            self.depth = self.wave_length
        depth = self.depth - precision
        while not self.broken:
            self.propagate(depth, echo=echo)
            depth -= precision

    def validate(self):
        """
        Shows what wave theory is applicable for given wave parameters using the
        wave theories' figure by Le Mehaute per USACE CEM Part II Chap. 1 p.II-1-58

        Returns
        -------
        (fig, ax)
        """

        return coastlib.waves.support.wave_theories(
            wave_height=self.wave_height, wave_period=self.wave_period, depth=self.depth, g=self.g
        )


if __name__ == '__main__':
    self = AiryWave(wave_height=6, wave_period=6, depth=20, angle=45)
    self.break_wave(echo=True)
    self.validate()
