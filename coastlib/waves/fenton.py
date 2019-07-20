import os
import shutil
import subprocess
import threading

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants

import coastlib.waves.support
import coastlib.helper.environment


coastlib.helper.environment.append_bin()


def _generate_data(
        h_by_d, measure_of_length, value_of_that_length,
        current_criterion, current_magnitude, n, number_of_height_steps
):
    return str(
        f'Wave settings\n'
        f'{h_by_d:.20f}\n'
        f'{measure_of_length}\n'
        f'{value_of_that_length:.20f}\n'
        f'{current_criterion:d}\n'
        f'{current_magnitude:.20f}\n'
        f'{n:d}\n'
        f'{number_of_height_steps:d}\n'
        f'FINISH'
    )


def _generate_convergence(max_iter, crit_conv):
    return str(
        f'Convergence settings\n'
        f'{max_iter:d}\n'
        f'{crit_conv}'
    )


def _generate_points(n_surface, n_profiles, n_vertical):
    return str(
        f'Output settings\n'
        f'{n_surface:d}\n'
        f'{n_profiles:d}\n'
        f'{n_vertical:d}'
    )


class FentonWave:
    """
    This class provides interface to John Fenton's stream function wave theory program Fourier.
    See http://johndfenton.com/Steady-waves/Fourier.html

    Parameters
    ----------
    wave_height : float
        Wave height in meters.
    depth : float
        Water depth in meters.
    wave_period : float, optional
        Wave period in seconds. Mandatory, if wave_length is not provided.
    wave_length : float, optional
        Wave length in meters. Mandatory, if wave_period is not provided.
    timeout : float, optional
        Timeout in seconds (default=5). Fourier.exe is terminated after timeout.
    kwargs
        g : float, optional
            Earth gravity in m/s^2 (default=scipy.constants.g).
        rho : float, optional
            Water density in kg/m^3 (default=1025).
        current_criterion : int, optional
            1 for Eulerian mean current, 2 for mass-transport velocity (default=1).
        current_velocity : float, optional
            Current velocity in meters per second (default=0).
        fourier_components : int, optional
            Number of Fourier components.
            10-20 OK for ordinary waves, more for highest (default=20).
        height_steps : int, optional
            Maximum number of iterations for each height step.
            10 OK for ordinary waves, 40 for highest (default=10).
        convergence : dict, optional
            Maximum number of iterations for each height step; 10 OK for ordinary waves, 40 for highest.
            Criterion for convergence, typically '1.e4' or '1.e5' for highest waves.
            default=dict(max_iter=40, crit_conv='1.e-5').
        points : dict, optional
            Number of points on free surface.
            Number of velocity/acceration profiles over half a wavelength, including 0 and lambda/2.
            Number of vertical points in each profile, including points at bottom and surface.
            default=dict(n_surface=10, n_profiles=100, n_vertical=50).

    Public Attributes
    -----------------
    self.__init__()
        self.wave_height : float
        self.depth : float
        self.g : float
        self.rho : float
        self.measure_of_length : str
        self.wave_period : float
        self.wave_length : float
        self.current_criterion : int
        self.current_velocity : float
        self.fourier_components : int
        self.height_steps : int
        self.convergence : dict
        self.points : dict
        self.data : dict

        self.fourier_log : str

        self.solution : pd.DataFrame
        self.surface : pd.DataFrame
        self.flowfield : pd.DataFrame

        self.wave_speed
        self.eulerian_current
        self.stokes_current
        self.mean_fluid_speed
        self.wave_volume_flux
        self.bernoulli_constant_r
        self.volume_flux
        self.bernoulli_constant_R
        self.momentum_flux
        self.impulse
        self.kinetic_energy
        self.potential_energy
        self.mean_square_of_bed_velocity
        self.radiation_stress
        self.wave_power

    Private Attributes
    ------------------
    self.__init__()
        self.__killed

    Public Methods
    --------------
    self.plot
    self.animate
    self.validate

    Private Methods
    ---------------
    self.__init__
    self.__repr__
    self.__run
    self.__write_inputs
    self.__run_fourier
    self.__parse_outputs
    """

    def __init__(self, wave_height, depth, wave_period=None, wave_length=None, timeout=5, **kwargs):
        """
        Initializes the FentonWave class instance object.

        Parameters
        ----------
        wave_height : float
            Wave height in meters.
        depth : float
            Water depth in meters.
        wave_period : float, optional
            Wave period in seconds. Mandatory, if wave_length is not provided.
        wave_length : float, optional
            Wave length in meters. Mandatory, if wave_period is not provided.
        timeout : float, optional
            Timeout in seconds (default=5). Fourier.exe is terminated after timeout.
        kwargs
            g : float, optional
                Earth gravity in m/s^2 (default=scipy.constants.g).
            rho : float, optional
                Water density in kg/m^3 (default=1025).
            current_criterion : int, optional
                1 for Eulerian mean current, 2 for mass-transport velocity (default=1).
            current_velocity : float, optional
                Current velocity in meters per second (default=0).
            fourier_components : int, optional
                Number of Fourier components.
                10-20 OK for ordinary waves, more for highest (default=20).
            height_steps : int, optional
                Maximum number of iterations for each height step.
                10 OK for ordinary waves, 40 for highest (default=10).
            convergence : dict, optional
                Maximum number of iterations for each height step; 10 OK for ordinary waves, 40 for highest.
                Criterion for convergence, typically '1.e4' or '1.e5' for highest waves.
                default=dict(max_iter=40, crit_conv='1.e-5').
            points : dict, optional
                Number of points on free surface.
                Number of velocity/acceration profiles over half a wavelength, including 0 and lambda/2.
                Number of vertical points in each profile, including points at bottom and surface.
                default=dict(n_surface=10, n_profiles=100, n_vertical=50).
        """

        self.wave_height = wave_height
        self.depth = depth

        # Physical parameters
        self.g = kwargs.pop('g', scipy.constants.g)
        self.rho = kwargs.pop('rho', 1025)

        # Wave length setup
        if wave_period is None and wave_length is None:
            raise ValueError('either wave_period or wave_length must be provided')
        elif wave_period is not None and wave_length is not None:
            raise ValueError('both wave_period or wave_length were be provided, only one is required')
        elif wave_period is not None:
            self.measure_of_length = 'Period'
            self.wave_period = wave_period
            value_of_that_length = self.wave_period * np.sqrt(self.g / depth)
        elif wave_length is not None:
            self.measure_of_length = 'Wavelength'
            self.wave_length = wave_length
            value_of_that_length = self.wave_length / depth
        else:
            raise RuntimeError

        # Current setup
        self.current_criterion = kwargs.pop('current_criterion', 1)
        self.current_velocity = kwargs.pop('current_velocity', 0)
        if self.current_criterion not in [1, 2]:
            raise ValueError(f'current_criterion must be 1 or 2, {self.current_criterion} was passed')

        # Numerical solution parameters
        self.fourier_components = kwargs.pop('fourier_components', 20)
        self.height_steps = kwargs.pop('height_steps', 10)
        self.convergence = kwargs.pop('convergence', dict(max_iter=40, crit_conv='1.e-5'))

        # Output quantity
        self.points = kwargs.pop('points', dict(n_surface=10, n_profiles=100, n_vertical=50))

        self.data = dict(
            h_by_d=self.wave_height / self.depth,
            measure_of_length=self.measure_of_length,
            value_of_that_length=value_of_that_length,
            current_criterion=self.current_criterion,
            current_magnitude=self.current_velocity / np.sqrt(self.g * self.depth),
            n=self.fourier_components,
            number_of_height_steps=self.height_steps
        )

        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        self.__killed = False
        self.__run(os.path.join(os.environ['TEMP'], 'fenton_temp'), timeout)

    def __repr__(self):
        """
        Generates a string with a summary of the FentonWave class instance object.
        """

        summary = str(
            f'{" "*24}Fenton Wave\n'
            f'{"="*59}\n'
        )
        summary += str(self.solution.round(3))
        summary += f'\n{"="*59}'
        return summary

    def __run(self, path, timeout):
        """
        Executes entire run sequence (write inputs -> run Fourier -> parse outputs -> clean up).

        Parameters
        ----------
        path : str
            Path to work folder.
        timeout : float, optional
            Timeout in seconds (default=5). Fourier.exe is terminated after timeout.
        """

        # Create work folder. Remove if already existed
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        # Write inputs and try to run Fourier.exe
        self.__write_inputs(path)
        self.__run_fourier(path, timeout)

        # Clean up if Fourier.exe was terminated
        if self.__killed:
            shutil.rmtree(path)
            if self.measure_of_length == 'Period':
                raise TimeoutError(
                    f'Fourier.exe was not responding and was terminated after {timeout:.2f} seconds. '
                    f'Likely a convergence problem caused by an inadequate combination of input parameters:\n'
                    f'wave_height={self.wave_height:.2f}, wave_period={self.wave_period:.2f}, depth={self.depth:.2f}'
                )
            elif self.measure_of_length == 'Wavelength':
                raise TimeoutError(
                    f'Fourier.exe was not responding and was terminated after {timeout:.2f} seconds. '
                    f'Likely a convergence problem caused by an inadequate combination of input parameters:\n'
                    f'wave_height={self.wave_height:.2f}, wave_length={self.wave_length:.2f}, depth={self.depth:.2f}'
                )

        # Parse output and clean up
        self.__parse_outputs(path)
        shutil.rmtree(path)

    def __write_inputs(self, path):
        """
        Generates Data.dat, Convergence.dat, and Points.dat input files for the Fourier.exe program.

        Parameters
        ----------
        path : str
            Path to work folder.
        """

        with open(os.path.join(path, 'Data.dat'), 'w') as f:
            f.write(_generate_data(**self.data))

        with open(os.path.join(path, 'Convergence.dat'), 'w') as f:
            f.write(_generate_convergence(**self.convergence))

        with open(os.path.join(path, 'Points.dat'), 'w') as f:
            f.write(_generate_points(**self.points))

    def __run_fourier(self, path, timeout):
        """
        Executes Fourier.exe for inputs created by <self._write_inputs()>. Writes an internal log.

        Parameters
        ----------
        path : str
            Path to work folder.
        timeout : float, optional
            Timeout in seconds (default=5). Fourier.exe is terminated after timeout.
        """

        log = []
        curdir = os.getcwd()
        os.chdir(path=path)
        p = subprocess.Popen('Fourier', bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        def killer(process):
            process.kill()
            self.__killed = True

        timer = threading.Timer(timeout, killer, [p])
        try:
            timer.start()
            while p.poll() is None:
                line = p.stdout.readline()
                try:
                    line = line.decode()
                    if len(line) > 0:
                        log.append(line)
                except AttributeError:
                    pass
        finally:
            timer.cancel()
        log.extend([f'\n=== "Fourier.exe" process finished with an exit code ({p.poll()}) ==='])
        self.fourier_log = ''.join(log)
        os.chdir(curdir)

    def __parse_outputs(self, path):
        """
        Parses dimensionless Fourier.exe outputs *.res and converts to dimensional data.

        Parameters
        ----------
        path : str
            Path to work folder.
        """

        # Parse Solution-Flat.res/Solution.res
        if 'Solution-Flat.res' in os.listdir(path):
            with open(os.path.join(path, 'Solution-Flat.res'), 'r') as f:
                rows, values = [], []
                for i, line in enumerate(f.readlines()[2:21]):
                    s_line = line.rstrip().split('\t')
                    rows.append(
                        ' '.join(
                            [
                                s_line[3][2:].split('  ')[0],
                                s_line[3][2:].split('  ')[-1].split(' ')[-1]
                            ]
                        )
                    )
                    values.append([float(s_line[1]), float(s_line[2])])
            self.solution = pd.DataFrame(
                data=values, index=rows, columns=pd.MultiIndex.from_tuples(
                    [
                        ('Solution non-dimensionalised by', 'g & wavenumber'),
                        ('Solution non-dimensionalised by', 'g & mean depth')
                    ]
                )
            )
            self.solution.index.name = 'Quantity (symbol)'
        else:
            with open(os.path.join(path, 'Solution.res'), 'r') as f:
                rows, values = [], []
                for line in f.readlines()[14:33]:
                    s_line = line.rstrip().split('\t')
                    rows.append(
                        ' '.join(
                            [
                                s_line[0][2:].split('  ')[0],
                                s_line[0][2:].split('  ')[-1].split(' ')[-1]
                            ]
                        )
                    )
                    values.append([float(s_line[1]), float(s_line[2])])
            self.solution = pd.DataFrame(
                data=values, index=rows, columns=pd.MultiIndex.from_tuples(
                    [
                        ('Solution non-dimensionalised by', 'g & wavenumber'),
                        ('Solution non-dimensionalised by', 'g & mean depth')
                    ]
                )
            )
            self.solution.index.name = 'Quantity (symbol)'

        # Parse Surface.res
        with open(os.path.join(path, 'Surface.res'), 'r') as f:
            surf = []
            for i, line in enumerate(f):
                if i >= 10 and len(line) > 3:
                    surf.append(
                        [float(value) for value in line.split(sep='\t')]
                    )
        self.surface = pd.DataFrame(data=surf, columns=['X/d', 'eta/d', 'check of surface pressure'])

        # Parse Flowfield.res
        with open(os.path.join(path, 'Flowfield.res'), 'r') as f:
            xd, phase, fields = [], [], []
            header = None
            for i, line in enumerate(f.readlines()[18:]):
                if line.startswith('# X/d'):
                    header = [float(line[7:16]), float(line[25:32])]
                elif len(line) > 3:
                    if header is None:
                        raise RuntimeError
                    xd.append(header[0])
                    phase.append(header[1])
                    fields.append([float(number) for number in line.split('\t')])
        self.flowfield = pd.DataFrame(
            data=fields, columns=[
                'y (d)', 'u (sqrt(gd))', 'v (sqrt(gd))', 'dphi/dt (gd)', 'du/dt (g)', 'dv/dt (g)',
                'du/dx (sqrt(g/d))', 'du/dy (sqrt(g/d))', 'Bernoully check (gd)'
            ]
        )
        self.flowfield['X/d'] = xd
        self.flowfield['Phase (deg)'] = phase

        # Convert self.solution from dimensionless to dimensional using g and mean depth
        summary = [row[1] for row in self.solution.values]
        self.wave_number = 2 * np.pi / (self.depth * summary[0] * summary[1])
        self.depth *= summary[0]
        self.wave_length = summary[1] * self.depth
        self.wave_height = summary[2] * self.depth
        self.wave_period = summary[3] / np.sqrt(self.g / self.depth)
        self.wave_speed = summary[4] * np.sqrt(self.g * self.depth)
        self.eulerian_current = summary[5] * np.sqrt(self.g * self.depth)
        self.stokes_current = summary[6] * np.sqrt(self.g * self.depth)
        self.mean_fluid_speed = summary[7] * np.sqrt(self.g * self.depth)
        self.wave_volume_flux = summary[8] * np.sqrt(self.g * self.depth ** 3)
        self.bernoulli_constant_r = summary[9] * (self.g * self.depth)
        self.volume_flux = summary[10] * np.sqrt(self.g * self.depth ** 3)
        self.bernoulli_constant_R = summary[11] * (self.g * self.depth)
        self.momentum_flux = summary[12] * (self.rho * self.g * self.depth ** 2)
        self.impulse = summary[13] * (self.rho * np.sqrt(self.g * self.depth ** 3))
        self.kinetic_energy = summary[14] * (self.rho * self.g * self.depth ** 2)
        self.potential_energy = summary[15] * (self.rho * self.g * self.depth ** 2)
        self.mean_square_of_bed_velocity = summary[16] * (self.g * self.depth)
        self.radiation_stress = summary[17] * (self.rho * self.g * self.depth ** 2)
        self.wave_power = summary[18] * (self.rho * self.g ** (3 / 2) * self.depth ** (5 / 2))
        self.solution = pd.DataFrame(
            data=[
                'm',  # d - depth
                'm',  # d - wave length
                'm',  # d - wave height
                's',  # /sqrt(g/d) - wave period
                'm/s',  # sqrt(gd) - wave speed
                'm/s',  # sqrt(gd) - eulerian current
                'm/s',  # sqrt(gd) - stokes current
                'm/s',  # sqrt(gd) - mean fluid speed
                'm^2/s',  # sqrt(gd^3) - wave volume flux
                '(m/s)^2',  # gd - bernoulli constant r
                'm^2/s',  # sqrt(gd^3) - volume flux
                '(m/s)^2',  # gd - bernoulli constant R
                'kg/s^2 or (N/m)',  # rho*gd^2 - momentum flux
                'kg/(m*s)',  # rho*sqrt(gd^3) - impulse
                'kg/s^2 or (N/m)',  # rho*gd^2 - kinetic energy
                'kg/s^2 or (N/m)',  # rho*gd^2 - potential energy
                '(m/s)^2',  # gd - mean square of bed velocity
                'kg/s^2 or (N/m)',  # rho*gd^2 - raidation stress
                'kg*m/s^3 or (W/m)'  # rho*g^(3/2)*d^(5/2) - wave power
            ],
            columns=[
                'Unit'
            ],
            index=[
                'depth',
                'wave length',
                'wave height',
                'wave period',
                'wave speed',
                'eulerian current',
                'stokes current',
                'mean fluid_speed',
                'wave volume flux',
                'bernoulli constant r',
                'volume flux',
                'bernoulli constant R',
                'momentum flux',
                'impulse',
                'kinetic energy',
                'potential energy',
                'mean square of bed velocity',
                'radiation stress',
                'wave_power'
            ]
        )
        self.solution['Value'] = [
            self.depth,
            self.wave_length,
            self.wave_height,
            self.wave_period,
            self.wave_speed,
            self.eulerian_current,
            self.stokes_current,
            self.mean_fluid_speed,
            self.wave_volume_flux,
            self.bernoulli_constant_r,
            self.volume_flux,
            self.bernoulli_constant_R,
            self.momentum_flux,
            self.impulse,
            self.kinetic_energy,
            self.potential_energy,
            self.mean_square_of_bed_velocity,
            self.radiation_stress,
            self.wave_power
        ]
        self.solution.index.rename('Parameter', inplace=True)

        # Convert self.surface from dimensionless to dimensional
        self.surface['X (m)'] = self.surface['X/d'].values * self.depth
        self.surface['eta (m)'] = self.surface['eta/d'].values * self.depth
        for _col in ['X/d', 'eta/d', 'check of surface pressure']:
            del self.surface[_col]

        # Convert self.flowfield from dimensionless to dimensional
        self.flowfield['X (m)'] = self.flowfield['X/d'].values * self.depth
        self.flowfield['y (m)'] = self.flowfield['y (d)'] * self.depth
        self.flowfield['u (m/s)'] = self.flowfield['u (sqrt(gd))'] * np.sqrt(self.g * self.depth)
        self.flowfield['v (m/s)'] = self.flowfield['v (sqrt(gd))'] * np.sqrt(self.g * self.depth)
        self.flowfield['dphi/dt (m^2/s^2)'] = self.flowfield['dphi/dt (gd)'] * self.g * self.depth
        self.flowfield['du/dt (m/s^2)'] = self.flowfield['du/dt (g)'] * self.g
        self.flowfield['dv/dt (m/s^2)'] = self.flowfield['dv/dt (g)'] * self.g
        self.flowfield['du/dx (1/s)'] = self.flowfield['du/dx (sqrt(g/d))'] * np.sqrt(self.g / self.depth)
        self.flowfield['du/dy (1/s)'] = self.flowfield['du/dy (sqrt(g/d))'] * np.sqrt(self.g / self.depth)
        for _col in [
            'y (d)', 'u (sqrt(gd))', 'v (sqrt(gd))', 'dphi/dt (gd)', 'du/dt (g)', 'dv/dt (g)',
            'du/dx (sqrt(g/d))', 'du/dy (sqrt(g/d))', 'Bernoully check (gd)', 'X/d'
        ]:
            del self.flowfield[_col]

    def plot(self, what='ua', scale=1, nprof=4):
        """
        Plots wave profile with multiple depth profiles for a given parameter.

        Parameters
        ----------
        what : str, optional
            Parameter to plot (defaul='ua').
        scale : float, optional
            Parameter scale (default=1).
        nprof : int, optional
            Number of parameter profiles (default=4).

        Returns
        -------
        tuple(fig, ax)
        """

        column = {
            'u': 'u (m/s)',
            'v': 'v (m/s)',
            'ua': 'du/dt (m/s^2)',
            'va': 'dv/dt (m/s^2)',
            'ux': 'du/dx (1/s)',
            'vx': 'du/dy (1/s)',
            'uy': 'du/dy (1/s)',
            'vy': 'du/dx (1/s)',
            'phi': 'dphi/dt (m^2/s^2)'
        }.pop(what)

        phases = np.unique(self.flowfield['X (m)'].values)
        surface = np.array(
            [self.flowfield[self.flowfield['X (m)'] == phase]['y (m)'].values.max() for phase in phases]
        )

        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.set_ylim([-0.1 * self.depth, 1.1 * self.surface['eta (m)'].values.max()])
            ax.set_xlim([self.surface['X (m)'].values.min() * 1.1, self.surface['X (m)'].values.max() * 1.1])

            ax.plot(
                np.unique(np.append(-phases[::-1], phases)),
                np.append(surface[::-1][:-1], surface),
                lw=3, color='k', zorder=5
            )  # free surface
            ax.plot(
                [-phases.max(), phases.max()],
                [0, 0], color='k', lw=3, ls='--', zorder=5
            )  # seabed

            for i in np.arange(0, len(phases), int(np.round(len(phases) / nprof))):
                profile = self.flowfield[self.flowfield['X (m)'] == phases[i]]
                ax.plot(
                    [phases[i], phases[i]],
                    [profile['y (m)'].values.min(), profile['y (m)'].values.max()],
                    color='k', lw=1, ls='--', zorder=5
                )  # vertical line
                if what == 'vy':
                    ax.plot(
                        [phases[i], phases[i] - profile[column].values[0] * scale],
                        [profile['y (m)'].values.min(), profile['y (m)'].values.min()],
                        color='orangered', lw=2, zorder=10
                    )  # horizontal line bottom
                    ax.plot(
                        [phases[i], phases[i] - profile[column].values[-1] * scale],
                        [profile['y (m)'].values.max(), profile['y (m)'].values.max()],
                        color='orangered', lw=2, zorder=10
                    )  # horizontal line top
                    ax.plot(
                        phases[i] - profile[column].values * scale,
                        profile['y (m)'].values,
                        color='orangered', lw=2, zorder=10
                    )  # profile
                else:
                    ax.plot(
                        [phases[i], phases[i] + profile[column].values[0] * scale],
                        [profile['y (m)'].values.min(), profile['y (m)'].values.min()],
                        color='orangered', lw=2, zorder=10
                    )  # horizontal line bottom
                    ax.plot(
                        [phases[i], phases[i] + profile[column].values[-1] * scale],
                        [profile['y (m)'].values.max(), profile['y (m)'].values.max()],
                        color='orangered', lw=2, zorder=10
                    )  # horizontal line top
                    ax.plot(
                        phases[i] + profile[column].values * scale,
                        profile['y (m)'].values,
                        color='orangered', lw=2, zorder=10
                    )  # profile

            ax.set_title(f'Fenton Wave, Parameter ${column}$')
            ax.set_xlabel('Phase (m)')
            ax.set_ylabel('Elevation (m)')
            fig.tight_layout()

            return fig, ax

    def animate(self, what='u', scale=1, fps=30, savepath=None):
        """
        Animates wave profile and a depth profile for a given parameter over time.

        Parameters
        ----------
        what : str, optional
            Parameter to plot (defaul='ua').
        scale : float, optional
            Parameter scale (default=1).
        fps : float, optional
            Frames-per-second (default=30).
        savepath : str, optional
            Path to *.mp4 file with the animation to be created.
            Requires imagemagick with ffmpeg being installed.

        Returns
        -------
        matplotlib animation object
        """

        column = {
            'u': 'u (m/s)',
            'v': 'v (m/s)',
            'ua': 'du/dt (m/s^2)',
            'va': 'dv/dt (m/s^2)',
            'ux': 'du/dx (1/s)',
            'vx': 'du/dy (1/s)',
            'uy': 'du/dy (1/s)',
            'vy': 'du/dx (1/s)',
            'phi': 'dphi/dt (m^2/s^2)'
        }.pop(what)

        phases = np.unique(self.flowfield['X (m)'].values)
        eta = np.array(
            [self.flowfield[self.flowfield['X (m)'] == phase]['y (m)'].values.max() for phase in phases]
        )

        # Mirror phases and water surface to get a full wave profile
        phases = np.unique(np.append(-phases[::-1], phases))
        etas = np.append(eta[::-1][:-1], eta)

        with plt.style.context('bmh'):
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.set_ylim([-0.1 * self.depth, 1.1 * self.surface['eta (m)'].values.max()])
            ax.set_xlim([self.surface['X (m)'].values.min() * 1.1, self.surface['X (m)'].values.max() * 1.1])

            surface, = ax.plot(phases, etas, lw=3, color='k', zorder=5)  # free surface
            lines = [surface]
            ax.plot(
                [-phases.max(), phases.max()],
                [0, 0], color='k', lw=3, ls='--', zorder=5
            )  # seabed

            profile = self.flowfield[self.flowfield['X (m)'] == 0]
            vert, = ax.plot(
                [0, 0],
                [profile['y (m)'].values.min(), profile['y (m)'].values.max()],
                color='k', lw=1, ls='--', zorder=5
            )  # vertical line
            if what == 'vy':
                hbot, = ax.plot(
                    [0, 0 - profile[column].values[0] * scale],
                    [profile['y (m)'].values.min(), profile['y (m)'].values.min()],
                    color='orangered', lw=2, zorder=10
                )  # horizontal line bottom
                htop, = ax.plot(
                    [0, 0 - profile[column].values[-1] * scale],
                    [profile['y (m)'].values.max(), profile['y (m)'].values.max()],
                    color='orangered', lw=2, zorder=10
                )  # horizontal line top
                prof, = ax.plot(
                    0 - profile[column].values * scale,
                    profile['y (m)'].values,
                    color='orangered', lw=2, zorder=10
                )  # profile
            else:
                hbot, = ax.plot(
                    [0, 0 + profile[column].values[0] * scale],
                    [profile['y (m)'].values.min(), profile['y (m)'].values.min()],
                    color='orangered', lw=2, zorder=10
                )  # horizontal line bottom
                htop, = ax.plot(
                    [0, 0 + profile[column].values[-1] * scale],
                    [profile['y (m)'].values.max(), profile['y (m)'].values.max()],
                    color='orangered', lw=2, zorder=10
                )  # horizontal line top
                prof, = ax.plot(
                    0 + profile[column].values * scale,
                    profile['y (m)'].values,
                    color='orangered', lw=2, zorder=10
                )  # profile

            lines.append(vert)
            lines.append(hbot)
            lines.append(htop)
            lines.append(prof)

            ax.set_title(f'Fenton Wave, Parameter ${column}$, {0:5.2f} sec')
            ax.set_xlabel('Phase (m)')
            ax.set_ylabel('Elevation (m)')
            fig.tight_layout()

            def find_nearest(array, value):
                idx = np.abs(array - value).argmin()
                return np.abs(array[idx])

            def get_frame(i):
                ax.set_title(f'Fenton Wave, Parameter ${column}$, {i:5.2f} sec')
                if i <= self.wave_period / 2:
                    frame_phase = (i / self.wave_period) * self.wave_length
                    eta_phase = frame_phase - self.wave_length / 2
                else:
                    frame_phase = (i / self.wave_period) * self.wave_length - self.wave_length
                    eta_phase = frame_phase + self.wave_length / 2
                idx = np.abs(phases - eta_phase).argmin()
                frame_etas = np.append(etas[idx:], etas[:idx])[::-1]
                for j, line in enumerate(lines):
                    frame_profile = self.flowfield[self.flowfield['X (m)'] == find_nearest(phases, frame_phase)]
                    # Update free surface
                    if j == 0:
                        line.set_data(phases, frame_etas)
                    # Update vertical line
                    elif j == 1:
                        line.set_data(
                            [0, 0],
                            [frame_profile['y (m)'].values.min(), frame_profile['y (m)'].values.max()],
                        )
                    # Update bottom horizontal line
                    elif j == 2:
                        if what == 'vy':
                            line.set_data(
                                [0, 0 - frame_profile[column].values[0] * scale],
                                [frame_profile['y (m)'].values.min(), frame_profile['y (m)'].values.min()]
                            )
                        else:
                            line.set_data(
                                [0, 0 + frame_profile[column].values[0] * scale],
                                [frame_profile['y (m)'].values.min(), frame_profile['y (m)'].values.min()]
                            )
                    # Update top horizontal line
                    elif j == 3:
                        if what == 'vy':
                            line.set_data(
                                [0, 0 - frame_profile[column].values[-1] * scale],
                                [frame_profile['y (m)'].values.max(), frame_profile['y (m)'].values.max()]
                            )
                        else:
                            line.set_data(
                                [0, 0 + frame_profile[column].values[-1] * scale],
                                [frame_profile['y (m)'].values.max(), frame_profile['y (m)'].values.max()]
                            )
                    # Update profile
                    elif j == 4:
                        if what == 'vy':
                            line.set_data(
                                0 - frame_profile[column].values * scale,
                                frame_profile['y (m)'].values
                            )
                        else:
                            line.set_data(
                                0 + frame_profile[column].values * scale,
                                frame_profile['y (m)'].values
                            )
                return lines

            # Type hint for pycharm inspection
            timesteps: int = np.arange(0, self.wave_period+1/fps, 1/fps)

            animation = matplotlib.animation.FuncAnimation(
                fig=fig, func=get_frame, frames=timesteps,
                interval=1000/fps, repeat=True, blit=False
            )

            if savepath is not None:
                writer = matplotlib.animation.writers['ffmpeg'](fps=fps, bitrate=1800)
                animation.save(savepath, writer=writer)

            return animation

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
    try:
        self = FentonWave(wave_height=4000, wave_period=8, depth=1, current_velocity=1)
    except TimeoutError as e:
        print(e)
        self = FentonWave(
            wave_height=6, wave_period=6, depth=12, current_velocity=0,
            timeout=30, fourier_components=20, height_steps=20,
            points=dict(n_surface=10, n_profiles=60*6, n_vertical=60*6)
        )
        self.validate()
        _fig, _ax = self.plot(what='ua', scale=1, nprof=4)
        _ani = self.animate(
            what='u', scale=2, fps=60,
            savepath=os.path.join(r'C:', os.environ['HOMEPATH'], r'Desktop\fenton.mp4')
        )
