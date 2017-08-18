import os
import shutil
import subprocess
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants


def _fdata(title, h_to_d, measure, value_of_that_length, current_criterion, current_magnitude, n, height_steps):
    data = r'''{title}
{h_to_d:.20f}	H/d
{measure}	Measure of length: "Wavelength" or "Period"
{value_of_that_length:.20f} 	Value of that length: L/d or T(g/d)^1/2 respectively
{current_criterion:d}		Current criterion (1 or 2)
{current_magnitude:.20f}		Current magnitude, (dimensionless) ubar/(gd)^1/2
{n:d}		Number of Fourier components or Order of Stokes/cnoidal theory
{height_steps:d}		Number of height steps to reach H/d
FINISH
'''.format(
        title=title, h_to_d=h_to_d, measure=measure, value_of_that_length=value_of_that_length,
        current_criterion=current_criterion, current_magnitude=current_magnitude,
        n=n, height_steps=height_steps
    )
    return data


def _fonvergence(maximum_number_of_iterations, criterion_for_convergence):
    convergence = r'''Control file to control convergence and output of results
{max_n_i:d}		Maximum number of iterations for each height step; 10 OK for ordinary waves, 40 for highest
{crit}	Criterion for convergence, typically 1.e-4, or 1.e-5 for highest waves
'''.format(max_n_i=maximum_number_of_iterations, crit=criterion_for_convergence)
    return convergence


def _fpoints(m, ua, vert):
    points = r'''Control output for graph plotting
{m:d}	Number of points on free surface (clustered near crest)
{ua:d}	Number of velocity profiles over half a wavelength to print out
{vert:d}	Number of vertical points in each profile
'''.format(m=m, ua=ua, vert=vert)
    return points


class FentonWave:
    """
    Mandatory inputs
    ================
    dimensional mode
        wave_height : float
            wave height in meters
        wave_period : float
            wave period in seconds
        depth : float
            water depth in meters
    dimensionless mode
        data : dict
            {
                'title'               : <title>, : str
                'h_to_d'              : <wave height> / <depth>, : float / float
                'measure'             : 'Period', : str (read manual before changing)
                'value_of_that_length': <wave period> * sqrt(<g> / <depth>), : float * sqrt(float * float)
                'current_criterion'   : 1, : int (read manual before changing)
                'current_magnitude'   : <current velocity> / sqrt(<g> * <depth>), : float / sqrt(float * float)
                'n'                   : 20, : int (read manual before changing)
                'height_steps'        : 10 : int (read manual before changing)
            }

    Optional inputs
    ===============
    current_velocity : float (default=0)
        Eulerian current velocity in m/s for dimensional mode
    path : str (default=None)
        save output to this folder
    g : float (default=scipy.constants.g) ~9.81
        gravity acceleration in m/s^2
    rho : float (default=1025)
        water density in kg/m^3
    max_iterations : int (default=20)
        maximum number of iterations
    run_title : str (default='Wave')
        title of a run
    convergence : dict
        {
            'maximum_number_of_iterations': 40, : int (read manual before changing)
            'criterion_for_convergence'   : '1.e-4' : str (read manual before changing)
        }
    points : dict
        {
            'm'   : 10, : int (read manual before changing)
            'ua'  : 10, : int (read manual before changing)
            'vert': 10 : int (read manual before changing)
        }

    Methods
    =======
    report : echoes solution summary and returns a dataframe with solution specifics
    plot : plots surface profile and velocity/acceleration profile slices
    propagate : propagates the wave to a new depth
    """

    def __init__(self, **kwargs):

        self._path: str = kwargs.pop('path', os.path.join(os.environ['TEMP'], 'fenton_temp'))
        self._g: typing.Union[float, int] = kwargs.pop('g', scipy.constants.g)
        self._rho: typing.Union[float, int] = kwargs.pop('rho', 1025)
        self._max_iterations: int = kwargs.pop('max_iterations', 3)
        self.run_title: str = kwargs.pop('run_title', 'Wave')

        self.wave_height: typing.Union[str, float] = kwargs.pop('wave_height', 'dimensionless')
        self.wave_period: typing.Union[str, float] = kwargs.pop('wave_period', 'dimensionless')
        self.depth: typing.Union[str, float] = kwargs.pop('depth', 'dimensionless')
        self.measure_of_wave_length: str = kwargs.pop('measure_of_wave_length', 'Period')
        self.current_criterion: int = kwargs.pop('current_criterion', 1)  #
        self.current_velocity: typing.Union[float, int] = kwargs.pop('current_velocity', 0)
        self.fourier_components: int = kwargs.pop('fourier_components', 20)
        self.height_steps: int = kwargs.pop('height_steps', 10)
        self.convergence: dict = kwargs.pop(
            'convergence',
            {
                'maximum_number_of_iterations': 40,
                'criterion_for_convergence'   : '1.e-4'
            }  # recommended convergence criteria
        )
        self.points: dict = kwargs.pop(
            'points',
            {
                'm'   : 10,
                'ua'  : 10,
                'vert': 10
            }  # 100 points per length/height by default
        )
        try:
            # Dimensional mode
            self.data: dict = {
                'title'               : self.run_title,
                'h_to_d'              : self.wave_height / self.depth,
                'measure'             : self.measure_of_wave_length,
                'value_of_that_length': self.wave_period * np.sqrt(self._g / self.depth),
                'current_criterion'   : self.current_criterion,
                'current_magnitude'   : self.current_velocity / np.sqrt(self._g * self.depth),
                'n'                   : self.fourier_components,
                'height_steps'        : self.height_steps,
            }
        except TypeError:
            try:
                # Dimensionless mode
                self.data: dict = kwargs.pop('data')
            except KeyError:
                assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
                raise ValueError('Incorrect inputs. Either dimensionless <data> dictionary or '
                                 'wave_height/period/depth should be provied')
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        self.__run()

    def __repr__(self):

        if isinstance(self.wave_height, str):
            _prefix = 'FentonWave class object. Dimensionless mode'
        else:
            _prefix = 'FentonWave class object. Dimensional mode'
        _prefix = [_prefix, '='*58]
        _report = str(self.solution.round(2)).split('\n')
        _report[0] = _report[1][:9] + _report[0][9:]
        _report[1] = ' '
        return '\n'.join(_prefix + _report)

    def __str__(self):
        _report = str(self.solution.round(2)).split('\n')
        _report[0] = _report[1][:9] + _report[0][9:]
        _report[1] = ' '
        return '\n'.join(_report)

    def __write_inputs(self):
        """
        Generates *.dat input files for the Fourier.exe program
        """

        with open(os.path.join(self._path, 'Data.dat'), 'w') as f:
            f.write(_fdata(**self.data))
        with open(os.path.join(self._path, 'Convergence.dat'), 'w') as f:
            f.write(_fonvergence(**self.convergence))
        with open(os.path.join(self._path, 'Points.dat'), 'w') as f:
            f.write(_fpoints(**self.points))

    def __execute_fourier(self):
        """
        Executes Fourier.exe for inputs written by <self.__write_inputs()>
        """

        self.log = []  # Fourier.exe logs (stdout)
        curdir = os.getcwd()
        os.chdir(path=self._path)
        p = subprocess.Popen('Fourier', bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        while p.poll() is None:
            line = p.stdout.readline()
            try:
                line = line.decode()
                if len(line) > 0:
                    self.log.append(line)
            except AttributeError:
                pass
        self.log.extend(['\n=== "Fourier.exe" process finished with an exit code ({}) ==='.format(p.poll())])
        self.log = ''.join(self.log)
        os.chdir(curdir)

    def __parse(self):
        """
        Parses non-dimensional *.res outputs of the Fourier.exe program
        """

        # Parse Soluion.res
        with open(os.path.join(self._path, 'Solution.res'), 'r') as f:
            rows, values = [], []
            for i, line in enumerate(f):
                if 14 <= i < 33:
                    s_line = line.split('\t')
                    rows.append(s_line[0])
                    values.append([float(s_line[1]), float(s_line[2])])
            self.solution = pd.DataFrame(data=values, index=rows, columns=pd.MultiIndex.from_tuples([
                ('Solution non-dimensionalised by', 'g & wavenumber'),
                ('Solution non-dimensionalised by', 'g & mean depth')
            ]))

        # Parse Surface.res
        with open(os.path.join(self._path, 'Surface.res'), 'r') as f:
            f_len = sum(1 for _ in f)
        with open(os.path.join(self._path, 'Surface.res'), 'r') as f:
            surf = []
            for i, line in enumerate(f):
                if 8 <= i < (f_len - 1):
                    surf.append(
                        [float(value) for value in line.split(sep='\t')]
                    )
        self.surface = pd.DataFrame(data=surf, columns=['X (d)', 'eta (d)', 'pressure check'])

        # Parse Flowfield.res
        with open(os.path.join(self._path, 'Flowfield.res'), 'r') as f:
            xd, phase, fields, _field = [], [], [], []
            for i, line in enumerate(f):
                if i >= 14:
                    if line[:3] == '# X':
                        _header = [float(line[7:16]), float(line[25:32])]
                    elif line == '\n' and len(_field) != 0:
                        fields.extend(_field)
                        xd.extend([_header[0]] * len(_field))
                        phase.extend([_header[1]] * len(_field))
                        _field = []
                    elif line != '\n':
                        _field.append([float(number) for number in line.split('\t')])
        self.flowfield = pd.DataFrame(data=fields, columns=[
            'Y (d)', 'u (sqrt(gd))', 'v (sqrt(gd))', 'dphi/dt (gd)', 'du/dt (g)', 'dv/dt (g)', 'du/dx (sqrt(g/d))',
            'du/dy (sqrt(g/d))', 'Bernoully check (gd)'
        ])
        self.flowfield['X (d)'] = xd
        self.flowfield['Phase (deg)'] = phase

    def __dimensionalize(self):
        """
        If the program was given dimensional data, updates attributes and dimensionalizes output data
        """

        if not isinstance(self.depth, str):

            # Dimensionalize solution
            summary = [row[1] for row in self.solution.values]
            self.depth *= summary[0]
            self.wave_length = summary[1] * self.depth
            self.wave_height = summary[2] * self.depth
            self.wave_period = summary[3] / np.sqrt(self._g / self.depth)
            self.wave_speed = summary[4] * np.sqrt(self._g * self.depth)
            self.eulerian_current = summary[5] * np.sqrt(self._g * self.depth)
            self.stokes_current = summary[6] * np.sqrt(self._g * self.depth)
            self.mean_fluid_speed = summary[7] * np.sqrt(self._g * self.depth)
            self.wave_volume_flux = summary[8] * np.sqrt(self._g * self.depth ** 3)
            self.bernoulli_constant_r = summary[9] * (self._g * self.depth)
            self.volume_flux = summary[10] * np.sqrt(self._g * self.depth ** 3)
            self.bernoulli_constant_R = summary[11] * (self._g * self.depth)
            self.momentum_flux = summary[12] * (self._rho * self._g * self.depth ** 2)
            self.impulse = summary[13] * (self._rho * np.sqrt(self._g * self.depth ** 3))
            self.kinetic_energy = summary[14] * (self._rho * self._g * self.depth ** 2)
            self.potential_energy = summary[15] * (self._rho * self._g * self.depth ** 2)
            self.mean_square_of_bed_velocity = summary[16] * (self._g * self.depth)
            self.radiation_stress = summary[17] * (self._rho * self._g * self.depth ** 2)
            self.wave_power = summary[18] * (self._rho * self._g ** (3/2) * self.depth ** (5/2))
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

            # Dimensionalize surface
            self.surface['X (m)'] = self.surface['X (d)'].values * self.depth
            self.surface['eta (m)'] = self.surface['eta (d)'].values * self.depth
            for _col in ['X (d)', 'eta (d)']:
                del self.surface[_col]

            # Dimensionalize flowfield
            self.flowfield['Y (m)'] = self.flowfield['Y (d)'] * self.depth
            self.flowfield['u (m/s)'] = self.flowfield['u (sqrt(gd))'] * np.sqrt(self._g * self.depth)
            self.flowfield['v (m/s)'] = self.flowfield['v (sqrt(gd))'] * np.sqrt(self._g * self.depth)
            self.flowfield['dphi/dt (m^2/s^2)'] = self.flowfield['dphi/dt (gd)'] * self._g * self.depth
            self.flowfield['du/dt (m/s^2)'] = self.flowfield['du/dt (g)'] * self._g
            self.flowfield['dv/dt (m/s^2)'] = self.flowfield['dv/dt (g)'] * self._g
            self.flowfield['du/dx (1/s)'] = self.flowfield['du/dx (sqrt(g/d))']\
                                                      * np.sqrt(self._g / self.depth)
            self.flowfield['du/dy (1/s)'] = self.flowfield['du/dy (sqrt(g/d))']\
                                                      * np.sqrt(self._g / self.depth)
            self.flowfield['Bernoully check (m^2/s^2)'] = self.flowfield['Bernoully check (gd)']\
                                                          * self._g * self.depth
            self.flowfield['X (m)'] = self.flowfield['X (d)'].values * self.depth
            for _col in [
                'Y (d)', 'u (sqrt(gd))', 'v (sqrt(gd))', 'dphi/dt (gd)', 'du/dt (g)', 'dv/dt (g)',
                'du/dx (sqrt(g/d))', 'du/dy (sqrt(g/d))', 'Bernoully check (gd)', 'X (d)'
            ]:
                del self.flowfield[_col]

    def __run(self):

        # Make sure work folder exists
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        elif self._path.endswith('fenton_temp'):
            shutil.rmtree(self._path)
            os.makedirs(self._path)

        # Try to generate inputs, call Fourier.exe, and parse outputs 20 times. Raise exception if failure persists
        sucess = False
        for iteration in range(self._max_iterations):
            try:
                self.__write_inputs()
                self.__execute_fourier()
                self.__parse()
                self.__dimensionalize()
                sucess = True
                break
            except FileNotFoundError:
                raise RuntimeError('Fourier.exe was not executed or output files were removed by another application'
                                   '\nCould be caused by permission or antivirus related issues')
            except Exception as _e:
                print('Got\n    {0}.\n Failure after {1} iterations. Repeating'.format(_e, iteration + 1))

        if not sucess:
            try:
                print(self.log)
            except Exception as _e:
                print('No logs were generated due to'
                      '\n    {}'.format(_e))
            raise RuntimeError(
                'No result was achieved after {0} iterations.\n'
                'Check input for correctness. Read warnings with echoed exception\n'
                'Try running Fourier.exe manually in the generated folder'
                ' to see the error'.format(self._max_iterations)
            )

        # Clean up
        if self._path.endswith('fenton_temp'):
            shutil.rmtree(self._path)

    def plot(self, what='ua', savepath=None, scale=1, reduction=0, profiles=4):

        try:
            if not isinstance(self.wave_height, str):
                what = {
                    'u'    : 'u (m/s)',
                    'du/dt': 'du/dt (m/s^2)',
                    'ua'   : 'du/dt (m/s^2)',
                    'v'    : 'v (m/s)',
                    'dv/dt': 'dv/dt (m/s^2)',
                    'va'   : 'dv/dt (m/s^2)',
                    'ux'   : 'du/dx (1/s)',
                    'uy'   : 'du/dy (1/s)',
                    'phi'  : 'dphi/dt (m^2/s^2)'
                }.pop(what)
            else:
                what = {
                    'u'    : 'u (sqrt(gd))',
                    'du/dt': 'du/dt (g)',
                    'ua'   : 'du/dt (g)',
                    'v'    : 'v (sqrt(gd))',
                    'dv/dt': 'dv/dt (g)',
                    'va'   : 'dv/dt (g)',
                    'ux'   : 'du/dx (sqrt(g/d))',
                    'uy'   : 'du/dy (sqrt(g/d))',
                    'phi'  : 'dphi/dt (gd)'
                }.pop(what)
        except:
            raise ValueError('Unrecognized value passed in what={}'.format(what))

        with plt.style.context('bmh'):
            if isinstance(self.wave_height, str):
                plt.figure()
                plt.plot(self.surface['X (d)'].values, self.surface['eta (d)'].values, lw=2, color='royalblue')
                plt.ylim([-0.1, 1.1])
                plt.xlim([self.surface['X (d)'].values.min()*1.1, self.surface['X (d)'].values.max()*1.1])
                plt.plot([self.surface['X (d)'].values.min(), self.surface['X (d)'].values.max()],
                         [0, 0], color='saddlebrown', lw=2, ls='--')
                x_flow = np.unique(self.flowfield['X (d)'].values)  # List of phases
                plt.title(r'{} plot'.format(what))
                plt.xlabel('Phase (depths)')
                plt.ylabel('Surface elevation relative to seabed (depths)')

                for i in np.arange(0, len(x_flow), int(np.round(len(x_flow)/profiles))):
                    flow_loc = self.flowfield[self.flowfield['X (d)'] == x_flow[i]]
                    plt.plot(
                        [x_flow[i], x_flow[i]],
                        [flow_loc['Y (d)'].values.min(), flow_loc['Y (d)'].values.max()],
                        color='k', lw=1
                    )  # vertical line
                    plt.plot(
                        [x_flow[i], x_flow[i] + (flow_loc[what].values[0] * scale - reduction)],
                        [flow_loc['Y (d)'].values.min(), flow_loc['Y (d)'].values.min()],
                        color='orangered', lw=1
                    )  # horizontal line bottom
                    plt.plot(
                        [x_flow[i], x_flow[i] + (flow_loc[what].values[-1] * scale - reduction)],
                        [flow_loc['Y (d)'].values.max(), flow_loc['Y (d)'].values.max()],
                        color='orangered', lw=1
                    )  # horizontal line top
                    plt.plot(
                        flow_loc[what].values * scale + x_flow[i] - reduction,
                        flow_loc['Y (d)'].values,
                        color='orangered', lw=1
                    )  # profile
            else:
                plt.figure()
                plt.plot(self.surface['X (m)'].values, self.surface['eta (m)'].values, lw=2, color='royalblue')
                plt.ylim([-0.1 * self.depth, 1.1 * self.depth])
                plt.xlim([self.surface['X (m)'].values.min() * 1.1, self.surface['X (m)'].values.max() * 1.1])
                plt.plot([self.surface['X (m)'].values.min(), self.surface['X (m)'].values.max()],
                         [0, 0], color='saddlebrown', lw=2, ls='--')
                x_flow = np.unique(self.flowfield['X (m)'].values)  # List of phases
                plt.title(r'{} plot'.format(what))
                plt.xlabel('Phase (m)')
                plt.ylabel('Surface elevation relative to seabed (m)')

                for i in np.arange(0, len(x_flow), int(np.round(len(x_flow) / profiles))):
                    flow_loc = self.flowfield[self.flowfield['X (m)'] == x_flow[i]]
                    plt.plot(
                        [x_flow[i], x_flow[i]],
                        [flow_loc['Y (m)'].values.min(), flow_loc['Y (m)'].values.max()],
                        color='k', lw=1
                    )  # vertical line
                    plt.plot(
                        [x_flow[i], x_flow[i] + (flow_loc[what].values[0] * scale - reduction)],
                        [flow_loc['Y (m)'].values.min(), flow_loc['Y (m)'].values.min()],
                        color='orangered', lw=1
                    )  # horizontal line bottom
                    plt.plot(
                        [x_flow[i], x_flow[i] + (flow_loc[what].values[-1] * scale - reduction)],
                        [flow_loc['Y (m)'].values.max(), flow_loc['Y (m)'].values.max()],
                        color='orangered', lw=1
                    )  # horizontal line top
                    plt.plot(
                        flow_loc[what].values * scale + x_flow[i] - reduction,
                        flow_loc['Y (m)'].values,
                        color='orangered', lw=1
                    )  # profile
            plt.show()
            if savepath:
                plt.savefig(savepath, dpi=300, bbox_inches='tight')

    def highest(self):

        # TODO - implement for dimensionless waves (doesn't work with parsed solution)
        raise NotImplementedError
        # Check if wave is dimensional
        if isinstance(self.wave_height, str):
            raise ValueError('Implemented only for dimensional waves')

        # Rerun the solver until highest possible wave is resolved
        while True:
            # Set wave height to heighest for this depth
            if format(self.wave_height / self.depth, '.3f') == self.solution[4].split(' ')[-4]:
                break
            self.wave_height = float(self.solution[4].split(' ')[-4]) * self.depth

            try:
                # Dimensional mode
                self.data = {
                    'title'               : self.run_title,
                    'h_to_d'              : self.wave_height / self.depth,
                    'measure'             : self.measure_of_wave_length,
                    'value_of_that_length': self.wave_period * np.sqrt(self._g / self.depth),
                    'current_criterion'   : self.current_criterion,
                    'current_magnitude'   : self.current_velocity,
                    'n'                   : self.fourier_components,
                    'height_steps'        : self.height_steps,
                }
            except TypeError:
                raise ValueError('Implemented only for dimensional waves')

            self.__run()

    def propagate(self, new_depth):

        # TODO - not ready to be used (use Airy theory)
        # TODO - include shoalind and refraction, check if new wave is valid (realistic)
        raise NotImplementedError
        self.depth = new_depth

        try:
            # Dimensional mode
            self.data = {
                'title'               : self.run_title,
                'h_to_d'              : self.wave_height / self.depth,
                'measure'             : self.measure_of_wave_length,
                'value_of_that_length': self.wave_period * np.sqrt(self._g / self.depth),
                'current_criterion'   : self.current_criterion,
                'current_magnitude'   : self.current_velocity,
                'n'                   : self.fourier_components,
                'height_steps'        : self.height_steps,
            }
        except TypeError:
            raise ValueError('Only dimensional waves can be propagated')

        self.__run()


if __name__ == '__main__':
    wave = FentonWave(wave_height=2, wave_period=6, depth=20, current_velocity=0.5)
    wave.plot()
    print(str(wave))
