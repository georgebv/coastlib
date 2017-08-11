import os
import shutil
import subprocess
import warnings

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


def _finput(path, fdata, fpoints, fconvergence):
    with open(os.path.join(path, 'Data.dat'), 'w') as f:
        f.write(_fdata(**fdata))
    with open(os.path.join(path, 'Convergence.dat'), 'w') as f:
        f.write(_fonvergence(**fconvergence))
    with open(os.path.join(path, 'Points.dat'), 'w') as f:
        f.write(_fpoints(**fpoints))


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
    path : str (default=None)
        save output to this folder
    g : float (default=scipy.constants.g) ~9.81
        gravity acceleration in m^2/s
    rho : float (default=1025)
        water density in kg/m^3
    max_iterations : int (default=20)
        maximum number of iterations
    write_output : bool (default=False)
        if True, saves output to <path> as .csv and .pyc
    run_title : str (default='Wave')
        title of a run
    convergence : dict
        {
            'maximum_number_of_iterations': 20, : int (read manual before changing)
            'criterion_for_convergence'   : '1.e-4' : str (read manual before changing)
        }
    points : dict
        {
            'm'   : 100, : int (read manual before changing)
            'ua'  : 100, : int (read manual before changing)
            'vert': 100 : int (read manual before changing)
        }

    Methods
    =======
    report : echoes solution summary and returns a dataframe with solution specifics
    plot : plots surface profile and velocity/acceleration profile slices
    propagate : propagates the wave to a new depth
    """

    def __init__(self, **kwargs):

        # TODO - check if input data makes sense and tell how to improve it
        # TODO - recompile Fourier source to make direct import from Python

        self._path = kwargs.pop('path', os.path.join(os.environ['TEMP'], 'fenton_temp'))
        self._g = kwargs.pop('g', scipy.constants.g)
        self._rho = kwargs.pop('rho', 1025)
        self._max_iterations = kwargs.pop('max_iterations', 20)
        self._write_output = kwargs.pop('write_output', False)
        self.run_title = kwargs.pop('run_title', 'Wave')

        self.wave_height = kwargs.pop('wave_height', 'dimensionless')
        self.wave_period = kwargs.pop('wave_period', 'dimensionless')
        self.depth = kwargs.pop('depth', 'dimensionless')
        self.measure_of_wave_length = kwargs.pop('measure_of_wave_length', 'Period')
        self.current_criterion = kwargs.pop('current_criterion', 1)  #
        self.current_velocity = kwargs.pop('current_velocity', 0)
        self.fourier_components = kwargs.pop('fourier_components', 20)
        self.height_steps = kwargs.pop('height_steps', 10)
        self.convergence = kwargs.pop(
            'convergence',
            {
                'maximum_number_of_iterations': 20,
                'criterion_for_convergence'   : '1.e-4'
            }  # recommended convergence criteria
        )
        self.points = kwargs.pop(
            'points',
            {
                'm'   : 100,
                'ua'  : 100,
                'vert': 100
            }  # 100 points per length/height by default
        )
        try:
            # Dimensional mode
            self.data = {
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
                self.data = kwargs.pop('data')
            except Exception as _e:
                assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
                raise ValueError('Bad input provided. Got {}'.format(_e))
        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

        # Make sure work folder exists
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        # Try to generate inputs, call Fourier.exe, and parse outputs 20 times. Raise exception if failure persists
        sucess = False
        for iteration in range(self._max_iterations):
            try:
                self.__write_inputs()
                self.__run()
                self.__parse()
                sucess = True
                break
            except Exception as _e:
                warnings.warn(
                    'Got {0}. Failure after {1} iterations. Repeating'.format(_e, iteration+1)
                )
        if not sucess:
            try:
                print(self.log)
            except Exception as _e:
                print('No logs were generated due to {}'.format(_e))
            raise RuntimeError(
                'No result was achieved after {0} iterations.\n'
                'Check input for correctness. Read warnings with echoed exception'.format(self._max_iterations)
            )

        # Clean up
        if self._path.endswith('fenton_temp'):
            shutil.rmtree(self._path)
        self.__update_variables()

    def __write_inputs(self):

        with open(os.path.join(self._path, 'Data.dat'), 'w') as f:
            f.write(_fdata(**self.data))
        with open(os.path.join(self._path, 'Convergence.dat'), 'w') as f:
            f.write(_fonvergence(**self.convergence))
        with open(os.path.join(self._path, 'Points.dat'), 'w') as f:
            f.write(_fpoints(**self.points))

    def __run(self):

        self.log = []  # Fourier.exe logs (stdout)
        curdir = os.getcwd()
        os.chdir(path=self._path)
        p = subprocess.Popen('Fourier', bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        while p.poll() is None:
            line = p.stdout.readline()
            try:
                line = line.decode()
                if len(line) > 0:
                    self.log.extend([line])
            except AttributeError:
                pass
        self.log.extend(['\n=== Process exited with return code ({}) ==='.format(p.poll())])
        self.log = ''.join(self.log)
        os.chdir(curdir)

    def __parse(self):

        # Open files generated by Fourier.exe
        with open(os.path.join(self._path, 'Solution.res'), 'r') as f:
            self.solution = f.readlines()
        with open(os.path.join(self._path, 'Surface.res'), 'r') as f:
            surface = f.readlines()
        with open(os.path.join(self._path, 'Flowfield.res'), 'r') as f:
            flowfield = f.readlines()[14:]

        # Parse Surface.res
        surf = [i.split(sep='\t') for i in surface][8:-1]
        surf = [[float(value) for value in row] for row in surf]
        self.surface = pd.DataFrame(data=surf, columns=['X (d)', 'eta (d)', 'pressure check'])

        # Parse Flowfield.res
        xd, phase, fields, _field = [], [], [], []
        for line in flowfield:
            if line[:3] == '# X':
                _header = [float(line[7:16]), float(line[25:32])]
            elif line == '\n' and len(_field) != 0:
                fields.extend(_field)
                xd.extend([_header[0]] * len(_field))
                phase.extend([_header[1]] * len(_field))
                _field = []
            elif line != '\n':
                _field.extend([[float(number) for number in line.split('\t')]])
        self.flowfield = pd.DataFrame(data=fields, columns=[
            'Y (d)', 'u (sqrt(gd))', 'v (sqrt(gd))', 'dphi/dt (gd)', 'du/dt (g)', 'dv/dt (g)', 'du/dx (sqrt(g/d))',
            'du/dy (sqrt(gd))', 'Bernoully check (gd)'
        ])
        self.flowfield['X (d)'] = xd
        self.flowfield['Phase (deg)'] = phase

        # Write output to self._path
        if self._write_output:
            # Surface
            self.surface.to_pickle(os.path.join(self._path, 'surface.pyc'))
            self.surface.to_csv(os.path.join(self._path, 'surface.csv'))
            # Flowfield
            try:
                self.flowfield = self.flowfield.to_frame()
            except AttributeError:
                pass
            self.flowfield.to_pickle(os.path.join(self._path, 'flowfield.pyc'))
            self.flowfield.to_csv(os.path.join(self._path, 'flowfield.csv'))
            # Echo completion
            print('\nSaved output to\n    "{slout}"\n    "{sout}"\n    "{fout}"'.format(
                slout=os.path.join(self._path, 'solution.pyc'), sout=os.path.join(self._path, 'surface.pyc'),
                fout=os.path.join(self._path, 'flowfield.pyc')
            ))

    def __update_variables(self):

        # Update variables
        if self.depth != 'dimensionless':
            summary = [row[1] for row in self.__parse_solution(echo=False).values]
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

    def __parse_solution(self, echo=False):

        # Echo summary
        if echo:
            print(''.join(self.solution[:10]))

        # Parameters
        parameters = [line.split('\t') for line in self.solution[14:33]]
        rows, values = [], []
        for row in parameters:
            rows.extend([row[0]])
            values.extend([[float(row[1]), float(row[2])]])
        return pd.DataFrame(data=values, index=rows, columns=pd.MultiIndex.from_tuples([
            ('Solution non-dimensionalised by', 'g & wavenumber'),
            ('Solution non-dimensionalised by', 'g & mean depth')
        ]))

    def report(self, echo=True, nround=2):

        # Echo summary
        if echo:
            print(''.join(self.solution[:10]))

        # Parameters
        if not isinstance(self.wave_height, str):
            frame = pd.DataFrame(
                data=[
                    'm',        # d - depth
                    'm',        # d - wave length
                    'm',        # d - wave height
                    's',        # /sqrt(g/d) - wave period
                    'm/s',      # sqrt(gd) - wave speed
                    'm/s',      # sqrt(gd) - eulerian current
                    'm/s',      # sqrt(gd) - stokes current
                    'm/s',      # sqrt(gd) - mean fluid speed
                    'm^2/s',    # sqrt(gd^3) - wave volume flux
                    '(m/s)^2',  # gd - bernoulli constant r
                    'm^2/s',    # sqrt(gd^3) - volume flux
                    '(m/s)^2',  # gd - bernoulli constant R
                    'kg/s^2 or (N/m)',   # rho*gd^2 - momentum flux
                    'kg/(m*s)',  # rho*sqrt(gd^3) - impulse
                    'kg/s^2 or (N/m)',   # rho*gd^2 - kinetic energy
                    'kg/s^2 or (N/m)',   # rho*gd^2 - potential energy
                    '(m/s)^2',  # gd - mean square of bed velocity
                    'kg/s^2 or (N/m)',   # rho*gd^2 - raidation stress
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
            frame['Value'] = [
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
            frame.index.rename('Parameter', inplace=True)
            return frame.round(nround)
        else:
            # Echo dimensionless parameters if wave defined through dictionary
            return self.__parse_solution(echo=False)

    def plot(self, what, savepath=None, scale=0.5, reduction=0, profiles=4):

        try:
            what = {
                'u'    : 'u (sqrt(gd))',
                'du/dt': 'du/dt (g)',
                'ua'   : 'du/dt (g)',
                'v'    : 'v (sqrt(gd))',
                'dv/dt': 'dv/dt (g)',
                'va'   : 'dv/dt (g)'
            }.pop(what)
        except:
            raise ValueError('Unrecognized value passed in what={}'.format(what))

        with plt.style.context('bmh'):
            plt.figure()
            plt.plot(self.surface['X (d)'].values, self.surface['eta (d)'].values, lw=2, color='royalblue')
            plt.ylim([-0.1, 1.1])
            plt.xlim([self.surface['X (d)'].values.min()*1.1, self.surface['X (d)'].values.max()*1.1])
            plt.plot([self.surface['X (d)'].values.min(), self.surface['X (d)'].values.max()],
                     [0, 0], color='saddlebrown', lw=2, ls='--')

            x_flow = np.unique(self.flowfield['X (d)'].values)  # List of phases

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
            plt.show()
            if savepath:
                plt.savefig(savepath, dpi=300, bbox_inches='tight')

    def highest(self):

        # TODO - implement for dimensionless waves

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

            # Make sure work folder exists
            if not os.path.exists(self._path):
                os.makedirs(self._path)

            # Try to generate inputs, call Fourier.exe, and parse outputs 20 times. Raise exception if failure persists
            sucess = False
            for iteration in range(self._max_iterations):
                try:
                    self.__write_inputs()
                    self.__run()
                    self.__parse()
                    sucess = True
                    break
                except Exception as exception:
                    warnings.warn(
                        'Got {0}. Failure after {1} iterations. Repeating'.format(exception, iteration + 1)
                    )
            if not sucess:
                raise RuntimeError(
                    'No result was achieved after {0} iterations.\n'
                    'Check input for correctness. Read warnings with echoed exception'.format(self._max_iterations)
                )

            # Clean up
            if self._path.endswith('fenton_temp'):
                shutil.rmtree(self._path)
            self.__update_variables()

    def propagate(self, new_depth):

        # TODO - not ready to be used (use Airy theory)
        # TODO - include shoalind and refraction, check if new wave is valid (realistic)

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

        # Make sure work folder exists
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        # Try to generate inputs, call Fourier.exe, and parse outputs 20 times. Raise exception if failure persists
        sucess = False
        for iteration in range(self._max_iterations):
            try:
                self.__write_inputs()
                self.__run()
                self.__parse()
                sucess = True
                break
            except Exception as exception:
                warnings.warn(
                    'Got {0}. Failure after {1} iterations. Repeating'.format(exception, iteration + 1)
                )
        if not sucess:
            raise RuntimeError(
                'No result was achieved after {0} iterations.\n'
                'Check input for correctness. Read warnings with echoed exception'.format(self._max_iterations)
            )

        # Clean up
        if self._path.endswith('fenton_temp'):
            shutil.rmtree(self._path)
        self.__update_variables()
