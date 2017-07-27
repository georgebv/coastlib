import os
import pandas as pd
import numpy as np
import subprocess
import datetime
import shutil
import time
import matplotlib.pyplot as plt
import warnings


def fData(title, H_to_d, measure, value_of_that_length, current_criterion, current_magnitude, N, height_steps):
    data = r'''{title}
{H_to_d:.20f}	H/d
{measure}	Measure of length: "Wavelength" or "Period"
{value_of_that_length:.20f} 	Value of that length: L/d or T(g/d)^1/2 respectively
{current_criterion:d}		Current criterion (1 or 2)
{current_magnitude:.20f}		Current magnitude, (dimensionless) ubar/(gd)^1/2
{N:d}		Number of Fourier components or Order of Stokes/cnoidal theory
{height_steps:d}		Number of height steps to reach H/d
FINISH
'''.format(
        title=title, H_to_d=H_to_d, measure=measure, value_of_that_length=value_of_that_length,
        current_criterion=current_criterion, current_magnitude=current_magnitude,
        N=N, height_steps=height_steps
    )
    return data


def fConvergence(maximum_number_of_iterations, criterion_for_convergence):
    convergence = r'''Control file to control convergence and output of results
{max_n_i:d}		Maximum number of iterations for each height step; 10 OK for ordinary waves, 40 for highest
{crit}	Criterion for convergence, typically 1.e-4, or 1.e-5 for highest waves
'''.format(max_n_i=maximum_number_of_iterations, crit=criterion_for_convergence)
    return convergence


def fPoints(M, ua, vert):
    points = r'''Control output for graph plotting
{M:d}	Number of points on free surface (clustered near crest)
{ua:d}	Number of velocity profiles over half a wavelength to print out
{vert:d}	Number of vertical points in each profile
'''.format(M=M, ua=ua, vert=vert)
    return points


def fInput(path, fdata, fpoints, fconvergence):
    with open(os.path.join(path, 'Data.dat'), 'w') as f:
        print(fData(**fdata), file=f)
    with open(os.path.join(path, 'Convergence.dat'), 'w') as f:
        print(fConvergence(**fconvergence), file=f)
    with open(os.path.join(path, 'Points.dat'), 'w') as f:
        print(fPoints(**fpoints), file=f)


def fRun(path, cPATH):
    fPATH = os.path.join(cPATH, 'Fourier.exe')
    curdir = os.getcwd()
    os.chdir(path=path)
    p = subprocess.Popen(fPATH, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    while p.poll() is None:
        line = p.stdout.readline()
        try:
            line = line.decode()
            if len(line) > 0:
                print(line)
            if line.startswith('# Solution summary'):
                p.kill()
        except:
            pass
    print('\n=== Process killed (as intended). Check for completion above ===')
    os.chdir(curdir)


def fParse(path, ret_val=False):
    with open(os.path.join(path, 'Solution.res'), 'r') as f:
        solution = f.readlines()
    with open(os.path.join(path, 'Surface.res'), 'r') as f:
        surface = f.readlines()
    with open(os.path.join(path, 'Flowfield.res'), 'r') as f:
        flowfield = f.readlines()

    sol = solution # TODO: parse solution
    print('\nSolution was not parsed (not yet implemented)')

    surf = [i.split(sep='\t') for i in surface][8:-1]
    surf = np.array([[float(value) for value in row] for row in surf])
    surf = pd.DataFrame(data=surf, columns=['X (d)', 'eta (d)', 'pressure check'])

    flow = ''.join(flowfield).split(sep='\n')[14:]
    headers = []
    for i in range(len(flow)):
        if flow[i][0:5] == '# X/d':
            headers += [i]
    fields = []
    for i in range(len(headers) - 1):
        fields += [list(filter(None, flow[headers[i]:headers[i+1]]))]
    fields += [list(filter(None, flow[headers[-1]:]))]
    fields = [[list(filter(None, row.split('\t'))) for row in field] for field in fields]
    headers = [field[0] for field in fields]
    fields = [[[float(item) for item in row] for row in field[1:]] for field in fields]
    flow = [pd.DataFrame(data=field, columns=[
        'Y (d)', 'u (sqrt(gd))', 'v (sqrt(gd))', 'dphi/dt (gd)', 'du/dt (g)', 'dv/dt (g)', 'du/dx (sqrt(g/d))',
        'du/dy (sqrt(gd))', 'Bernoully check (gd)'
    ]) for field in fields]
    headers = [list(filter(None, header[0].split(sep=' '))) for header in headers]
    headers = [[float(header[3].split(sep=',')[0]), float(header[6].split(sep='°')[0])] for header in headers]
    for i in range(len(flow)):
        flow[i]['X (d)'] = [headers[i][0]] * len(flow[i])
        flow[i]['Phase (deg)'] = [headers[i][1]] * len(flow[i])
    flow = pd.concat(flow)

    surf.to_pickle(os.path.join(path, 'surface.pyc'))
    flow.to_pickle(os.path.join(path, 'flowfield.pyc'))

    print('\nSaved output to\n    "{slout}"\n    "{sout}"\n    "{fout}"'.format(
        slout=os.path.join(path, 'solution.pyc'), sout=os.path.join(path, 'surface.pyc'),
        fout=os.path.join(path, 'flowfield.pyc')
    ))

    if ret_val:
        return sol, surf, flow


def fourier(path, fdata, cPATH, ret_val=False, **kwargs):
    """
    !!!!!!!!! USE FentonWave class - its better !!!!!!!!!

    Processes a wave using the Fenton Fourier method http://johndfenton.com/Steady-waves/Fourier.html

    Parameters
    ----------
    path : str
        Path to work folder
    fdata : dict
        Parameters of a wave
        {
            'title' : 'Test wave',
            'H_to_d' : 0.5,
            'measure' : Wavelength,
            'value_of_that_length' : 10.0,
            'current_criterion' : 1,
            'current_magnitude' : 0.0,
            'N' : 20,
            'height_steps' : 1
        }
    cPATH : str
        Path to "Fourier.exe"
    ret_val : bool
        If Ture, returns tuple of outputs (solution, surface, flowfield)
    fpoints : dict
        Optional. Determines number of points for output
    fconvergence : dict
        Optional. Determines convergence parameters

    Returns
    -------
    Saves (solution, surface, flowfield) pickles of pandas dataframe in <path>. Optionally returns a tuple
    with (solution, surface, flowfield).

    """

    fpoints = kwargs.pop(
        'fpoints',
        {
            'M': 100,
            'ua': 50,
            'vert': 100
        }
    )
    fconvergence = kwargs.pop(
        'fconvergence',
        {
            'maximum_number_of_iterations': 20,
            'criterion_for_convergence': '1.e-4'
        }
    )
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    if not os.path.exists(path):
        os.makedirs(path)

    fInput(path=path, fdata=fdata, fpoints=fpoints, fconvergence=fconvergence)
    fRun(path=path, cPATH=cPATH)
    out = fParse(path=path, ret_val=ret_val)

    if ret_val:
        return out


class FentonWave:
    '''
    data : dict
        Data.dat input
    cPATH : str
        Absolute path to folder with Fourier.exe (can handle both path to folder or to .exe itself)
    '''

    def __init__(self, data, cPATH, path=None, convergence=None, points=None, write_output=False):

        self.data, self.cPATH = data, cPATH

        if self.cPATH.endswith('Fourier.exe'):
            self.fPATH = self.cPATH
        else:
            self.fPATH = os.path.join(self.cPATH, 'Fourier.exe')

        if not isinstance(self.data, dict):
            raise ValueError('data should be a dictionary')
            pass
            # TODO - parse other forms of inputs

        if not path:
            self.path = os.path.join(
                os.environ['ALLUSERSPROFILE'],
                datetime.datetime.now().strftime('%y%m%d') + 'ftmp')
        else:
            self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not convergence:
            self.convergence = {
                'maximum_number_of_iterations': 20,
                'criterion_for_convergence': '1.e-4'
            }  # recommended convergence criteria
        else:
            self.convergence = convergence

        if not points:
            self.points = {
                'M': 100,
                'ua': 100,
                'vert': 100
            }  # 100 points per length/height by default
        else:
            self.points = points

        sucess = False
        for i in range(20):
            try:
                self.__write_inputs()
                self.__run()
                self.__parse(write_output=write_output)
                sucess = True
                break
            except:
                warnings.warn('Failure after {} iterations. Repeating'.format(i+1))
        if not sucess:
            raise RuntimeError('Wave was not resolved after 20 iterations. Time to debug :(')

        if self.path.endswith('ftmp'):
            shutil.rmtree(self.path)
            self.path = None

    def __write_inputs(self):
        with open(os.path.join(self.path, 'Data.dat'), 'w') as f:
            f.write(fData(**self.data))
        with open(os.path.join(self.path, 'Convergence.dat'), 'w') as f:
            f.write(fConvergence(**self.convergence))
        with open(os.path.join(self.path, 'Points.dat'), 'w') as f:
            f.write(fPoints(**self.points))

    def __run(self):

        self.log = []  # Fourier.exe logs (stdout)
        curdir = os.getcwd()
        os.chdir(path=self.path)
        time.sleep(0.1)  # TODO - is this useful? (code still unstable for long loops)
        p = subprocess.Popen(self.fPATH, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        while p.poll() is None:
            line = p.stdout.readline()
            try:
                line = line.decode()
                if len(line) > 0:
                    self.log += [line]
                if line.startswith('# Solution summary'):
                    p.kill()
            except:
                pass

        time.sleep(0.1)  # TODO - is this useful?
        os.chdir(curdir)

    def __parse(self, write_output=False):

        # Open files generated by Fourier.exe
        with open(os.path.join(self.path, 'Solution.res'), 'r') as f:
            solution = f.readlines()
        with open(os.path.join(self.path, 'Surface.res'), 'r') as f:
            surface = f.readlines()
        with open(os.path.join(self.path, 'Flowfield.res'), 'r') as f:
            flowfield = f.readlines()

        # Parse Solution.res
        self.solution = solution  # TODO: parse solution

        # Parse Surface.res
        surf = [i.split(sep='\t') for i in surface][8:-1]
        surf = np.array([[float(value) for value in row] for row in surf])
        self.surface = pd.DataFrame(data=surf, columns=['X (d)', 'eta (d)', 'pressure check'])

        # Parse Flowfield.res
        flow = ''.join(flowfield).split(sep='\n')[14:]
        headers = []
        for i in range(len(flow)):
            if flow[i][0:5] == '# X/d':
                headers += [i]
        fields = []
        for i in range(len(headers) - 1):
            fields += [list(filter(None, flow[headers[i]:headers[i + 1]]))]
        fields += [list(filter(None, flow[headers[-1]:]))]
        fields = [[list(filter(None, row.split('\t'))) for row in field] for field in fields]
        headers = [field[0] for field in fields]
        fields = [[[float(item) for item in row] for row in field[1:]] for field in fields]
        flow = [pd.DataFrame(data=field, columns=[
            'Y (d)', 'u (sqrt(gd))', 'v (sqrt(gd))', 'dphi/dt (gd)', 'du/dt (g)', 'dv/dt (g)', 'du/dx (sqrt(g/d))',
            'du/dy (sqrt(gd))', 'Bernoully check (gd)'
        ]) for field in fields]
        headers = [list(filter(None, header[0].split(sep=' '))) for header in headers]
        headers = [[float(header[3].split(sep=',')[0]), float(header[6].split(sep='°')[0])] for header in headers]
        for i in range(len(flow)):
            flow[i]['X (d)'] = [headers[i][0]] * len(flow[i])
            flow[i]['Phase (deg)'] = [headers[i][1]] * len(flow[i])
        self.flowfield = pd.concat(flow)

        # Write output to self.path
        if write_output:
            # Surface
            self.surface.to_pickle(os.path.join(self.path, 'surface.pyc'))
            self.surface.to_csv(os.path.join(self.path, 'surface.csv'))
            # Flowfield
            self.flowfield.to_pickle(os.path.join(self.path, 'flowfield.pyc'))
            self.flowfield.to_csv(os.path.join(self.path, 'flowfield.csv'))
            # Echo completion
            print('\nSaved output to\n    "{slout}"\n    "{sout}"\n    "{fout}"'.format(
                slout=os.path.join(self.path, 'solution.pyc'), sout=os.path.join(self.path, 'surface.pyc'),
                fout=os.path.join(self.path, 'flowfield.pyc')
            ))

    def report(self):

        # Echo summary
        print(''.join(self.solution[:10]))

        # Parameters
        parameters = self.solution[14:33]
        parameters = [line.split('\t') for line in parameters]
        rows = [line[0] for line in parameters]
        values = [[float(line[1]), float(line[2])] for line in parameters]
        frame = pd.DataFrame(data=values, index=rows)
        frame.columns = pd.MultiIndex.from_tuples([
            ('Solution non-dimensionalised by', 'g & wavenumber'),
            ('Solution non-dimensionalised by', 'g & mean depth')
        ])
        return frame

    def plot(self, what, scale=0.5, reduction=0, profiles=4):

        try:
            what = {
                'u': 'u (sqrt(gd))',
                'du/dt': 'du/dt (g)',
                'v': 'v (sqrt(gd))',
                'dv/dt': 'dv/dt (g)'
            }.pop(what)
        except:
            raise ValueError('Unrecognized "{}" value passed in .plot'.format(what))

        with plt.style.context('bmh'):
            plt.figure(figsize=(24, 8))
            plt.plot(self.surface['X (d)'].values, self.surface['eta (d)'].values, lw=2, color='royalblue')
            plt.ylim([-0.1, 1.1])
            plt.xlim([self.surface['X (d)'].values.min()*1.1, self.surface['X (d)'].values.max()*1.1])
            plt.plot([self.surface['X (d)'].values.min(), self.surface['X (d)'].values.max()],
                     [0, 0], color='saddlebrown', lw=2, ls='--')

            X_flow = np.unique(self.flowfield['X (d)'].values)
            print(self.flowfield.columns)

            for i in np.arange(0, len(X_flow), int(np.round(len(X_flow)/profiles))):
                flow_loc = self.flowfield[self.flowfield['X (d)'] == X_flow[i]]
                plt.plot(
                    [X_flow[i], X_flow[i]],
                    [flow_loc['Y (d)'].values.min(), flow_loc['Y (d)'].values.max()],
                    color='k', lw=1
                )  # vertical line
                plt.plot(
                    [X_flow[i], X_flow[i] + (flow_loc[what].values[0] * scale - reduction)],
                    [flow_loc['Y (d)'].values.min(), flow_loc['Y (d)'].values.min()],
                    color='orangered', lw=1
                )  # horizontal line bottom
                plt.plot(
                    [X_flow[i], X_flow[i] + (flow_loc[what].values[-1] * scale - reduction)],
                    [flow_loc['Y (d)'].values.max(), flow_loc['Y (d)'].values.max()],
                    color='orangered', lw=1
                )  # horizontal line top
                plt.plot(
                    flow_loc[what].values * scale + X_flow[i] - reduction,
                    flow_loc['Y (d)'].values,
                    color='orangered', lw=1
                )  # profile
            plt.show()
