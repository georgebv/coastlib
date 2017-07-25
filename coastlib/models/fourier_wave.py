import os
import pandas as pd
import numpy as np
import subprocess


def fData(title, H_to_d, measure, value_of_that_length, current_criterion, current_magnitude, N, height_steps):
    data = r'''{title}
{H_to_d:.2f}	H/d
{measure}	Measure of length: "Wavelength" or "Period"
{value_of_that_length:.2f} 	Value of that length: L/d or T(g/d)^1/2 respectively
{current_criterion:d}		Current criterion (1 or 2)
{current_magnitude:.2f}		Current magnitude, (dimensionless) ubar/(gd)^1/2
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
            if line.startswith('# Current'):
                p.kill()
                print('\nProcess killed (as intended). Check for completion above.')
        except:
            pass
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
