import coastlib.waves
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.constants
import scipy.optimize

from coastlib.plotting.styles import coastlib_rc


def solve_dispersion_relation(wave_period, depth, g=scipy.constants.g):
    """
    Solves dispersion relation for given wave period and water depth.

    Parameters
    ----------
    wave_period : float
        Wave period in seconds.
    depth : float
        Water depth in meters.
    g : float, optional
        Gravity constant (m/s^2) (default=scipy.constants.g).

    Returns
    -------
    wave_length : float
        Estimated wavelength (m) for parameters entered.

    Raises
    ------
    ValueError
        if wave_period <= 0
        if depth <= 0
        if g <= 0

    Examples
    --------
    >>> solve_dispersion_relation(wave_period=6, depth=20)
    55.032374326531
    """

    for value in [wave_period, depth, g]:
        if value <= 0:
            raise ValueError(f'Invalid value passed in "{value}"')

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

    assert solution[2] == 1, f'Solution was not found. Reason: {solution[3]}'

    return 2 * np.pi / solution[0][0]


def wave_theories(wave_height, wave_period, depth, g=scipy.constants.g):
    """
    Shows where a wave with given paramters falls within the wave theory applicability chart
    by Le Mehaute per USACE CEM Part II Chap.1 p.II-1-58

    Parameters
    ----------
    wave_height : float
        Wave height in meters.
    wave_period : float
        Wave period in seconds.
    depth : float
        Water depth in meters.
    g : float, optional
        Earth gravity in m/s^2 (default=scipy.constants.g, which is ~9.81).

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axes object

    Raises
    ------
    ValueError
        if wave_height <= 0
        if wave_period <= 0
        if depth <= 0
        if g <= 0
        if parameters fall outside the figure range

    Examples
    --------
    >>> fig, ax = wave_theories(wave_height=2, wave_period=6, depth=20)
    """

    for value in [wave_height, wave_period, depth, g]:
        if value <= 0:
            raise ValueError(f'Invalid value passed in "{value}"')

    rootdir = os.path.split(str(coastlib.waves.__file__))[0]
    path = os.path.join(rootdir, 'LeMehaute.png')

    # Load and reinterpolate image to smooth pixellated regions
    image = Image.open(path)
    resampling_factor = 2
    original_size = image.size
    image = image.resize(
        (
            int(image.size[0] / resampling_factor),
            int(image.size[1] / resampling_factor)
        ),
        Image.LANCZOS
    )
    image = image.resize(
        original_size,
        Image.LANCZOS
    )

    def x2pix(x):
        return 434 + (np.log(x) / np.log(10) - np.log(0.001) / np.log(10)) * (1451 - 434) / 2

    def y2pix(y):
        return 1404 - (np.log(y) / np.log(10) - np.log(0.0001) / np.log(10)) * (1404 - 370.5) / 2

    _x = depth / (g * wave_period ** 2)
    _y = wave_height / (g * wave_period ** 2)

    if not (0.0005 <= _x <= 0.2 and 0.00005 <= _y <= 0.05):
        raise ValueError('Wave parameters outside of the LeMehaute figure range')

    with plt.rc_context(rc=coastlib_rc):
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.imshow(image, zorder=5)
        ax.plot([292, 1587], [y2pix(_y), y2pix(_y)], ls='--', c='#F85C50', lw=1.4, zorder=15)
        ax.plot([292, 1587], [y2pix(_y), y2pix(_y)], ls='-', c='#FFFFFF', lw=4, zorder=10)
        ax.plot([x2pix(_x), x2pix(_x)], [29, 1540], ls='--', c='#F85C50', lw=1.4, zorder=15)
        ax.plot([x2pix(_x), x2pix(_x)], [29, 1540], ls='-', c='#FFFFFF', lw=4, zorder=10)
        ax.scatter(x2pix(_x), y2pix(_y), edgecolors='#FFFFFF', facecolor='#F85C50', s=100, lw=1.4, zorder=20)
        ax.axis('off')
        ax.set_title('Ranges of suitability of various wave theories, Le Mehaute (1976)', x=.57, fontsize=12)
        fig.subplots_adjust(top=.96, bottom=0.01, right=1, left=0)

    return fig, ax
