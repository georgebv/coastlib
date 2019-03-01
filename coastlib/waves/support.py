import coastlib.waves
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.constants


def wave_theories(wave_height, wave_period, depth, g=scipy.constants.g):
    """
    Shows where a wave with given paramters falls within the wave theory applicability chart
    by Le Mehaute per USACE CEM Part II Chap. 1 p.II-1-58

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
    """

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

    if _x > .2 or _x < .0005 or _y > .05 or _y < .00005:
        raise ValueError('Wave parameters outside of the LeMehaute figure range')

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(image, zorder=5)
    plt.plot([286, 1592], [y2pix(_y), y2pix(_y)], ls='--', c='orangered', lw=1.4, zorder=15)
    plt.plot([286, 1592], [y2pix(_y), y2pix(_y)], ls='-', c='white', lw=4, zorder=10)
    plt.plot([x2pix(_x), x2pix(_x)], [24, 1548], ls='--', c='orangered', lw=1.4, zorder=15)
    plt.plot([x2pix(_x), x2pix(_x)], [24, 1548], ls='-', c='white', lw=4, zorder=10)
    plt.scatter(x2pix(_x), y2pix(_y), edgecolors='k', facecolor='orangered', s=80, lw=2, zorder=20)
    plt.axis('off')
    plt.title('Ranges of suitability of various wave theories, Le Mehaute (1976)', x=.57)

    return fig, ax


if __name__ == '__main__':
    _fig, _ax = wave_theories(wave_height=6, wave_period=8, depth=20)
