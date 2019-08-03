from coastlib.waves import solve_dispersion_relation, wave_theories
from matplotlib.testing.compare import compare_images
import numpy as np
import os
import matplotlib.pyplot as plt

plt.ioff()


def test_solve_dispersion_relation():
    expected = np.array(
        [
            [6.05010492, 6.24257614, 6.24310727, 6.24310729, 6.24310729],
            [20.95030772, 30.30246101, 36.58313869, 38.44435152, 38.89764852],
            [43.69187713, 67.66809112, 92.35581695, 109.02674052, 121.20984404],
            [66.03394019, 103.46738056, 144.1020394, 173.76148476, 197.49164398],
            [88.27657155, 138.87199203, 194.72980439, 236.45276928, 270.67236449]
        ]
    )
    computed = np.empty((5, 5))
    for i, wave_period in enumerate([2, 5, 10, 15, 20]):
        for j, depth in enumerate([2, 5, 10, 15, 20]):
            computed[i][j] = solve_dispersion_relation(wave_period=wave_period, depth=depth)
    assert np.allclose(expected, computed)


def test_solve_dispersion_relation_basic_custom_g():
    expected = np.array([11.4591559, 28.63904225, 56.01973205])
    computed = np.empty(3)
    for i, g in enumerate([2, 5, 10]):
        computed[i] = solve_dispersion_relation(wave_period=6, depth=20, g=g)
    assert np.allclose(expected, computed)


def run_wave_theories(wave_height, wave_period, depth, figname, **kwargs):
    fig, ax = wave_theories(wave_height=wave_height, wave_period=wave_period, depth=depth, **kwargs)
    figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_images')
    baseline_path = os.path.join(figure_path, f'{figname}.png')
    compare_path = os.path.join(figure_path, f'{figname}_compare.png')
    fig.savefig(compare_path, dpi=100)
    comparison = compare_images(baseline_path, compare_path, .001)
    os.remove(compare_path)
    plt.close(fig)
    return comparison


def test_wave_theories_basic():
    assert run_wave_theories(wave_height=2, wave_period=6, depth=20, figname='wave_theories_basic') is None
