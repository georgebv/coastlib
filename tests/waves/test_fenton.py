from coastlib.waves import FentonWave
from matplotlib.testing.compare import compare_images
import numpy as np
import os
import matplotlib.pyplot


matplotlib.pyplot.ion()


def test_fentonwave_wavelength():
    wave = FentonWave(
            wave_height=6, wave_length=57, depth=12, current_velocity=0,
            timeout=30, fourier_components=20, height_steps=20,
            points=dict(n_surface=10, n_profiles=100, n_vertical=100)
        )
    assert np.isclose(wave.wave_period, 5.998403058041828)
    assert len(wave.solution) == 19
    assert np.isclose(wave.solution['Value'].mean(), 112909.18515512646)
    assert len(wave.flowfield) == 101 * 101
    assert np.allclose(
        wave.flowfield.values.mean(axis=0),
        np.array(
            [
                9.00000000e+01, 1.42501663e+01, 6.00501292e+00, 1.31737179e-01, 6.75623329e-01,
                -1.25181383e+00, 1.26112413e+00, -2.11397381e-01, -1.32715001e-01, 2.22463085e-02
            ]
        )
    )
    assert len(wave.surface) == 11
    assert np.allclose(
        wave.surface.values.mean(axis=0),
        np.array([0, 12.97134545])
    )


def test_fentonwave_period():
    wave = FentonWave(
            wave_height=6, wave_period=6, depth=12, current_velocity=0,
            timeout=30, fourier_components=20, height_steps=20,
            points=dict(n_surface=10, n_profiles=100, n_vertical=100)
        )
    assert np.isclose(wave.wave_length, 57.0201816)
    assert len(wave.solution) == 19
    assert np.isclose(wave.solution['Value'].mean(), 112924.25756144822)
    assert len(wave.flowfield) == 101 * 101
    assert np.allclose(
        wave.flowfield.values.mean(axis=0),
        np.array(
            [
                9.00000000e+01, 1.42550495e+01, 6.00500974e+00, 1.31694748e-01, 6.75574198e-01,
                -1.25147120e+00, 1.26100839e+00, -2.11297786e-01, -1.32690684e-01, 2.22337601e-02
            ]
        )
    )
    assert len(wave.surface) == 11
    assert np.allclose(
        wave.surface.values.mean(axis=0),
        np.array([-3.22973971e-16,  1.29713455e+01])
    )


def run_fentonwave_plot(figname, **kwargs):
    wave = FentonWave(
            wave_height=6, wave_period=6, depth=12, current_velocity=0,
            timeout=30, fourier_components=20, height_steps=20,
            points=dict(n_surface=10, n_profiles=100, n_vertical=100)
        )
    fig, ax = wave.plot(**kwargs)
    figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_images')
    baseline_path = os.path.join(figure_path, f'{figname}.png')
    compare_path = os.path.join(figure_path, f'{figname}_compare.png')
    fig.savefig(compare_path, dpi=100)
    comparison = compare_images(baseline_path, compare_path, .001)
    os.remove(compare_path)
    return comparison


def test_fentonwave_plot_basic():
    assert run_fentonwave_plot(figname='fentonwave_basic') is None


def test_fentonwave_plot_vy():
    assert run_fentonwave_plot(what='vy', figname='fentonwave_vy') is None


def test_fentonwave_plot_scale_nprof():
    assert run_fentonwave_plot(what='u', scale=2, nprof=3, figname='fentonwave_scale_nprof') is None
