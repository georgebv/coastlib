from coastlib.plotting.rose import get_rose_parameters, rose_plot
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.testing.compare import compare_images


plt.ioff()


data_folder = os.sep.join([*os.path.realpath(__file__).split(os.sep)[:-1], 'data'])


@pytest.fixture
def wind_data():
    data = pd.read_pickle(os.path.join(data_folder, 'wind.pyc'))
    return data[~np.isnan(data['s']) & ~np.isnan(data['d'])]


def test_get_rose_parameters_basic(wind_data):
    theta, radii, bottoms, colors, calm_percentage, value_bin_boundaries = get_rose_parameters(
        values=wind_data['s'].values, directions=wind_data['d'].values,
        value_bin_boundaries=np.arange(6, 18, 3), calm_value=3, n_dir_bins=6,
        center_on_north=True, calm_size=1
    )

    assert np.isclose(calm_percentage, 7.009215949613301)

    assert np.allclose(value_bin_boundaries, np.array([3., 6., 9., 12., 15., np.inf]))

    assert theta.shape == (5, 6)
    assert np.isclose(theta[0][0], 0)
    assert np.isclose(theta[0][-1], 5.235987755982988)

    assert radii.shape == (5, 6)
    assert np.isclose(radii[0][0], 2.3630343166767007)
    assert np.isclose(radii[0][-1], 2.4290279257166674)
    assert np.isclose(radii.mean(), 3.099692801679556)

    assert np.isclose(radii.sum() + calm_percentage, 100)
    assert np.isclose(bottoms[-1].sum() + radii[-1].sum() - bottoms[0].sum() + calm_percentage, 100)

    assert np.isclose(colors.sum(), 11.528529)
    assert np.isclose(colors.mean(), 0.57642645)


def test_get_rose_parameters_custom(wind_data):
    theta, radii, bottoms, colors, calm_percentage, value_bin_boundaries = get_rose_parameters(
        values=wind_data['s'].values, directions=wind_data['d'].values,
        value_bin_boundaries=np.arange(6, 18, 3), calm_value=None, n_dir_bins=16,
        center_on_north=False, calm_size=None
    )

    assert np.isclose(calm_percentage, 20.12226184411615)

    assert np.allclose(value_bin_boundaries, np.array([6., 9., 12., 15., np.inf]))

    assert theta.shape == (4, 16)
    assert np.isclose(theta[0][0], 0.19634954084936207)
    assert np.isclose(theta[0][-1], 6.086835766330224)

    assert radii.shape == (4, 16)
    assert np.isclose(radii[0][0], 0.7803454823322373)
    assert np.isclose(radii[0][-1], 1.5942666604918259)
    assert np.isclose(radii.mean(), 1.2480896586856853)

    assert np.isclose(radii.sum() + calm_percentage, 100)
    assert np.isclose(bottoms[-1].sum() + radii[-1].sum() - bottoms[0].sum() + calm_percentage, 100)

    assert np.isclose(colors.sum(), 9.198019)
    assert np.isclose(colors.mean(), 0.5748761875)


def run_rose_plot(data, figname, **kwargs):
    fig, ax = rose_plot(values=data['s'].values, directions=data['d'].values, **kwargs)
    figure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_images')
    baseline_path = os.path.join(figure_path, f'rose_plot_{figname}.png')
    compare_path = os.path.join(figure_path, f'rose_plot_{figname}_compare.png')
    fig.savefig(compare_path, dpi=100)
    comparison = compare_images(baseline_path, compare_path, .001)
    os.remove(compare_path)
    plt.close(fig)
    return comparison


def test_rose_plot_basic(wind_data):
    run_rose_plot(
        data=wind_data, figname='basic',
        value_bin_boundaries=np.arange(6, 18, 3), n_dir_bins=12, cmap=plt.get_cmap('viridis'), rose_type='bar',
        fig=None, ax=None, center_on_north=True, calm_size=1, calm_value=3, title='Rose Plot', value_name=None,
        rwidths=None
    )


def test_rose_plot_contour(wind_data):
    run_rose_plot(
        data=wind_data, figname='contour',
        value_bin_boundaries=np.arange(6, 18, 3), n_dir_bins=12, cmap=plt.get_cmap('magma'), rose_type='contour',
        fig=None, ax=None, center_on_north=True, calm_size=1, calm_value=3, title='Rose Plot Contourf',
        value_name='Wind Speed', rwidths=None
    )


def test_rose_plot_contourf(wind_data):
    run_rose_plot(
        data=wind_data, figname='contourf',
        value_bin_boundaries=np.arange(6, 18, 3), n_dir_bins=12, cmap=plt.get_cmap('magma'), rose_type='contourf',
        fig=None, ax=None, center_on_north=True, calm_size=1, calm_value=3, title='Rose Plot Contourf',
        value_name='Wind Speed', rwidths=None
    )


def test_rose_plot_bargeom(wind_data):
    run_rose_plot(
        data=wind_data, figname='bargeom',
        value_bin_boundaries=np.arange(6, 18, 3), n_dir_bins=12, cmap=plt.get_cmap('Blues'), rose_type='bar',
        fig=None, ax=None, center_on_north=True, calm_size=1, calm_value=3, title='Rose Plot Geomspace',
        value_name='Wind Speed', rwidths=None, geomspace=True
    )
