# coastlib, a coastal engineering Python library
# Copyright (C), 2019 Georgii Bocharov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
from .styles import coastlib_rc


def get_rose_parameters(
        values, directions, value_bin_boundaries, n_dir_bins, cmap=plt.get_cmap('viridis'),
        center_on_north=True, calm_size=1, calm_value=None
):
    """
    Calculates rose plot parameters.

    Parameters
    ----------
    values : array_like
        Array with values.
    directions : array_like
        Array with directions.
    value_bin_boundaries : array_like
        Sorted array with boundaries of values bins.
    n_dir_bins : int
        Number of direction bins.
    cmap : matplotlib.colors.ListedColormap
        Matplotlib colormap (default=plt.get_cmap('viridis')).
    center_on_north : bool, optional
        If True, first bin has its center on North. If False, first bin has its start on North (default=True).
    calm_value : float, optional
        Value defining calm region. All values smaller than this value are placed into the calm region (default=None).
        Must not be larger than smallest value in value_bin_boundaries.
        If None, takes the smallest value from value_bin_boundaries.

    Returns
    -------
    theta : array_like
        Bar centers (x coordinates).
    radii : array_like
        Bar heights.
    bottoms : array_like
        Bar bottom y coordinates.
    colors : array_like
        Colors for each concentric row of bars.
    calm_percentage : float
        Percentage of calm values.
    value_bin_boundaries : array_like
        Sorted array with boundaries of values bins.
    """

    assert np.all(~np.isnan(values)), '<values> array must not contain nan values'
    assert np.all(~np.isnan(directions)), '<directions> array must not contain nan values'

    assert len(values) == len(directions), f'number of values must be the same as number of directions, ' \
        f'{len(values)} != {len(directions)}'

    value_bin_boundaries = np.sort(value_bin_boundaries)

    # Make sure <calm_value> is accounted for in value_bin_boundaries and calculate number of calms
    if calm_value is None:
        calm_value = value_bin_boundaries[0]
        calm_count = np.sum(values < calm_value)
    elif calm_value <= value_bin_boundaries[0]:
        calm_count = np.sum(values < calm_value)
        if calm_value != value_bin_boundaries[0]:
            value_bin_boundaries = np.append(calm_value, value_bin_boundaries)
    else:
        raise ValueError(
            f'calm_value {calm_value:.2f} must not be larger than smallest '
            f'value in value_bin_boundaries {value_bin_boundaries[0]:.2f}'
        )

    # Make sure values larger than highest values bin are accounted for
    if np.inf not in value_bin_boundaries:
        value_bin_boundaries = np.append(value_bin_boundaries, np.inf)

    # Calculate thetas - x-coordinates (centers) of bars on the polar rose plot
    if center_on_north:
        t = np.linspace(0, 2 * np.pi, n_dir_bins, endpoint=False)
    else:
        t = np.linspace(0, 2 * np.pi, n_dir_bins, endpoint=False) + np.pi / n_dir_bins
    theta = np.array([t for _ in range(len(value_bin_boundaries) - 1)])

    # Get direction bin boundaries
    angles = np.rad2deg(t)
    delta_angle = 180 / n_dir_bins
    direction_bin_boundaries = np.round(np.append(angles - delta_angle, angles[-1] + delta_angle), 10)
    # Shifts all bins and directions for the case of <center_on_north>
    if center_on_north:
        direction_bin_boundaries = [_bin + delta_angle for _bin in direction_bin_boundaries]
        fixed_directions = (directions + delta_angle) % 360
    else:
        fixed_directions = directions

    # Get only values that are outside the calm region
    mask = values >= calm_value
    fixed_values = values[mask]
    fixed_directions = fixed_directions[mask]

    # Get counts for each bin
    binned, _, _ = np.histogram2d(fixed_values, fixed_directions, [value_bin_boundaries, direction_bin_boundaries])

    # Get coordinates of bar (bar on a rose plot) bottoms as counts (calm region not taken into account)
    bottom_counts = np.transpose(
        [
            np.append(
                0, np.cumsum(bt[:-1])
            ) for bt in binned.T
        ]
    )

    # Make sure all values are counted and convert counts to percentages
    assert calm_count + np.sum(binned) == len(values), 'Number of binned values is not equal to total number of values'
    assert np.isclose(
        bottom_counts[-1].sum() + binned[-1].sum(),
        len(fixed_values)
    ), 'Number of binned values in bar bottoms is not equal to total number of adjusted values'
    calm_percentage = calm_count / len(values) * 100
    radii = binned / len(fixed_values) * 100
    bottoms = bottom_counts / len(fixed_values) * 100

    # Get colors for each value bin
    colors = np.array([cmap(i) for i in np.linspace(0.0, 1.0, radii.shape[0])])

    return theta, radii, bottoms, colors, calm_percentage, value_bin_boundaries


def rose_plot(
        values, directions, value_bin_boundaries, n_dir_bins=12, cmap=plt.get_cmap('viridis'), rose_type='bar',
        fig=None, ax=None, center_on_north=True, calm_size=None, calm_value=None, title='Rose Plot', value_name=None,
        rwidths=None, geomspace=False, **kwargs
):
    """
    Generates a rose plot for given values and directions.

    Parameters
    ----------
    values : array_like
        Array with values.
    directions : array_like
        Array with directions.
    value_bin_boundaries : array_like
        Sorted array with boundaries of values bins.
    n_dir_bins : int, optional
        Number of direction bins (default=12).
    cmap : matplotlib.colors.ListedColormap, optional
        Matplotlib colormap (default=plt.get_cmap('viridis')).
    rose_type : str, optional
        Rose type. Can be 'bar' (classic windrose), 'contour', 'contourf'. (default='bar')
    fig : matplotlib figure object, optional
        Matplotlib Figure object. Must be passed if custom ax is passed.
        If None, creates a new figure (default=None)
    ax : matplotlib axes object, optional
        Matplotlib Axes object within an existing Figure. Must have polar projection.
        If None, creates a new figure (default=None)
    center_on_north : bool, optional
        If True, first bin has its center on North. If False, first bin has its start on North (default=True).
    calm_size : float, optional
        Size of calm region in percent (defaul=None).
    calm_value : float, optional
        Value defining calm region. All values smaller than this value are placed into the calm region (default=None).
        Must not be larger than smallest value in value_bin_boundaries.
        If None, takes the smallest value from value_bin_boundaries.
    title : str, optional
        Rose plot title (default='Rose Plot').
    value_name : str, optional
        Name of value to display in legend (default=None).
    rwidths : array_like or float, optional
        Widths of bars in each row (one value per row). If None, geomspace from 0.1 to 0.95 (default=None).
        If scalar value is passed (float), draws a regular rose plot with constant widths.
    geomspace : bool, optional
        If True, scales rwidths in geometric progression (default=False).
    bar_props : dict, optional
        Dictionary with keyword arguments passed to ax.bar object.
        Default=dict(edgecolor='k', linewidth=.3, zorder=-15)
    contour_props : dict, optional
        Dictionary with keyword arguments passed to ax.plot object.
        Default=dict(lw=2, ls='-', zorder=-10)
    contourf_props : dict, optional
        Dictionary with keyword arguments passed to ax.fill_between object.
        Default=dict(zorder=-15)
    legend : bool, optional
        If True, plots legend (default=True).

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axes object
    """

    bar_props = kwargs.pop(
        'bar_props',
        dict(
            edgecolor='#454545',
            linewidth=.3,
            zorder=-15
        )
    )
    contour_props = kwargs.pop(
        'contour_props',
        dict(
            lw=2,
            ls='-',
            zorder=-10
        )
    )
    contourf_props = kwargs.pop(
        'contourf_props',
        dict(
            zorder=-15
        )
    )
    legend = kwargs.pop('legend', True)
    assert len(kwargs) == 0, f'unrecognized arguments passed in: {", ".join(kwargs.keys())}'

    with plt.rc_context(rc=coastlib_rc):
        if fig is None and ax is None:
            fig_created = True
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        elif fig is not None and ax is not None:
            fig_created = False
            if not hasattr(ax, 'set_theta_zero_location'):
                raise ValueError('passed axes must have polar projection')
        else:
            raise ValueError('both fig and ax must be passed')

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetagrids(
            angles=np.arange(0, 360, 45), labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
            size=12
        )

        theta, radii, bottoms, colors, p_calms, value_bin_boundaries = get_rose_parameters(
            values=values, directions=directions,
            value_bin_boundaries=value_bin_boundaries, n_dir_bins=n_dir_bins, cmap=cmap,
            center_on_north=center_on_north, calm_value=calm_value
        )

        if calm_size is None:
            calm_size = p_calms

        if rwidths is None:
            if geomspace:
                rwidths = np.geomspace(.1, .95, len(theta))
            else:
                rwidths = [0.95] * len(theta)
        elif np.isscalar(rwidths):
            rwidths = [rwidths] * len(theta)

        # Generate bar labels for legend
        if value_name is None:
            value_name = 'Value'
        bar_labels = [
            f'{value_bin_boundaries[i]:.2f} ≤ {value_name} < {value_bin_boundaries[i + 1]:.2f}'
            for i in range(0, len(value_bin_boundaries) - 2)
        ]
        bar_labels.append(f'{value_name} ≥ {value_bin_boundaries[-2]:.2f}')

        # Draw calm region and add to legend
        ax.set_rorigin(-calm_size)
        if p_calms > 0:
            # ax.bar(
            #     0, 0, 2*np.pi, 0, color='None',
            #     label=f'{p_calms:.2f}% Calms ({value_bin_boundaries[0]:.2f})'
            # )
            ax.text(
                0, -calm_size, f'{p_calms:.1f}%',
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, color='#454545', zorder=10
            )

        # Draw rose
        if rose_type == 'bar':
            for i in range(len(theta)):
                ax.bar(
                    theta[i], radii[i], width=(2 * np.pi / n_dir_bins) * rwidths[i], bottom=bottoms[i],
                    color=colors[i], label=bar_labels[i], **bar_props
                )
        elif rose_type == 'contour':
            for i in range(len(theta)):
                # Append 0'th element to obtain closed lines
                x = np.append(theta[i], theta[i][0])
                y = bottoms[i] + radii[i]
                y = np.append(y, y[0])
                ax.plot(
                    x, y,
                    color=colors[i], label=bar_labels[i], **contour_props
                )
        elif rose_type == 'contourf':
            for i in range(len(theta)):
                # Append 0'th element to obtain closed lines
                x = np.append(theta[i], theta[i][0])
                y = bottoms[i] + radii[i]
                y = np.append(y, y[0])
                ax.plot(
                    x, y,
                    color='#454545', lw=.5, ls='--', zorder=-10
                )
                ax.fill_between(
                    x, np.append(bottoms[i], bottoms[i][0]), y,
                    color=colors[i], label=bar_labels[i],  **contourf_props
                )
        else:
            raise ValueError(f'rose_type {rose_type} not recognized')

        # Draw legend
        if legend:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.1, 1), title=f'{value_name}', fontsize=10,
                edgecolor='#ffffff', title_fontsize=12
            )
        ax.set_title(title, fontsize=16)
        if fig_created:
            fig.tight_layout()

        # Label y-axis (percentage)
        fig.canvas.draw()
        labels = []
        for i, item in enumerate(ax.get_yticklabels()):
            t = float(item.get_text())
            # If there are more than 6 labels, leave only odd labels
            if len(ax.get_yticklabels()) > 6:
                if i % 2 != 0:
                    labels.append('')
                else:
                    labels.append(f'{t:.1f} %')
            else:
                labels.append(f'{t:.1f} %')
        ax.set_yticklabels(
            labels,
            fontdict=dict(
                path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=4, foreground='white')
                ], fontsize=10
            )
        )

        ax.grid(which='major', color='#454545', ls='--', lw=.5)

        return fig, ax


if __name__ == '__main__':
    import os
    import pandas as pd
    source = 'RR_wind'
    df = pd.read_csv(
        os.path.join(os.getcwd(), f'test data\\{source}.csv'),
        index_col=0, parse_dates=True
    )

    with plt.style.context('bmh'):
        _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(12, 4), subplot_kw=dict(projection='polar'))
        rose(
            values=df['Spd'].values, directions=df['Dir'].values, value_bin_boundaries=np.arange(3, 10, 1), n_dir_bins=16,
            ax=_ax1, rose_type='bar', fig=_fig, legend=False,
            cmap=plt.get_cmap('viridis'), center_on_north=True, calm_size=1,
            calm_value=2, title='', value_name='Wind Speed (ft/s)'
        )
        rose(
            values=df['Spd'].values, directions=df['Dir'].values, value_bin_boundaries=np.arange(3, 10, 1), n_dir_bins=16,
            ax=_ax2, rose_type='contour', fig=_fig, legend=False,
            cmap=plt.get_cmap('viridis'), center_on_north=True, calm_size=1,
            calm_value=2, title='', value_name='Wind Speed (ft/s)'
        )
        rose(
            values=df['Spd'].values, directions=df['Dir'].values, value_bin_boundaries=np.arange(3, 10, 1), n_dir_bins=16,
            ax=_ax3, rose_type='contourf', fig=_fig, legend=False,
            cmap=plt.get_cmap('viridis'), center_on_north=True, calm_size=1,
            calm_value=2, title='', value_name='Wind Speed (ft/s)'
        )

    _fig, _ax = rose(
        values=df['Spd'].values, directions=df['Dir'].values, value_bin_boundaries=np.arange(3, 10, 1),
        n_dir_bins=16, cmap=plt.get_cmap('viridis'), center_on_north=True, calm_size=1, calm_value=2,
        title='Wind Rose', value_name='Wind Speed (ft/s)'
    )
