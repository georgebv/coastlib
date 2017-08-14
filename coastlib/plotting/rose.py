import warnings

import matplotlib.pyplot as plt
import numpy as np


def __get_calms(values, calm_region_magnitude):
    """
    Calculates percentage of events below specified threshold. Used to get the size of empty center circle.

    :param calm_region_magnitude:
    :return: percentage of calms
    """

    return sum(values < calm_region_magnitude) / len(values) * 100


def __get_theta(number_of_direction_bins, number_of_value_bins, center_on_north):
    """
    Calculates 2D array with #rows=number_of_value_bins; #items per row=number_of_direction_bins.
    Each value in the array represents angle in radians of a center of each bin. Repeated for each value bin.

    :param number_of_direction_bins:
    :param number_of_value_bins:
    :param center_on_north:
    :return: array of angles in radians of centers of direction bins
    """

    if center_on_north:
        theta = np.linspace(0, 2 * np.pi, number_of_direction_bins, endpoint=False)
    else:
        theta = np.linspace(0, 2 * np.pi, number_of_direction_bins, endpoint=False) + np.pi / number_of_direction_bins

    return np.array([theta for _ in range(number_of_value_bins+1)])


def __get_radii(
        value_bin_boundaries, theta, values, directions,
        number_of_direction_bins
):
    """
    Calculates percentages of each absolute bin (by direction + by value). Used to get height of each bar.

    :param value_bin_boundaries:
    :param theta:
    :param values:
    :param directions:
    :param number_of_value_bins:
    :param number_of_direction_bins:
    :return: 2D array with percentages of each absolute bin (aka widths of the bars)
    """

    angles = np.rad2deg(theta[0])
    dangle = 180 / number_of_direction_bins
    d_bins = [(angle - dangle) for angle in angles] + [angles[-1] + dangle]

    # Shifts all bins and directions for the case of <center_on_north>
    if d_bins[0] < 0:
        d_bins = [_bin + dangle for _bin in d_bins]
        for i, direction in enumerate(directions):
            if direction + dangle > 360:
                directions[i] = direction + dangle - 360
            else:
                directions[i] = direction + dangle
    v_bins = np.append(value_bin_boundaries, np.inf)
    binned = np.histogram2d(values, directions, [v_bins, d_bins])

    return binned[0] / len(values) * 100


def __get_colors(number_of_value_bins, colormap):
    """
    Get colors for each value bins.

    :param number_of_value_bins:
    :param colormap:
    :return: array of tuples with color for each value bin
    """

    return [colormap(i) for i in np.linspace(0.0, 1.0, number_of_value_bins+1)]


def __get_bottoms(radii, percentage_of_calms):
    """
    Calculates cooridanes of bottom for each bar.
    All bars of the first value bin start from percentage_of_calms.
    All subsequent bars start from ends of previous bars.

    :param radii: numpy.ndarray
    :param percentage_of_calms:
    :return: 2D array with distances from 0 to start of each bar
    """

    bottoms = np.zeros(shape=np.shape(radii))
    bottoms[0] = [percentage_of_calms] * len(bottoms[0])
    for i in range(1, len(bottoms)):
        for j in range(len(bottoms[i])):
            bottoms[i][j] = bottoms[i-1][j] + radii[i-1][j]  # this is the bottom of each bar
    return bottoms


def rose_plot(
        values, directions, direction_bins=16, **kwargs
):
    """
    Mandatory inputs
    ================
    values : numpy.ndarray
        1D array with values (e.g. wave heights or wind speeds). takes values in [0;inf)
    directions : numpy.ndarray
        1D array with directions (same length as <values>). takes directions in [0;360]

    Optional inputs
    ===============
    value_bins : numpy.ndarray (default=split into 10 quantile ranges)
        1D array with value bin boundaries (e.g. [0, 1,... n] will return [0;1)...[n;inf) ),
        unless <calm_region> is specified
    direction_bins : int (default=16)
        number of direction bins (results in a bin size 360/<direction_bins>)
    calm_region : float (default=-np.inf)
        threshold below which values are dicarded (i.e. noise)
    center_on_north : bool (default=False)
        if True, shifts direction bins so that they start from North (0 degrees)
    notch : float (default=0.95)
        magnitude of notch between bars (notch should be in (0;1] - if 1, no notches are drawn)
    colormap : matplotlib colormap object (use plt.get_cmap['name'])
        colormap to visualise value bins
    value_name : str (default='Value')
        name of the value on the figure legend (e.g. 'Hs' or 'Wind Speed')
    alpha : float (default=0.8)
        bars' transparency (alpha should be in [0;1] with 0 invisible, 1 fully visible)
    bar_props : dict
        dictionary with properties of ax.bar object (carfully read matplotlib tutorial before changing!!!)
    fig_title : str (default='Rose Plot')
        title of the figure
    unit_name : str (default=None)
        name of the <values> unit for the plot
    save_path : str (default=None)
        if save path is give, saves figure to this path (e.g. "C:\path\to\file\image.png")
    min_ticks : int (default=4)
        minimum number of frequency radii (increase if not enough)

    Output
    ======
    """

    calm_region_magnitude = kwargs.pop('calm_region', -np.inf)
    value_bins = kwargs.pop(
        'value_bins',
        np.unique(
            [np.percentile(values[values >= calm_region_magnitude], _p) for _p in np.arange(0, 100, 10)]
        )
    )
    center_on_north = kwargs.pop('center_on_north', False)
    notch = kwargs.pop('notch', 0.95)
    colormap = kwargs.pop('colormap', plt.get_cmap('jet'))
    value_name = kwargs.pop('value_name', 'Value')
    alpha = kwargs.pop('alpha', 0.8)
    bar_props = kwargs.pop(
        'bar_props',
        {
            'edgecolor': 'k',
            'linewidth': 0.3,
            'antialiased': True
        }
    )
    fig_title = kwargs.pop('fig_title', 'Rose Plot')
    unit_name = kwargs.pop('unit_name', None)
    save_path = kwargs.pop('save_path', None)
    min_ticks = kwargs.pop('min_ticks', 4)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    # Ensure data is numpy array
    if (not isinstance(values, np.ndarray)) or (not isinstance(directions, np.ndarray))\
            or not isinstance(value_bins, np.ndarray):
        try:
            values = np.array(values)
            directions = np.array(directions)
            value_bins = np.array(value_bins)
        except Exception as _e:
            raise ValueError('Input values should be arrays.'
                             '{}'.format(_e))

    # Update value_bins array to include calm_region
    value_bins = value_bins.astype(float)
    if calm_region_magnitude != 0:
        while True:
            if value_bins[0] < calm_region_magnitude:
                value_bins = value_bins[1:]
            elif value_bins[0] > calm_region_magnitude:
                value_bins = np.insert(value_bins, 0, calm_region_magnitude)
            else:
                break
    elif value_bins[0] > 0:
        value_bins = np.insert(value_bins, 0, [0])
    number_of_value_bins = len(value_bins) - 1

    # Calculate percentage of calms
    calms = __get_calms(values=values, calm_region_magnitude=calm_region_magnitude)

    # Get an array of angluar coordinates
    theta = __get_theta(
        number_of_direction_bins=direction_bins, number_of_value_bins=number_of_value_bins,
        center_on_north=center_on_north
    )

    # Get an array of radial coordinates
    radii = __get_radii(
        value_bin_boundaries=value_bins, theta=theta, values=values,
        directions=directions, number_of_direction_bins=direction_bins
    )
    error = radii.sum() + calms - 100
    if not np.isclose([error], [0]):
        warnings.warn('Warning: cumulative error of {:.5f}%'.format(error))

    # Get an array of radial coordinates of bar bottoms
    bottoms = __get_bottoms(radii=radii, percentage_of_calms=calms)

    # Get an array of color values for value_bins
    colors = __get_colors(number_of_value_bins=number_of_value_bins, colormap=colormap)

    # Create a figure with poolar axes
    plt.figure(figsize=(8, 8), facecolor='w', edgecolor='w')
    ax = plt.axes(polar=True)
    ax.grid(b=True, color='grey', linestyle=':', axis='y')  # frequency grid
    ax.grid(b=True, color='grey', linestyle='--', axis='x')  # direction grid
    ax.set_theta_zero_location('N')  # move 'N' to top
    ax.set_theta_direction(-1)  # make plot clockwise

    # Setup radii axis
    min_rtick_number = min_ticks  # set minimum number of r-ticks (ensures clarity)
    if min_ticks <= 10:
        max_rtick_number = 10  # set maximum number of r-ticks (ensures clarity)
    else:
        max_rtick_number = 100
    rmax = max([i.sum() for i in radii.T]) + calms
    ntix = int(min(
        max(np.ceil(rmax / 5), min_rtick_number),
        max_rtick_number
    ))
    tick_size = int(np.ceil(rmax / ntix))
    ytix = np.array([tick_size * i for i in range(1, ntix + 1)])
    ytix = ytix[ytix <= rmax + tick_size]
    ytix = ytix[ytix >= calms]
    ymargin = tick_size / 2
    ax.set_rgrids(
        radii=ytix, labels=['{}%'.format(tick, '.0f') for tick in ytix],
        size='medium', style='italic'
    )
    ax.set_ylim(0, ytix[-1] + ymargin)

    # Setup theta axis
    ax.set_thetagrids(
        angles=np.arange(0, 360, 45), labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        size='large', style='italic'
    )

    # Generate bar labels for legend
    if calm_region_magnitude != -np.inf:
        bar_labels = ['{0:.2f} ≤ {1} < {2:.2f}'.format(calm_region_magnitude, value_name, value_bins[1])]
    else:
        bar_labels = ['{0} < {1:.2f}'.format(value_name, value_bins[1])]
    bar_labels.extend([
        '{0:.2f} ≤ {1} < {2:.2f}'.format(value_bins[i], value_name, value_bins[i + 1])
        for i in range(1, len(value_bins) - 1)
    ])
    bar_labels.extend(['{0} ≥ {1:.2f}'.format(value_name, value_bins[-1])])

    # Plot bars
    for i in range(len(theta)):
        ax.bar(
            theta[i], radii[i], width=(2 * np.pi / direction_bins) * notch, bottom=bottoms[i],
            color=colors[i], alpha=alpha, label=bar_labels[i], **bar_props
        )

    # Add legend and title
    ax.set_title(label=fig_title, y=1.08, size='xx-large')
    if calm_region_magnitude != -np.inf:
        # change color to see calms extent
        ax.bar(0, calms, 2*np.pi, 0, color='white', alpha=0.3,
               label='{0:.2f}% Calms ({1:.2f})'.format(calms, calm_region_magnitude))
    if unit_name:
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), title='{0} ({1})'.format(value_name, unit_name))
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), title='{0}'.format(value_name))

    # Save the figure and close it
    if save_path:
        try:
            plt.savefig(
                save_path,
                bbox_inches='tight', dpi=300
            )  # extend window to see legend, it saves perfectly as is
        except ValueError:
            plt.savefig(
                save_path + '.png',
                bbox_inches='tight', dpi=300
            )  # extend window to see legend, it saves perfectly as is
        finally:
            plt.close()
    else:
        plt.show()
