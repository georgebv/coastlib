import numpy as np
from coastlib.core.plotting.rose import rose_plot
import matplotlib.pyplot as plt


# Parse test data
with open(r'C:\Users\georg\Documents\GitHub\coastlib\coastlib\core\plotting\test2.txt', 'r') as f:
    data = f.readlines()
data = [list(filter(None, i.split(sep=' '))) for i in data]
values = np.array([float(i[8]) for i in data[1:]])
directions = np.array([float(i[11]) for i in data[1:]])
values = values[directions != 999]
directions = directions[directions != 999]
assert len(values) == len(directions)

rose_plot(
    values, directions, value_bins=np.arange(0, 5, 1),
    colormap=plt.get_cmap('jet'), alpha=.8, direction_bins=32, calm_region=0.35)


#=================================#
value_bins = np.arange(0, 5, 1)
direction_bins=16
number_of_direction_bins = 16
number_of_value_bins = 4
center_on_north = False
calm_region_magnitude = 0.35
value_bin_boundaries = value_bins

calms=__get_calms(values, calm_region_magnitude)
(values < calm_region_magnitude).sum()
theta = __get_theta(number_of_direction_bins, number_of_value_bins, center_on_north)
radii = __get_radii(value_bin_boundaries, theta, values, directions, number_of_value_bins, number_of_direction_bins)
