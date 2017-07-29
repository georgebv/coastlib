import matplotlib.pyplot as plt
import numpy as np


with open(r'C:\Users\georg\Documents\GitHub\coastlib\coastlib\core\plotting\test.txt', 'r') as f:
    data = f.readlines()
data = [list(filter(None, i.split(sep=' '))) for i in data]
values = np.array([float(i[8]) for i in data[1:]])
directions = np.array([float(i[11]) for i in data[1:]])
values = values[directions != 999]
directions = directions[directions != 999]
assert len(values) == len(directions)


number_of_direction_bins = 8
calm_region_magnitude = 0.5
value_bin_boundaries = np.arange(calm_region_magnitude, 3, 0.5)  # boundaries for the value bins [0;1)...[4;inf)
number_of_value_bins = len(value_bin_boundaries) - 1
center_on_north = False  # This flag specifies if bins intersect main directions or not
bar_notch = 0.9  # notch between bars proportional to bin width (only visual)
colormap = plt.cm.jet
assert len(value_bin_boundaries)-1 == number_of_value_bins

# Get calms and filter the data
percentage_of_calms = (values < calm_region_magnitude).sum() / len(values)
percentage_of_calms *= 100

# Generate an array of centers of each bin
if center_on_north:
    theta = np.linspace(0, 2 * np.pi, number_of_direction_bins, endpoint=False)
else:
    theta = np.linspace(0, 2 * np.pi, number_of_direction_bins, endpoint=False) + np.pi / number_of_direction_bins
theta = np.array([theta for i in range(number_of_value_bins+1)])  # array of thetas with 1 row per 1 value bin

# Generate an array of value bin boundaries in frequencies
if not value_bin_boundaries.any():
    value_bin_boundaries = np.linspace(0, 1, number_of_value_bins+1) * values.max()  # value bin boundaries
radii = []
__dangle = np.rad2deg((np.pi / number_of_direction_bins))
for __angle in np.rad2deg(theta[0]):
    __values = values[(directions >= __angle-__dangle) & (directions < __angle+__dangle)]  # select values in this bin
    value_bins = []
    for j in range(number_of_value_bins):
        value_bins.extend([(
            (__values >= value_bin_boundaries[j]) & (__values < value_bin_boundaries[j+1])
            ).sum() / len(values)])
    value_bins.extend([(__values >= value_bin_boundaries[-1]).sum() / len(values)])
    radii += [value_bins]
radii = np.array(radii).T  # frequencies for each value bin per each direction bin (or coordinates of end of each bar)
radii *= 100

colors = [colormap(i) for i in np.linspace(0.0, 1.0, number_of_value_bins+1)]   # an array of colors for each value bin
                                                                                # patch

__radii = [row.cumsum() for row in radii]
__radii = [np.insert(row, 0, 0) for row in __radii]
__radii = [row[:-1] for row in __radii]
bottoms = np.zeros(shape=np.shape(radii))
bottoms[0] = [percentage_of_calms] * len(bottoms[0])
for i in range(1, len(bottoms)):
    for j in range(len(bottoms[i])):
        bottoms[i][j] = bottoms[i-1][j] + radii[i-1][j]  # this is the bottom of each bar



ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location('N')
for i in range(len(theta)):
    bars = ax.bar(theta[i], radii[i], width=np.deg2rad(__dangle*2*bar_notch), bottom=bottoms[i])
    for bar in bars:
        bar.set_facecolor(colors[i])
        bar.set_alpha(0.8)
plt.show()




N = 20  # number of bars
bottom = 1  # 'calm region' height


theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False) + np.pi / N
radii = 10*np.random.rand(N)
width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=bottom)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()
