import numpy as np
from coastlib.core.plotting.rose_plot import rose_plot
import matplotlib.pyplot as plt


# Parse test data
with open(r'C:\Users\georg\Documents\GitHub\coastlib\coastlib\core\plotting\test.txt', 'r') as f:
    data = f.readlines()
data = [list(filter(None, i.split(sep=' '))) for i in data]
values = np.array([float(i[8]) for i in data[1:]])
directions = np.array([float(i[11]) for i in data[1:]])
values = values[directions != 999]
directions = directions[directions != 999]
assert len(values) == len(directions)

rose_plot(
    values, directions, value_bins=np.arange(0, 5, 1),
    colormap=plt.get_cmap('jet'), alpha=1, direction_bins=64, calm_region=0.35,
    min_ticks=4,
    save_path=r'C:\Users\georg\Documents\GitHub\coastlib\coastlib\core\plotting\on north',
    center_on_north=True
)

