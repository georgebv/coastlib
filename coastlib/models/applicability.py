import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


path = r'C:\Users\GRBH\Desktop\GitHub Repositories\coastlib\coastlib\models\data'

h_gt2 = np.linspace(start=0.00005, stop=0.05, num=100, endpoint=True)
d_gt2 = np.linspace(start=0, stop=0.2, num=100, endpoint=True)

data_all = pd.read_csv(os.path.join(path, r'applicability_all.csv'))
