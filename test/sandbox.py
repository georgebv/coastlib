import pandas as pd
import numpy as np
import math


####
from coastlib.coreutils.adcp_tools import SentinelV as SlV
adcproot = 'C:\\Users\GRBH.COWI.001\Desktop\desktop projects\Living breakwaters\ADCP data processing\Data'
paths = [
    adcproot + '\ADCP1\ADCP1 034 May.mat'
]
data = SlV(paths[0])
data.waves_parse()
data.convert('waves', 'Hs', 'HsSea', 'HsSwell', systems='m to ft')
df = data.waves
kwargs = {}
####



val1 = kwargs.get('val1', 'Hs')
val2 = kwargs.get('val2', 'Tp')
binsize1 = kwargs.get('binsize1', 0.3)
binsize2 = kwargs.get('binsize2', 4)
savepath = kwargs.get('savepath', None)
savename = kwargs.get('savename', 'Joint Probability')

a = df[pd.notnull(df[val1])]
a = a[pd.notnull(a[val2])]
vals1 = a[val1]
vals2 = a[val2]
bins1 = math.ceil(vals1.max() / binsize1)
bins2 = math.ceil(vals2.max() / binsize2)
columns = []
low = 0
for i in range(bins1):
    up = low + binsize1
    columns += [str(int(low * 10) / 10) + ' - ' + str(int(up * 10) / 10)]
    low += binsize1
rows = []
low = 0
for i in range(bins2):
    up = low + binsize2
    rows += [str(int(low * 10) / 10) + ' - ' + str(int(up * 10) / 10)]
    low += binsize2
jp_raw = pd.DataFrame(0, index=rows, columns=columns)

for i in range(bins2):
    bin2_low = i * binsize2
    bin2_up = bin2_low + binsize2
    for j in range(bins1):
        bin1_low = j * binsize1
        bin1_up = bin1_low + binsize1
        count = 0
        for k in range(len(a)):
            if bin1_up > a[val1][k] > bin1_low and bin2_up > a[val2][k] > bin2_low:
                count += 1
        jp_raw[columns[j]][i] = count

#jp_raw = pd.concat([a[val1], a[val2]], axis=1, keys=[val1, val2])
