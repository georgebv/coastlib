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
