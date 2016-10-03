"TEST FUNCTIONS"
from coastlib.coreutils.adcp_tools import SentinelV as SlV
from coastlib.coreutils.data_analysis_tools import joint_probability
from coastlib.coreutils.data_analysis_tools import associated_value

# Set paths to adcp *.mat data file
adcproot = 'C:\\Users\GRBH.COWI.001\Desktop\desktop projects\Living breakwaters\ADCP data processing\Data'
paths = [
    adcproot + '\ADCP1 May.mat',
    adcproot + '\ADCP2 August.mat'
]
data = SlV(paths[0])
data.waves_parse()
data.convert('waves', 'Hs', 'HsSea', 'HsSwell', systems='m to ft')
df = data.waves
savepath = 'C:\\Users\GRBH.COWI.001\Desktop\GitHub repositories\coastlib\\test'

joint_probability(df, savepath=savepath)
associated_value(df, 'Hs', 'Tp', 2)
