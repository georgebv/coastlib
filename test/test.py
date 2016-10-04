"TEST FUNCTIONS"
from coastlib.coreutils.adcp_tools import SentinelV as SlV
import coastlib.coreutils.plot_tools as plots

# Set paths to adcp *.mat data file
adcproot = 'C:\\Users\GRBH.COWI.001\Desktop\desktop projects\Living breakwaters\ADCP data processing\Data'
paths = [
    adcproot + '\ADCP1\ADCP1 034 May.mat'
]
data = SlV(paths[0])
data.waves_parse()
data.convert('waves', 'Hs', 'HsSea', 'HsSwell', systems='m to ft')
df = data.waves
savepath = 'C:\\Users\GRBH.COWI.001\Desktop\GitHub repositories\coastlib\\test'

plots.pdf_plot(df, savepath=savepath)
# plots.joint_plot(df, savepath=savepath)
plots.time_series_plot(df, savepath=savepath)
plots.rose_plot(df, val='Hs', direction='Dp', dirbins=24, startfromzero=True, savepath=savepath)
plots.joint_plot(df, val1='Hs', val2='Tp', xlabel='Hs [ft]', ylabel='Tp [sec]', savepath=savepath)
