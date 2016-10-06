"""TEST FUNCTIONS"""
import coastlib.coreutils.plot_tools as plots
import coastlib.coreutils.data_analysis_tools as dat
from coastlib.coreutils.adcp_tools import SentinelV as SlV

# Set paths to adcp *.mat data file
adcproot = 'C:\\Users\GRBH.COWI.001\Desktop\desktop projects\Living breakwaters\ADCP data processing\Data'
paths = [
    adcproot + '\ADCP1\ADCP1 537 March.mat'
]
data = SlV(paths[0])
data.waves_parse()
data.convert('waves', 'Hs', 'HsSea', 'HsSwell', 'SHmax', systems='m to ft')
df = data.waves
savepath = 'C:\\Users\GRBH.COWI.001\Desktop\GitHub repositories\coastlib\\test'
#data.export('waves', save_format='csv', save_name='waves dataframe', save_path=savepath)
#data.export('waves', save_format='xlsx', save_name='waves dataframe', save_path=savepath)

plots.pdf_plot(df, savepath=savepath)
plots.time_series_plot(df, savepath=savepath)
plots.rose_plot(df, val='Hs', direction='Dp', dirbins=24, valbinsize=0.5, valbin_max=3,
                startfromzero=True, savepath=savepath)
plots.rose_plot(df, val='Tp', direction='Dp', dirbins=24, valbinsize=4,
                startfromzero=True, savepath=savepath)
plots.joint_plot(df, val1='Hs', val2='Dp', xlabel='Hs [ft]', ylabel='Dp [deg]', savepath=savepath)
dat.joint_probability(df, val1='Hs', val2='Tp', savepath=savepath)
plots.heatmap(dat.joint_probability(df, val1='Hs', val2='Tp'), savepath=savepath)