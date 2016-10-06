import math
import pandas as pd
import statsmodels.api as sm


def joint_probability(df, **kwargs):
    """
    Generates a joint probability table of 2 variables.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe
    val1, val2 : str
        Column names in df
    binsize1, binsize2 : float
        Bin sizes for variables val1 and val2
    savepath, savename : str
        Save folder path and file save name
    output_format : str
        Joint table values (absolute 'abs' or relative / percent 'rel')
    """
    val1 = kwargs.get('val1', 'Hs')
    val2 = kwargs.get('val2', 'Tp')
    binsize1 = kwargs.get('binsize1', 0.3)
    binsize2 = kwargs.get('binsize2', 4)
    savepath = kwargs.get('savepath', None)
    savename = kwargs.get('savename', 'Joint Probability')
    output_format = kwargs.get('output_format', 'rel')

    a = df[pd.notnull(df[val1])]
    a = a[pd.notnull(a[val2])]
    bins1 = math.ceil(a[val1].max() / binsize1)
    bins2 = math.ceil(a[val2].max() / binsize2)
    columns = []
    low = 0
    for i in range(bins1):
        up = low + binsize1
        columns += ['{0:.1f} - {1:.1f}'.format(low, up)]
        low += binsize1
    rows = []
    low = 0
    for i in range(bins2):
        up = low + binsize2
        rows += ['{0:.1f} - {1:.1f}'.format(low, up)]
        low += binsize2
    if output_format == 'abs':
        jp_raw = pd.DataFrame(0, index=rows, columns=columns)
    else:
        jp_raw = pd.DataFrame(.0, index=rows, columns=columns)
    for i in range(bins2):
        bin2_low = i * binsize2
        bin2_up = bin2_low + binsize2
        for j in range(bins1):
            bin1_low = j * binsize1
            bin1_up = bin1_low + binsize1
            b = a[(a[val1] < bin1_up) &
                  (a[val1] > bin1_low) &
                  (a[val2] < bin2_up) &
                  (a[val2] > bin2_low)]
            if output_format == 'abs':
                jp_raw[columns[j]][i] = len(b)
            elif output_format == 'rel':
                jp_raw[columns[j]][i] = len(b) / len(a)
            else:
                raise ValueError('output format should be either *abs* or *rel*')
    if savepath is not None:
        jp_raw.to_excel(pd.ExcelWriter(savepath + '\\' + savename + '.xlsx'), sheet_name='joint_prob', )
    else:
        return jp_raw


def associated_value(df, val, par, value, search_range=0.1):
    """
    For datframe df, value *val* (i.e. 'Hs') and parameter *par* (i.e. 'Tp')
    returns parameter value statistically associated with *val* *value*
    """
    df = df[pd.notnull(df[val])]
    df = df[pd.notnull(df[par])]
    val_range = df[val].max() - df[val].min()
    a_low = value - search_range * val_range
    a_up = value + search_range * val_range
    a = df[(df[val] > a_low) & (df[val] < a_up)]
    par_array = a[par].as_matrix()
    dens = sm.nonparametric.KDEUnivariate(par_array)
    dens.fit()
    return dens.support[dens.density.argmax()]