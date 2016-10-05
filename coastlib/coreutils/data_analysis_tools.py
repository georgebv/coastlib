import math

import numpy as np
import openpyxl
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
    kwargs:
    """
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
