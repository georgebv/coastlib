import numpy as np
import pandas as pd
import warnings


def joint(value_1, value_2, binsize_1=0.3, binsize_2=4, relative=True):
    """
    Generates a joint probability table of 2 variables
    Filtering data before and removing empty columns after is up to user

    Input
    =====
    value_1, value_2 : list or array
        Arrays of equal length
    binsize_1, binsize_2 : float
        Bin sizes for variables value_1 and value_2

    Output
    ======
    Pandas dataframe with joint probability table
    """

    if (not isinstance(value_1, np.ndarray)) or (not isinstance(value_2, np.ndarray)):
        try:
            value_1 = np.array(value_1)
            value_2 = np.array(value_2)
        except Exception as _e:
            raise ValueError('Input values should be arrays.'
                             '{}'.format(_e))

    data = pd.DataFrame(data=value_1, columns=['v1'])
    data['v2'] = value_2

    def _round(_x):
        return float(format(_x, '.5f'))

    _b1min = _round(value_1.min() - value_1.min() % binsize_1)
    _b1max = _round(value_1.max() - value_1.max() % binsize_1 + binsize_1)

    _b2min = _round(value_2.min() - value_2.min() % binsize_2)
    _b2max = _round(value_2.max() - value_2.max() % binsize_2 + binsize_2)

    bots_1 = np.arange(_b1min-binsize_1, _b1max+binsize_1, binsize_1)
    bots_2 = np.arange(_b2min-binsize_2, _b2max+binsize_2, binsize_2)

    index_1 = ['(-inf ; {0:.2f}]'.format(bots_1[1])]
    for bot in bots_1[1:-1]:
        index_1 += ['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_1)]
    index_1 += ['[{0:.2f} ; inf)'.format(bots_1[-1])]

    index_2 = ['(-inf ; {0:.2f}]'.format(bots_2[1])]
    for bot in bots_2[1:-1]:
        index_2 += ['[{0:.2f} ; {1:.2f})'.format(bot, bot + binsize_2)]
    index_2 += ['[{0:.2f} ; inf)'.format(bots_2[-1])]

    bins = [[_round(bot), _round(bot + binsize_1)] for bot in bots_1]
    datas = [data[(data['v1'] >= bin[0]) & (data['v1'] < bin[1])] for bin in bins]

    table = np.zeros(shape=(len(index_1), len(index_2)))
    for i, _data in enumerate(datas):
        for j, bot_2 in enumerate(bots_2):
            top_2 = bot_2 + binsize_2
            table[i][j] = (
                (_data['v2'] >= bot_2) &
                (_data['v2'] < top_2)
            ).sum()

    if not np.isclose(len(value_1), table.sum()):
        warnings.warn('THE RESULT IS WRONG. Missing {} values.'.format(len(value_1) - table.sum()))

    if relative:
        table /= len(data)

    return pd.DataFrame(data=table, index=index_1, columns=index_2)
