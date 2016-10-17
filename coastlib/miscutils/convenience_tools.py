import os
import sys
import functools
import pandas as pd


def splice(df_list, sort=True):
    """
    Takes a list of dataframes and returns a merged dataframe (sorted by default).
    """
    df = functools.reduce((lambda x, y: pd.concat([x, y])), df_list)
    if sort:
        df.sort_index(inplace=True)
    return df


def ensure_dir(f):
    """
    Checks the existance of path *f* and if it doesn't exist - creates it.
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def intersection(z, tab):
    """
    Finds 'x' value (1st column) of a point with a given value 'z' (second column) in an array 'xztab' through linear
    interpolation. Return the first match starting from the beginning (1st row) of the input array.
    Parameters
    ----------
    z : float
        Target value in the second column of the input array.
    tab : ndarray
        Numpy array with 'x' and 'z' values (column 0 - x, column 1 - z)
    Returns
    -------
    xint : list
        List of 'x' values corresponding to the 'z' value in the 'xztab' array.
    """
    xint = []
    for row in range(0, len(tab[:, 0]) - 1):
        if tab[row, 1] <= z <= tab[row + 1, 1] or tab[row, 1] >= z >= tab[row + 1, 1]:
            xint += [
                ((z - tab[row, 1]) * (tab[row + 1, 0] - tab[row, 0]) /
                    (tab[row + 1, 1] - tab[row, 1]) + tab[row, 0])
            ]
    return xint


def print_update(text):
    """
    Updates the same line in the terminal as long as no new lines are printed inbetween.
    Useful when creating progress bars.
    """
    sys.stdout.write('\r{}'.format(text))
    sys.stdout.flush()
