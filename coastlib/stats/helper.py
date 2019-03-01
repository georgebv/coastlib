import numpy as np
import pandas as pd


def joint_table(array_1, array_2, bins_1, bins_2, normed=False):
    """
    Generates joint probability table of 2 variables.

    Parameters
    ----------
    array_1, array_2 : array_like
        Arrays with values. Must have equal length.
    bins_1, bins_2 : array_like
        Arrays with bin edges for <array_1> and <array_2> accordingly.
    normed : bool, optional
        If True, scales joint quantities so that sum of all vlaues is equal to 1 (default=False).

    Returns
    -------
    out : pandas DataFrame
        Pandas dataframe with joint probability quantities.
    """

    # Make sure values are arrays
    if (not isinstance(array_1, np.ndarray)) or (not isinstance(array_2, np.ndarray)):
        try:
            array_1 = np.array(array_1)
            array_2 = np.array(array_2)
        except Exception as _e:
            raise ValueError(f'{_e}\nInput values must be 1D arrays.')

    # Generate indexes for values_1
    index_1 = []
    for i, value in enumerate(bins_1[:-1]):
        index_1.append(f'[{value:.2f} ; {bins_1[i+1]:.2f})')

    # Generate indexes for values_2
    index_2 = []
    for i, value in enumerate(bins_2[:-1]):
        index_2.append(f'[{value:.2f} ; {bins_2[i+1]:.2f})')

    table = np.histogram2d(array_1, array_2, [bins_1, bins_2], normed=False)

    if not np.isclose(len(array_1), table[0].sum()):
        print(f'Missing {len(array_1) - table[0].sum()} values. Check <bins_1> and <bins_2>')

    if normed:
        return pd.DataFrame(data=table[0] / len(array_1), index=index_1, columns=index_2)
    else:
        return pd.DataFrame(data=np.int64(table[0]), index=index_1, columns=index_2)


if __name__ == '__main__':
    import os
    source = 'RR_wind'
    df = pd.read_csv(
        os.path.join(os.getcwd(), f'test data\\{source}.csv'),
        index_col=0, parse_dates=True
    )
    jp = joint_table(
        array_1=df['Spd'].values, array_2=df['Dir'].values,
        bins_1=np.arange(0, 24, 2), bins_2=np.arange(0, 390, 30)
    )
