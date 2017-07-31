import numpy as np
import warnings
import scipy.constants


def seabrook_hall(wave_height, ds, crest_width, wave_length, stone_size):
    """
    Calculates wave transmission coefficient for submerged breakwaters
    using the Seabrook & Hall (1998) formula

    :param wave_height: incident significant wave height [m]
    :param ds: depth of submergence (positive value of the negative freeboard) [m]
    :param crest_width: crest width [m]
    :param wave_length: wave length [m]
    :param stone_size: rock armor median rock size [m]
    :return: wave transmission coefficient
    """

    kt = 1 - (
        np.exp(-0.65 * ds / wave_height - 1.09 * wave_height / crest_width) 
        + 0.047 * crest_width * ds / (wave_length * stone_size) 
        - 0.067 * ds * wave_height / (crest_width * stone_size)
    )
    condition_1 = crest_width * ds / (wave_length * stone_size)
    condition_2 = ds * wave_height / (crest_width * stone_size)
    if 0 <= condition_1 <= 7.08 and 0 <= condition_2 <= 2.14:
        return kt
    else:
        warnings.warn('crest_widthreakwater parameters beyond the validity levels')
        return np.nan


def d_angremond(freeboard, wave_height, crest_width, wave_period, tana):
    """
    Calculates wave transmission coefficient for emerged breakwaters
    using the dAngremond (1996) (EurOtop 2016) formula

    :param freeboard: freeboard [m]
    :param wave_height: incident significant wave height [m]
    :param crest_width: crest width [m]
    :param wave_period: wave period [sec]
    :param tana: seaward breakwater slope
    :return: wave transmission coefficient
    """

    s_op = 2 * np.pi * wave_height / (scipy.constants.g * wave_period ** 2)
    e_op = tana / s_op ** 0.5
    kt_small = -0.4 * freeboard / wave_height \
               + 0.64 * (crest_width / wave_height) ** (-0.31) * (1 - np.exp(-0.5 * e_op))
    kt_large = -0.35 * freeboard / wave_height \
               + 0.51 * (crest_width / wave_height) ** (-0.65) * (1 - np.exp(-0.41 * e_op))
    scale = crest_width / wave_height

    if scale < 8:
        kt = kt_small
    elif scale > 12:
        kt = kt_large
    else:
        kt = (kt_large - kt_small) * (scale - 8) / (12 - 8) + kt_small  # linear interpolation

    if kt >= 0.8:
        return 1
    elif kt <= 0.075:
        return 0.075
    else:
        return kt
