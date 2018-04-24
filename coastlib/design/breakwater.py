import warnings

import numpy as np
import scipy.constants
import scipy.interpolate
import scipy.stats


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
        print('Parameters beyond the validity levels. Returned <nan>')
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


def hudson(wave_height, alpha, rock_density, kd=4, **kwargs):
    """
    Solves Hudson equation for median rock diameter. Checks stability (Ns < 2)

    Mandatory inputs
    ================
    wave_height : float
        Significant wave height at structure's toe (m) (CEM formulation suggests using a 1.27 factor for Hs)
    alpha : float
        Structure angle (degrees to horizontal)
    rock_density : float
        Rock density (kg/m^3)
    kd : float
        Dimensionless stability coefficient (4 for pemeable core (default), 1 for impermeable core)
    sd : float
        Damage level (2 for 0-5% damage level)
    formulation : str
        CEM - Coastal Engineering Manual VI-5-73 (Hudson 1974)
        CIRIA (default) - The Rock Manual (p.565, eq.5.135) - the more in-depth formulation by Van der Meer

    Returns
    =======
    Dn50 : float
        Nominal median diameter of armour blocks (m)
    """

    sd = kwargs.pop('sd', 2)
    formulation = kwargs.pop('formulation', 'CIRIA')
    rho = kwargs.pop('rho', 1025)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    delta = rock_density / rho - 1
    alpha = np.deg2rad(alpha)
    if formulation == 'CIRIA':
        dn50 = wave_height / (delta * 0.7 * (kd / np.tan(alpha)) ** (1 / 3) * sd ** 0.15)
        ns = wave_height / (delta * dn50)
    elif formulation == 'CEM':
        dn50 = wave_height / (delta * (kd / np.tan(alpha)) ** (1 / 3) / 1.27)
        ns = wave_height / (delta * dn50)
    else:
        raise ValueError('Formulation {0} not recognized. Use CIRIA or CEM.'.format(formulation))

    if ns > 2:
       print(
           'Armor is not stable with the stability number Ns={0}, Dn50={1} m'.
           format(round(ns, 2), round(dn50, 2))
       )
    return dn50


def runup(wave_height, wave_period, slope, **kwargs):
    # TODO : update using new Eurotop 2016
    """
    Calculates run-up height.
    Find 2% run-up height for <type> structure using the EurOtop (2007) manual methods.

    Mandatory inputs
    ================
    wave_height : float
        Significant wave height at structure toe (m) .
    wave_period : float
        Peak wave period (s).
    slope : float
        Structure slope.
    Yf : float (optional)
        Roughness factor (1 for concrete), p.88 EurOtop manual (2007).
    B : float (optional)
        Wave attack angle (deg).
    rB : float (optional)
        Berm width (m), 0 for no berm.
    Lb : float (optional)
        Characteristic berm length (m), refer to p.95 EurOtop (2007).
    rdb : float (optional)
        Difference between SWL and berm elevation (m), refer to p.96 EurOtop (2007).
    strtype : string (optional)
        Structure type: 'sap' for simple armored slope (default);
    dmethod : string (optional)
        Design method: 'det' for deterministic design (default), more conservative; 'prob' for probabilistic design.

    Returns
    =======
    ru : float
        Estimated 2% runup (m) for parameters entered.
    """

    B = kwargs.pop('B', 0)
    Yf = kwargs.pop('Yf', 1)
    rB = kwargs.pop('rB', 0)
    Lb = kwargs.pop('Lb', 1)
    rdb = kwargs.pop('rdb', 0)
    strtype = kwargs.pop('strtype', 'sap')
    dmethod = kwargs.pop('dmethod', 'det')
    g = kwargs.pop('g', scipy.constants.g)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    Lm10 = g * (wave_period ** 2) / (2 * np.pi)  # Deep water wave length
    Sm10 = wave_height / Lm10  # Wave steepness
    Em10 = np.tan(slope) / Sm10 ** 0.5  # Breaker type
    rB /= Lb
    Yb = 1 - rB * (1 - rdb)  # Berm factor (1 for no berm)

    if B < 80:
        YB = 1 - 0.0022 * B
    else:
        YB = 0.824

    if Em10 > 10:
        Yfsurg = 1
    else:
        Yfsurg = Yf + (Em10 - 1.8) * (1 - Yf) / 8.2

    if strtype is 'sap':
        if dmethod is 'det':
            ru = wave_height * 1 * Yb * Yfsurg * YB * (4.3 - 1.6 / (Em10 ** 0.5))
            ru = min(ru, wave_height * 2.11)  # Maximum for permeable core
            return ru
        elif dmethod is 'prob':
            ru = wave_height * 1 * Yb * Yfsurg * YB * (4 - 1.5 / (Em10 ** 0.5))
            ru = min(ru, wave_height * 1.97)  # Maximum for permeable core
            return ru
        else:
            raise ValueError('ERROR: Design method not recognized')
    else:
        raise ValueError('ERROR: Structure type not recognized')


# TODO - below are not implemented


def van_der_meer(Hs, h, Tp, alpha, rock_density, **kwargs):
    """
    Finds median rock diameter Dn50 using Van der Meer formula (The Rock Manual 2007, p.)

    :param Hs: float
        Significant wave height at structure toe [m]
    :param h: float
        Water depth at structure toe [m]
    :param Tp: float
        Peak wave period [s]
    :param alpha: float
        Structure seaward side slope [degrees]
    :param rock_density: float
        Armor rock density [kg/m^3]
    :param kwargs:
        Tm : float
            Mean wave period. By default calculated using JONSWAP spectra with gamma=3.3
        P : float
            Notional premeability of the structure. 0.1 <= P <= 0.6
        Sd : int
            Damage level parameter (default 2 for 0-5% damage as originally per Hudson)
        N : int
            Number of waves in a storm (N <= 7500 - default values)
    :return:
        Dn50 : float
            Nominal median diameter of armour blocks (m)
    """
    Tm = kwargs.pop('Tm', 0.8 * Tp)
    P = kwargs.pop('P', 0.4)
    Sd = kwargs.pop('Sd', 2)
    N = kwargs.pop('N', min(7500, int(np.ceil(3*3600/Tm))))
    echo = kwargs.pop('echo', True)
    sea_water_density = kwargs.pop('sea_water_density', 1030)
    assert isinstance(N, int), 'Number of waves should be a natural number'
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    alpha = np.deg2rad(alpha)
    delta = rock_density / sea_water_density - 1
    H_2p = 1.4 * Hs
    if h < 3 * Hs:
        if echo:
            print('Shallow water')
        c_pl = 8.4
        c_s = 1.3
        xi_cr = ((c_pl / c_s) * (P ** 0.31) * np.sqrt(np.tan(alpha))) ** (1 / (P + 0.5))
        xi_s10 = np.tan(alpha) / np.sqrt(2 * np.pi * Hs / (scipy.constants.g * Tm ** 2))
        if xi_s10 < xi_cr:
            if echo:
                print('Plunging conditions')
            right = c_pl * (P ** 0.18) * ((Sd / np.sqrt(N)) ** 0.2) * (Hs / H_2p) * (xi_s10 ** (-0.5))
            return Hs / (delta * right)
        else:
            if echo:
                print('Surging conditions')
            right = c_s * (P ** (-0.13)) * ((Sd / np.sqrt(N)) ** 0.2) * (Hs / H_2p) * np.sqrt(1 / np.tan(alpha)) \
                    * (xi_s10 ** (-0.5))
            return Hs / (delta * right)
    else:
        if echo:
            print('Deep water')
        c_pl = 6.2
        c_s = 1.0
        xi_cr = ((c_pl / c_s) * (P ** 0.31) * np.sqrt(np.tan(alpha))) ** (1 / (P + 0.5))
        xi_m = np.tan(alpha) / np.sqrt(2 * np.pi * Hs / (scipy.constants.g * Tm ** 2))
        if xi_m < xi_cr:
            if echo:
                print('Plunging conditions')
            right = c_pl * (P ** 0.18) * ((Sd / np.sqrt(N)) ** 0.2) * (xi_m ** (-0.5))
            return Hs / (delta * right)
        else:
            if echo:
                print('Surging conditions')
            right = c_s * (P ** (-0.13)) * ((Sd / np.sqrt(N)) ** 0.2) * np.sqrt(1 / np.tan(alpha)) * (xi_m ** P)
            return Hs / (delta * right)


def vanGent(Hs, rock_density, alpha, Dn50_core, **kwargs):
    """
    Finds median rock diameter Dn50 using Van Gent formula (The Rock Manual 2007)

    Parameters
    ----------
    Hs : float
        Significant wave height (average of 1/3 highest waves, NOT SPECTRAL)
    rock_density : float
        Armor unit density [kg/m^3]
    alpha : float
        Seaward side slope [degrees]
    Dn50_core : float
        Core stone density [kg/m^3]
    kwargs : varies
        Sd : int
            Damage level parameter (default 2 for 0-5% damage as originally per Hudson)
        N : int
            Number of waves in a storm (N <= 7500 - default values)

    Returns
    -------
    Dn50 : float
        Nominal median diameter of armour blocks (m)
    """
    Sd = kwargs.pop('Sd', 2)
    N = kwargs.pop('N', 7500)
    sea_water_density = kwargs.pop('sea_water_density', 1030)
    assert isinstance(N, int), 'Number of waves should be a natural number'
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    alpha = np.deg2rad(alpha)
    delta = rock_density / sea_water_density - 1

    def VGf(Dn50):
        left = Hs / (delta * 1.75 * np.sqrt(1 / np.tan(alpha)) * (Sd / np.sqrt(N)) ** 0.2)
        right = (Dn50 ** 3 + 2 * Dn50_core * Dn50 ** 2 + Dn50 * Dn50_core ** 2) ** (1 / 3)
        return left - right

    def VGf_prime(Dn50):
        left = 0
        numerator = (1 / 3) * (3 * Dn50 ** 2 + 4 * Dn50_core * Dn50 + Dn50_core ** 2)
        denominator = (Dn50 ** 3 + 2 * Dn50_core * Dn50 ** 2 + Dn50 * Dn50_core ** 2) ** (2 / 3)
        right = numerator / denominator
        return right - left

    try:
        Dn50_res = scipy.optimize.newton(VGf, 0.5, fprime=VGf_prime, maxiter=50)
        print('Solved using Newton-Rhapson method')
        return Dn50_res
    except:
        try:
            Dn50_res = scipy.optimize.newton(VGf, 0.5, maxiter=50)
            print('Solved using Secant method')
            return Dn50_res
        except:
            print('Solved using "Brute Force" method')
            Dn50 = np.arange(0.05, 20, 0.002)
            def abs_Gent(Dn50):
                left = Hs / (delta * 1.75 * np.sqrt(1 / np.tan(alpha)) * (Sd / np.sqrt(N)) ** 0.2)
                right = (Dn50 ** 3 + 2 * Dn50_core * Dn50 ** 2 + Dn50 * Dn50_core ** 2) ** (1 / 3)
                return np.abs(left - right)
            return Dn50[abs_Gent(Dn50).argmin()]


def Vidal(Hs, rock_density, Rc, **kwargs):
    """
    Finds median rock diameter Dn50 using Vidal formula (The Rock Manual 2007, p.602, eq.5.167)

    Parameters
    ----------
    Hs : float
        Significant wave height (average of 1/3 highest waves, NOT SPECTRAL) [m]
    rock_density : float
        Armor unit density [kg/m^3]
    Rc : float
        Water freeboard (crest elevation - water elevation) [m]

    Returns
    -------
    Dn50 : float
        Nominal median diameter of armour blocks (m)
    """
    sea_water_density = kwargs.pop('sea_water_density', 1030)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    coefficients = {
        'front slope' : (1.831, -0.2450, 0.0119),
        'crest' : (1.652, 0.0182, 0.1590),
        'back slope' : (2.575, -0.5400, 0.1150),
        'total section' : (1.544, -0.230, 0.053)
        # 'kramer and burcharth' : (1.36, -0.23, 0.06)
    }
    # solutions = []
    roots = []
    delta = rock_density / sea_water_density - 1
    for segment in coefficients.keys():
        A, B, C = coefficients[segment]
        a, b, c = A, B * Rc - Hs / delta, C * Rc ** 2
        # def func(Dn50):
        #     return a * Dn50 ** 2 + b * Dn50 + c
        # def func_prime(Dn50):
        #     return 2 * a * Dn50 + b
        #
        # solutions += [scipy.optimize.fsolve(func=func, x0=1, fprime=func_prime)]
        roots += [max(
            (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a),
            (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        )]
    roots = np.array(roots)[~np.isnan(np.array(roots))]
    if len(roots) == 0:
        warnings.warn('No solutions exist for input conditions')
        return np.nan
    else:
        if Rc / max(roots) < -2.01 or Rc / max(roots) > 2.41\
                or Hs / (delta * max(roots)) < 1.1 or Hs / (delta * max(roots)) > 3.7:
            print('Parameters beyond the range of validity!')
            return np.nan
        else:
            return max(roots)


def d50w50(unit: float, mode: str='d50 to w50') -> float:
    """
    W50 to D50 and vise versa as per CEM p.VI-5-129 (Table VI-5-50)

    Parameters
    ----------
    unit : float
        Unit to convert - either [ft] or [ton]
    mode : str
    Returns
    -------

    """
    dimensions = np.concatenate((
        np.array([4.3, 5.42, 6.21, 6.83, 7.36, 7.82, 8.23, 8.6, 8.95, 9.27, 9.57,
                  9.85, 10.12, 10.37, 10.61, 10.84, 11.06, 11.28, 11.48]) / 12,
        np.array([0.97, 1.23, 1.40, 1.54, 1.66, 1.77, 1.86, 1.95, 2.02, 2.10, 2.16,
                  2.23, 2.27, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60]),
        np.array([2.64, 3.33, 3.81, 4.19, 4.52, 4.80, 5.05, 5.28, 5.49, 5.69, 5.88,
                  6.05, 6.21, 6.37, 6.51, 6.66, 6.79, 6.92, 7.05, 7.17])
    ))  # US_foot
    weights = np.concatenate((
        np.arange(5, 100, 5) / 2000,
        np.arange(100, 2000, 100) / 2000,
        np.arange(1, 21, 1)
    ))  # US_ton
    if mode == 'd50 to w50':
        fit = scipy.interpolate.interp1d(dimensions, weights, kind='cubic')
    elif mode == 'w50 to d50':
        fit = scipy.interpolate.interp1d(weights, dimensions, kind='cubic')
    else:
        raise ValueError('Mode not recognized. Use \'w50 to d50\' or \'d50 to w50\'')
    return float(fit(unit))




def overtopping(Hm0, Rc, **kwargs):
    """
    Calculates mean overtopping discharge.
    Find mean overtopping discharge for <type> structure using the EurOtop (2007) and (2016) manual methods.

    Parameters
    ----------
    Hm0 : float
        Significant wave height at structure toe (m).
    Rc : float
        Freeboard, distance from structure crest to SWL (m).
    Yf : float (optional)
        Roughness factor (1 for concrete), p.88 EurOtop manual (2007).
    B : float (optional)
        Wave attack angle (deg).
    strtype : string (optional)
        Structure type: 'sap' for simple armored slope (default);
    dmethod : string (optional)
        Design method: 'det' for deterministic design (default), more conservative; 'prob' for probabilistic design.
    manual : string (optional)
        Manual to be used (2016 the default, 2007 is the old one).
    confidence : float (optional)
        Confidence level for overtopping in the manual=2016 formulation.
    bound : str (optional)
        For 2016 manual, set to either lower, median, or upper.

    Returns
    -------
    q : float
        Estimated mean overtopping discharge (m^3/s/m) for parameters entered.
    """
    B = kwargs.pop('B', 0)
    Yf = kwargs.pop('Yf', 1)
    strtype = kwargs.pop('strtype', 'sap')
    dmethod = kwargs.pop('dmethod', 'det')
    manual = kwargs.pop('manual', 2016)
    confidence = kwargs.pop('confidence', 0.90)
    bound = kwargs.pop('bound', 'upper')
    g = kwargs.pop('g', scipy.constants.g)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    if manual == 2007:
        if B < 80:
            YB = 1 - 0.0033 * B
        else:
            YB = 0.736

        if strtype is 'sap':
            if dmethod is 'det':
                q = ((g * (Hm0 ** 3)) ** 0.5) * 0.2 * np.exp(-2.3 * Rc / (Hm0 * Yf * YB))
                return q
            elif dmethod is 'prob':
                q = ((g * (Hm0 ** 3)) ** 0.5) * 0.2 * np.exp(-2.6 * Rc / (Hm0 * Yf * YB))
                return q
            else:
                raise ValueError('ERROR: Design method not recognized')
        else:
            raise ValueError('ERROR: Structure type not recognized')
    elif manual == 2016:

        if np.abs(B) < 80:
            YB = 1 - 0.0063 * np.abs(B)
        else:
            raise NotImplementedError('Oblique waves >80 degrees not implemented')

        if bound == 'lower':
            coeff_009 = scipy.stats.distributions.norm.interval(alpha=confidence, loc=0.09, scale=0.0135)[0]
            coeff_15 = scipy.stats.distributions.norm.interval(alpha=confidence, loc=1.5, scale=0.15)[1]
        elif bound == 'upper':
            coeff_009 = scipy.stats.distributions.norm.interval(alpha=confidence, loc=0.09, scale=0.0135)[1]
            coeff_15 = scipy.stats.distributions.norm.interval(alpha=confidence, loc=1.5, scale=0.15)[0]
        elif bound == 'median':
            coeff_009 = 0.09
            coeff_15 = 1.5
        else:
            raise ValueError('ERROR: Unrecognized bound value')

        if strtype is 'sap':
            return ((g * (Hm0 ** 3)) ** 0.5) * coeff_009 * np.exp(-(coeff_15 * Rc / (Hm0 * Yf * YB)) ** (1.3))
        else:
            raise ValueError('ERROR: Structure type not recognized')
    else:
        raise ValueError('ERROR: Manual not recognized')
