import math
from coastlib.models.linear_wave_theory import LinearWave as lw
import scipy.constants
import pandas as pd
import warnings

g = scipy.constants.g  # gravity constant (m/s^2) as defined by ISO 80000-3
sea_water_density = 1025  # sea water density (kg/m^3)


def runup(Hm0, Tp, slp, **kwargs):
    """
    Calculates run-up height.
    Find 2% run-up height for <type> structure using the EurOtop (2007) manual methods.

    Parameters
    ----------
    Hm0 : float
        Significant wave height at structure toe (m) .
    Tp : float
        Peak wave period (s).
    slp : float
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
    -------
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
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    Lm10 = g * (Tp ** 2) / (2 * math.pi)  # Deep water wave length
    Sm10 = Hm0 / Lm10  # Wave steepness
    Em10 = math.tan(slp) / Sm10 ** 0.5  # Breaker type
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
            ru = Hm0 * 1 * Yb * Yfsurg * YB * (4.3 - 1.6 / (Em10 ** 0.5))
            ru = min(ru, Hm0 * 2.11)  # Maximum for permeable core
            return ru
        elif dmethod is 'prob':
            ru = Hm0 * 1 * Yb * Yfsurg * YB * (4 - 1.5 / (Em10 ** 0.5))
            ru = min(ru, Hm0 * 1.97)  # Maximum for permeable core
            return ru
        else:
            raise ValueError('ERROR: Design method not recognized')
    else:
        raise ValueError('ERROR: Structure type not recognized')


def overtopping(Hm0, Rc, **kwargs):
    """
    Calculates mean overtopping discharge.
    Find mean overtopping discharge for <type> structure using the EurOtop (2007) manual methods.

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

    Returns
    -------
    q : float
        Estimated mean overtopping discharge (m^2/s) for parameters entered.
    """
    B = kwargs.pop('B', 0)
    Yf = kwargs.pop('Yf', 1)
    strtype = kwargs.pop('strtype', 'sap')
    dmethod = kwargs.pop('dmethod', 'det')
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    if B < 80:
        YB = 1 - 0.0033 * B
    else:
        YB = 0.736

    if strtype is 'sap':
        if dmethod is 'det':
            q = ((g * (Hm0 ** 3)) ** 0.5) * 0.2 * math.exp(-2.3 * Rc / (Hm0 * Yf * YB))
            return q
        elif dmethod is 'prob':
            q = ((g * (Hm0 ** 3)) ** 0.5) * 0.2 * math.exp(-2.6 * Rc / (Hm0 * Yf * YB))
            return q
        else:
            raise ValueError('ERROR: Design method not recognized')
    else:
        raise ValueError('ERROR: Structure type not recognized')


def hudson(Hs, alfa, rock_density, **kwargs):
    """
    Solves Hudson equation for median rock diameter. Checks stability (Ns < 2)

    Paramters
    ---------
    Hs : float
        Significant wave height at structure's toe (m)
    alfa : float
        Structure angle (degrees to horizontal)
    rock_density : float
        Rock density (kg/m^3)
    Kd : float
        Dimensionless stability coefficient (3 for quarry rock (default), 10 for concrete blocks)

    Returns
    -------
    Dn50 : float
        Nominal median diameter of armour blocks (m)
    """
    kd = kwargs.pop('kd', 3)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    delta = rock_density / sea_water_density - 1

    def cot(x):
        return 1 / math.tan(x)

    def rad(x):
        return x * math.pi / 180

    Dn50 = (Hs * 1.27) / (((kd * cot(rad(alfa))) ** (1 / 3)) * delta)
    Ns = Hs / (delta * Dn50)
    if Ns > 2:
        warnings.warn('Armour is not stable with the stability number Ns={0}, Dn50={1} [m]'.
                      format(round(Ns, 2), round(Dn50, 2)))
    return Dn50


def goda_1974(Hs, hs, T, d, hc, hw, **kwargs):
    """
    Calculates wave load on vertical wall according to Goda (1974) formula
    (Coastal Engineering Manual, VI-5-154)

    Parameters
    ----------
    Hs : float
        Significant wave height (m)
    hs : float
        Water depth at structure toe (m)
    T : float
        Wave period (s)
    d : float
        Water depth at the wall (m)
    hc : float
        Freeboard (m)
    hw : float
        Vertical wall height (m)
    angle : float (optional)
        Angle of wave attack (degrees, 0 - normal to structure)
    l_1, .._2, .._3 : float (optional)
        Modification factors (tables in CEM)
    hb : float (optional)
        Water depth at distance 5Hs seaard from the structure
    h_design : float (optional)
        Design wave height = highest of the random breaking
        waves at a distance 5Hs seaward of the structure
        (if structure is located within the surf zone)

    Returns
    -------
    A dictionary with: total wave load (N/m), three horizontal pressure components (Pa),
    vertical pressure component (Pa)
    """
    angle = kwargs.pop('angle', 0)
    l_1 = kwargs.pop('l_1', 1)
    l_2 = kwargs.pop('l_2', 1)
    l_3 = kwargs.pop('l_3', 1)
    h_design = kwargs.pop('h_design', None)
    hb = kwargs.pop('hb', hs)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    def deg2rad(x):
        return x * math.pi / 180

    if h_design is None:
        h_design = 1.8 * Hs
    wave = lw(T, Hs, depth=hs)
    a_1 = 0.6 + 0.5 * (((4 * math.pi * hs / wave.L) / (math.sinh(4 * math.pi * hs / wave.L))) ** 2)
    a_2 = min(
        (((hb - d) / (3 * hb)) * ((h_design / d) ** 2)),
        ((2 * d) / h_design)
    )
    a_3 = 1 - ((hw - hc) / hs) * (1 - 1 / math.cosh(2 * math.pi * hs / wave.L))
    a_star = a_2
    s = 0.75 * (1 + math.cos(deg2rad(angle))) * l_1 * h_design
    p1 = 0.5 * (1 + math.cos(deg2rad(angle))) * (l_1 * a_1 + l_2 * a_star *
                                                 (math.cos(deg2rad(angle)) ** 2)) * sea_water_density * g * h_design
    if s > hc:
        p2 = (1 - hc / s) * p1
    else:
        p2 = 0
    p3 = a_3 * p1
    pu = 0.5 * (1 + math.cos(deg2rad(angle))) * l_3 * a_1 * a_3 * sea_water_density * g * h_design
    if s > hc:
        load_aw = hc * p2 + hc * (p1 - p2) * 0.5
    else:
        load_aw = p1 * s * 0.5
    load_uw = (hw - hc) * p3 + (hw - hc) * (p1 - p3) * 0.5
    load = load_aw + load_uw
    return {
        'Total wave load (N/m)': load,
        'Total wave load (lbf/ft)': load * 0.3048 / 4.4482216152605,
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'pu': pu
    }


def goda_2000(H13, T13, h, hc, **kwargs):
    """
    Calculates wave load on vertical wall according to Goda (2000) formula
    (Random seas and design of maritime structures, p.134 - p.139)

    Parameters
    ----------
    H13 : float
        Significant wave height (m)
    h : float
        Water depth at structure toe (m)
    T13 : float
        Wave period (s)
    d : float
        Water depth at the wall (m)
    hc : float
        Freeboard (m)
    h_prime : float
        Vertical wall submerged height (m)
    angle : float (optional)
        Angle of wave attack (degrees, 0 - normal to structure)
    hb : float (optional)
        Water depth at distance 5H13 seaward from the structure
    Hmax : float (optional)
        Design wave height = highest of the random breaking
        waves at a distance 5H13 seaward of the structure
        (if structure is located within the surf zone)

    Returns
    -------
    A pandas dataframe with pressures, total load, load centroid (above wall footing, i.e. depth d)
    in both metric and customary units
    """
    d = kwargs.pop('d', h)
    Hmax = kwargs.pop('Hmax', None)
    angle = kwargs.pop('angle', 0)
    hb = kwargs.pop('hb', h)
    h_prime = kwargs.pop('h_prime', d)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    B = angle * math.pi / 180
    if Hmax is None:
        Hmax = 1.8 * H13
    wave = lw(T13, H13, depth=h)
    L = wave.L
    s = 0.75 * (1 + math.cos(B)) * Hmax
    a_1 = 0.6 + 0.5 * (((4 * math.pi * h / L) / (math.sinh(4 * math.pi * h / L))) ** 2)
    a_2 = min(
        (((hb - d) / (3 * hb)) * ((Hmax / d) ** 2)),
        ((2 * d) / Hmax)
    )
    a_3 = 1 - (h_prime / h) * (1 - (1 / math.cosh(2 * math.pi * h / L)))
    p1 = 0.5 * (1 + math.cos(B)) * (a_1 + a_2 * (math.cos(B) ** 2)) * sea_water_density * g * Hmax
    p2 = p1 / math.cosh(2 * math.pi * h / L)
    p3 = a_3 * p1
    if s > hc:
        p4 = p1 * (1 - hc / s)
    else:
        p4 = 0
    hc_star = min(s, hc)
    pu = 0.5 * (1 + math.cos(B)) * a_1 * a_3 * sea_water_density * g * Hmax
    P = 0.5 * (p1 + p3) * h_prime + 0.5 * (p1 + p4) * hc_star
    Mp = (1 / 6) * (2 * p1 + p3) * (h_prime ** 2) + 0.5 * (p1 + p4) * h_prime * hc_star +\
         (1 / 6) * (p1 + 2 * p4) * (hc_star ** 2)
    P_centroid = Mp / P
    output = pd.DataFrame(data=[
        round(P, 3),
        round(P * 0.3048 * 0.224808943871, 3),
        round(P_centroid, 3),
        round(P_centroid / 0.3048, 3),
        round(hc_star, 3),
        round(hc_star / 0.3048, 3),
        round(p1, 3),
        round(p1 / 6894.75729, 3),
        round(p2, 3),
        round(p2 / 6894.75729, 3),
        round(p3, 3),
        round(p3 / 6894.75729, 3),
        round(p4, 3),
        round(p4 / 6894.75729, 3),
        round(pu, 3),
        round(pu / 6894.75729, 3),
        round(a_1, 3),
        round(a_2, 3),
        round(a_3, 3),
        round(s, 3),
        round(s / 0.3048, 3)
    ],
        index=[
            'Total wave load [N/m]',
            'Total wave load [lbf/ft]',
            'Load centroid [m]',
            'Load centroid [ft]',
            'hc_star [m]',
            'hc_star [ft]',
            'p1 [Pa]',
            'p1 [psi]',
            'p2 [Pa]',
            'p2 [psi]',
            'p3 [Pa]',
            'p3 [psi]',
            'p4 [Pa]',
            'p4 [psi]',
            'pu [Pa]',
            'pu [psi]',
            'a_1',
            'a_2',
            'a_3',
            'Wave reach [m]',
            'Wave reach [ft]'
            ],
        columns=[
            'Value'
        ]
    )
    return output


def dAngremond(Rc, Hm0, B, Tp, tana):
    """
    Calculates wave transmission coefficient for emerged breakwaters
    using the dAngremond (1996) (EurOtop 2016) formula

    :param Rc: freeboard [m]
    :param Hm0: incident significant wave height [m]
    :param B: crest width [m]
    :param Tp: wave period [sec]
    :param tana: seaward breakwater slope
    :return: wave transmission coefficient
    """
    Sop = 2 * math.pi * Hm0 / (scipy.constants.g * Tp ** 2)
    Eop = tana / Sop ** 0.5
    Kt_small = -0.4 * Rc / Hm0 + 0.64 * (B / Hm0) ** (-0.31) * (1 - math.exp(-0.5 * Eop))
    Kt_large = -0.35 * Rc / Hm0 + 0.51 * (B / Hm0) ** (-0.65) * (1 - math.exp(-0.41 * Eop))
    scale = B / Hm0

    if scale < 8:
        Kt = Kt_small
    elif scale > 12:
        Kt = Kt_large
    else:
        Kt = (Kt_large - Kt_small) * (scale - 8) / (12 - 8) + Kt_small # linear interpolation

    if Kt >= 0.8:
        return 1
    elif Kt <= 0.075:
        return 0.075
    else:
        return Kt


def Seabrook(Hm0, ds, B, L, D50):
    """
    Calculates wave transmission coefficient for submerged breakwaters
    using the Seabrook & Hall (1998) formula

    :param Hm0: incident significant wave height [m]
    :param ds: depth of submergence (positive value of the negative freeboard) [m]
    :param B: crest width [m]
    :param L: wave length [m]
    :param D50: rock armor median rock size [m]
    :return: wave transmission coefficient
    """
    Kt = 1 - (math.exp(-0.65 * ds / Hm0 - 1.09 * Hm0 / B) + 0.047 * B * ds / (L * D50) - 0.067 * ds * Hm0 / (B * D50))
    condition_1 = B * ds / (L * D50)
    condition_2 = ds * Hm0 / (B * D50)
    if 0 <= condition_1 <= 7.08 and 0 <= condition_2 <= 2.14:
        return Kt
    else:
        return 1
