from math import pi, tan, exp, cos, sinh, cosh
from coastlib.models.linear_wave_theory import LinearWave as lw
import scipy.constants

g = scipy.constants.g  # gravity constant (m/s^2) as defined by ISO 80000-3
swd = 1025  # sea water density (kg/m^3)


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
    assert len(kwargs) == 0, "unrecognized arguments passed in: %s" % ", ".join(kwargs.keys())

    Lm10 = g * (Tp ** 2) / (2 * pi)  # Deep water wave length
    Sm10 = Hm0 / Lm10  # Wave steepness
    Em10 = tan(slp) / Sm10 ** 0.5  # Breaker type
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
    assert len(kwargs) == 0, "unrecognized arguments passed in: %s" % ", ".join(kwargs.keys())

    if B < 80:
        YB = 1 - 0.0033 * B
    else:
        YB = 0.736

    if strtype is 'sap':
        if dmethod is 'det':
            q = ((g * (Hm0 ** 3)) ** 0.5) * 0.2 * exp(-2.3 * Rc / (Hm0 * Yf * YB))
            return q
        elif dmethod is 'prob':
            q = ((g * (Hm0 ** 3)) ** 0.5) * 0.2 * exp(-2.6 * Rc / (Hm0 * Yf * YB))
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

    delta = rock_density / swd - 1

    def cot(x):
        return 1 / tan(x)

    Dn50 = (Hs * 1.27) / (((kd * cot(alfa * pi / 180)) ** (1 / 3)) * delta)
    Ns = Hs / (delta * Dn50)
    if Ns > 2:
        print('Armour is not stable with the stability number Ns={0}'.format(round(Ns, 2)))
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
        return x * pi / 180

    if h_design is None:
        h_design = 1.8 * Hs
    wave = lw(T, Hs, depth=hs)
    a_1 = 0.6 + 0.5 * (((4 * pi * hs / wave.L) / (sinh(4 * pi * hs / wave.L))) ** 2)
    a_2 = min(
        (((hb - d) / (3 * hb)) * ((h_design / d) ** 2)),
        ((2 * d) / h_design)
    )
    a_3 = 1 - ((hw - hc) / hs) * (1 - 1 / cosh(2 * pi * hs / wave.L))
    a_star = a_2
    s = 0.75 * (1 + cos(deg2rad(angle))) * l_1 * h_design
    p1 = 0.5 * (1 + cos(deg2rad(angle))) * (l_1 * a_1 + l_2 * a_star * (cos(deg2rad(angle)) ** 2)) * swd * g * h_design
    if s > hc:
        p2 = (1 - hc / s) * p1
    else:
        p2 = 0
    p3 = a_3 * p1
    pu = 0.5 * (1 + cos(deg2rad(angle))) * l_3 * a_1 * a_3 * swd * g * h_design
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
    A dictionary with: total wave load (N/m), wave load application depth (m),
    three horizontal pressure components (Pa), vertical pressure component (Pa)
    """
    d = kwargs.pop('d', h)
    Hmax = kwargs.pop('Hmax', None)
    angle = kwargs.pop('angle', 0)
    hb = kwargs.pop('hb', h)
    h_prime = kwargs.pop('h_prime', d)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    B = angle * pi / 180
    if Hmax is None:
        Hmax = 1.8 * H13
    wave = lw(T13, H13, depth=h)
    L = wave.L
    s = 0.75 * (1 + cos(B)) * Hmax
    a_1 = 0.6 + 0.5 * (((4 * pi * h / L) / (sinh(4 * pi * h / L))) ** 2)
    a_2 = min(
        (((hb - d) / (3 * hb)) * ((Hmax / d) ** 2)),
        ((2 * d) / Hmax)
    )
    a_3 = 1 - (h_prime / h) * (1 - (1 / cosh(2 * pi * h / L)))
    p1 = 0.5 * (1 + cos(B)) * (a_1 + a_2 * (cos(B) ** 2)) * swd * g * Hmax
    p2 = p1 / cosh(2 * pi * h / L)
    p3 = a_3 * p1
    if s > hc:
        p4 = p1 * (1 - hc / s)
    else:
        p4 = 0
    hc_star = min(s, hc)
    pu = 0.5 * (1 + cos(B)) * a_1 * a_3 * swd * g * Hmax
    P = 0.5 * (p1 + p3) * h_prime + 0.5 * (p1 + p4) * hc_star
    Mp = (1 / 6) * (2 * p1 + p3) * (h_prime ** 2) + 0.5 * (p1 + p4) * h_prime * hc_star +\
         (1 / 6) * (p1 + 2 * p4) * (hc_star ** 2)
    P_centroid = (p4 * hc_star) * (hc_star / 2) + (p3 * h_prime) * ((-1) * h_prime / 2) +\
                 (0.5 * (p1 - p4) * hc_star) * (hc_star / 3) + (0.5 * (p1 - p3) * h_prime) * ((-1) * h_prime / 3)
    P_centroid /= P
    return {
        'Total wave load [N/m]': P,
        'Total wave load [lbf/ft]': P * 0.3048 / 4.4482216152605,
        'Load centroid [m]': P_centroid,
        'Load centroid [ft]': P_centroid / 0.3048,
        'p4 extent [m]': hc_star,
        'p4 extent [ft]': hc_star / 0.3048,
        'p1, [Pa]': p1,
        'p2, [Pa]': p2,
        'p3, [Pa]': p3,
        'p4, [Pa]': p4,
        'pu, [Pa]': pu
    }
