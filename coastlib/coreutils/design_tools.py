from numpy import tanh, tan, exp
from math import pi
import scipy.constants
from scipy.optimize import newton

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

    B = kwargs.get('B', 0)
    Yf = kwargs.get('Yf', 1)
    rB = kwargs.get('rB', 0)
    Lb = kwargs.get('Lb', 1)
    rdb = kwargs.get('rdb', 0)
    strtype = kwargs.get('strtype', 'sap')
    dmethod = kwargs.get('dmethod', 'det')

    Lm10 = g*(Tp**2)/(2*pi)  # Deep water wave length
    Sm10 = Hm0/Lm10  # Wave steepness
    Em10 = tan(slp)/Sm10**0.5  # Breaker type
    rB /= Lb
    Yb = 1-rB*(1-rdb)  # Berm factor (1 for no berm)

    if B < 80:
        YB = 1-0.0022*B
    else:
        YB = 0.824
    if Em10 > 10:
        Yfsurg = 1
    else:
        Yfsurg = Yf+(Em10-1.8)*(1-Yf)/8.2

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
    B = kwargs.get('B', 0)
    Yf = kwargs.get('Yf', 1)
    strtype = kwargs.get('strtype', 'sap')
    dmethod = kwargs.get('dmethod', 'det')

    if B < 80:
        YB = 1 - 0.0033 * B
    else:
        YB = 0.736

    if strtype is 'sap':
        if dmethod is 'det':
            q = ((g*(Hm0**3))**0.5)*0.2*exp(-2.3*Rc/(Hm0*Yf*YB))
            return q
        elif dmethod is 'prob':
            q = ((g * (Hm0 ** 3)) ** 0.5) * 0.2 * exp(-2.6 * Rc / (Hm0 * Yf * YB))
            return q
        else:
            raise ValueError('ERROR: Design method not recognized')
    else:
        raise ValueError('ERROR: Structure type not recognized')