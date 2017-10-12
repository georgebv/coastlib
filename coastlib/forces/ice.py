import numpy as np
from scipy.special import kei, ker, keip, kerp
import scipy.constants


def vertical_ice_load(d, h, e, nu, delta, sigma_b, tau_b, **kwargs):
    """
    Computes the maximum vertical force on an ice sheet (kN) following Nakazawa et al (1988).
    Flowchart in Fig.6

    Mandatory inputs
    ================
    d : float
        Structure diameter (m)
    h : float
        Thickness of ice (m)
    e : float
        Young's modulus of ice (Pa)
    nu : float
        Poisson's ratio of ice (-)
    delta : float
        Change in water level (m)
    sigma_b : float
        Bending strength of ice (Pa)
    tau_b : float
        Adfreeze bond strengtg of ice to structure (Pa)

    Optional inputs
    ===============
    g : float
        Gravity acceleration (m/s^2). Default = scipy.constants.g ~ 9.806 m/s^2
    rho_ice : float
        Sea water ice density (kg/m^3). Default = 916.7 kg/m^3
    units : str
        Output units ('kip' or 'kN'). Default = 'kip'

    Returns
    =======
    p : float
        Vertical ice force (units)
    """

    g = kwargs.pop('g', scipy.constants.g)
    rho_ice = kwargs.pop('rho_ice', 916.7)
    units = kwargs.pop('units', 'kip')
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    # Calculate input parameters
    a = d / 2
    K = e / (1 - nu ** 2)
    D = K * h ** 3 / 12
    omega_0 = rho_ice * g
    lam = (omega_0 / D) ** .25
    k_1 = kei(lam * a) * kerp(lam * a) - keip(lam * a) * ker(lam * a)

    # Adfreeze failure (Equation 2)
    p_adfreeze = 2 * np.pi * a * h * tau_b

    # Assume elastic plate (Equation 1)
    p_elastic = 2 * np.pi * a * D * lam ** 3 * delta * (keip(lam * a) ** 2 + kerp(lam * a) ** 2) / k_1

    # Bending failure (Equation 7)
    k_2 = kei(lam * a) ** 2 + ker(lam * a) ** 2
    p_bending = np.pi * a * lam * h ** 2 * sigma_b / 3 * (kerp(lam * a) ** 2 + keip(lam * a) ** 2) / k_2

    if units == 'kip':
        return {
            'Adfreeze (kip)': p_adfreeze / 10 ** 3 / 4.448222,
            'Elastic (kip)': p_elastic / 10 ** 3 / 4.448222,
            'Bending (kip)': p_bending / 10 ** 3 / 4.448222
        }
    elif units == 'kN':
        return {
            'Adfreeze (kN)': p_adfreeze / 10 ** 3,
            'Elastic (kN)': p_elastic / 10 ** 3,
            'Bending (kN)': p_bending / 10 ** 3
        }
    else:
        raise ValueError(f'Units {units} not recognized')


if __name__ == '__main__':
    p = vertical_ice_load(
        d=1.2 / .3048, h=0.9, rho_ice=940, e=30000 * scipy.constants.g * 10000, nu=.4, delta=1.2 / .3048,
        sigma_b=100 * 6894.757, tau_b=.3e6, units='kip'
    )
    print(p)
