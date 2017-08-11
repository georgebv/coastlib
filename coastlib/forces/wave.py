import numpy as np
import pandas as pd
import scipy.constants

from coastlib.wavemodels.fenton import FentonWave


def drag_force(velocity, drag_coefficient, face_area, rho=1025):
    # https://en.wikipedia.org/wiki/Drag_equation
    # https://en.wikipedia.org/wiki/Drag_coefficient
    # <face_area> - unit area normal to flow
    return (1 / 2) * rho * velocity * np.abs(velocity) * drag_coefficient * face_area


def inertia_force(acceleration, volume, inertia_coefficient, rho=1025):
    # https://en.wikipedia.org/wiki/Froude%E2%80%93Krylov_force
    # hydrodynamic mass force
    return rho * inertia_coefficient * volume * acceleration


class Morrison:
    # https://en.wikipedia.org/wiki/Morison_equation

    def __init__(self, wave_height, wave_period, depth, **kwargs):

        # Parse general inputs
        self.wave_height = wave_height  # up to user to provide the 1.8Hs value
        self.wave_period = wave_period
        self.depth = depth
        self.current_velocity = kwargs.pop('current_velocity', 0)
        self._rho = kwargs.pop('rho', 1025)
        self._g = kwargs.pop('g', scipy.constants.g)
        self.fenton_wave = FentonWave(
            wave_heigh=wave_height, wave_period=wave_period, depth=depth,
            current_criterion=1, current_velocity=self.current_velocity,
            rho=self._rho, g=self._g, points=dict(m=100, ua=100, vert=1000)
        )  # 1000 points per vertical profile for extra accuracy
        self.type = kwargs.pop('element_type', 'vertical cylinder')

        # Parse phases
        self._x = np.unique(self.fenton_wave.flowfield['X (m)'].values)

        if self.type == 'vertical cylinder':
            # Parse type-specific inputs
            self.cylinder_diameter = kwargs.pop('cylinder_diameter', None)
            self.cylinder_top = kwargs.pop('cylinder_top', None)  # Relative to seabed
            self.cylinder_bottom = kwargs.pop('cylinder bottom', None)  # Relative to seabed
            self.drag_coefficient = kwargs.pop('drag_coefficient', 1.2)
            self.inertia_coefficient = kwargs.pop('inertia_coefficient', 2)
            # Calculate wave force on the cylinder and store results
            self.__vertical_cylinder()
        elif self.type == 'sloped cylinder':
            # Parse type-specific inputs
            self.cylinder_diameter = kwargs.pop('cylinder_diameter', None)
            self.cylinder_top = kwargs.pop('cylinder_top', None)  # Relative to seabed
            self.cylinder_bottom = kwargs.pop('cylinder bottom', None)  # Relative to seabed
            self.drag_coefficient = kwargs.pop('drag_coefficient', 1.2)
            self.inertia_coefficient = kwargs.pop('inertia_coefficient', 2)
            self.slope = kwargs.pop('slope', 3)  # Slope V:H
            # TODO - inclined cylinder
            pass
        else:
            raise ValueError('Type {} is not supported'.format(self.type))

        assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    def __vertical_cylinder(self):
        forces, moments, centroids = [], [], []
        for i, x in enumerate(self._x):
            flowfield = self.fenton_wave.flowfield[
                (self.fenton_wave.flowfield['X (m)'] == x) &
                (self.fenton_wave.flowfield['Y (m)'] >= self.cylinder_bottom) &
                (self.fenton_wave.flowfield['Y (m)'] <= self.cylinder_top)
            ]
            # integration slice heights
            dz = flowfield['Y (m)'].values[1:] - flowfield['Y (m)'].values[:-1]
            # average <u> for each slice
            u = (flowfield['u (m/s)'].values[1:] + flowfield['u (m/s)'].values[:-1]) / 2
            # average <ua> for each slice
            ua = (flowfield['ua (m/s)'].values[1:] + flowfield['ua (m/s)'].values[:-1]) / 2
            # center coordinates for each slice (from seabed to slice center)
            y = (flowfield['Y (m)'].values[1:] + flowfield['Y (m)'].values[:-1]) / 2

            # Wave force
            force = drag_force(
                velocity=u, drag_coefficient=self.drag_coefficient,
                face_area=self.cylinder_diameter*dz, rho=self._rho
                ) +\
                inertia_force(
                    acceleration=ua, volume=np.pi*self.cylinder_diameter*dz,
                    inertia_coefficient=self.inertia_coefficient, rho=self._rho
                )
            forces.extend([sum(force)])
            moment = force * y
            moments.extend([sum(moment)])
            centroids.extend(([sum(moment) / sum(force)]))
        self.force = pd.DataFrame(data=self._x, columns=['X (m)'])
        self.force['F (N)'] = forces
        self.force['M (N-m)'] = moments
        self.force['Centroid (m from seabed)'] = centroids


# TODO - below are not implemented


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
    # angle = kwargs.pop('angle', 0)
    # l_1 = kwargs.pop('l_1', 1)
    # l_2 = kwargs.pop('l_2', 1)
    # l_3 = kwargs.pop('l_3', 1)
    # h_design = kwargs.pop('h_design', None)
    # hb = kwargs.pop('hb', hs)
    # g = kwargs.pop('g', scipy.constants.g)
    # sea_water_density = kwargs.pop('sea_water_density', 1030)
    # assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
    #
    # def deg2rad(x):
    #     return x * math.pi / 180
    #
    # if h_design is None:
    #     h_design = 1.8 * Hs
    # wave = AiryWave(T, Hs, depth=hs)
    # a_1 = 0.6 + 0.5 * (((4 * math.pi * hs / wave.L) / (math.sinh(4 * math.pi * hs / wave.L))) ** 2)
    # a_2 = min(
    #     (((hb - d) / (3 * hb)) * ((h_design / d) ** 2)),
    #     ((2 * d) / h_design)
    # )
    # a_3 = 1 - ((hw - hc) / hs) * (1 - 1 / math.cosh(2 * math.pi * hs / wave.L))
    # a_star = a_2
    # s = 0.75 * (1 + math.cos(deg2rad(angle))) * l_1 * h_design
    # p1 = 0.5 * (1 + math.cos(deg2rad(angle))) * (l_1 * a_1 + l_2 * a_star *
    #                                              (math.cos(deg2rad(angle)) ** 2)) * sea_water_density * g * h_design
    # if s > hc:
    #     p2 = (1 - hc / s) * p1
    # else:
    #     p2 = 0
    # p3 = a_3 * p1
    # pu = 0.5 * (1 + math.cos(deg2rad(angle))) * l_3 * a_1 * a_3 * sea_water_density * g * h_design
    # if s > hc:
    #     load_aw = hc * p2 + hc * (p1 - p2) * 0.5
    # else:
    #     load_aw = p1 * s * 0.5
    # load_uw = (hw - hc) * p3 + (hw - hc) * (p1 - p3) * 0.5
    # load = load_aw + load_uw
    # return {
    #     'Total wave load (N/m)': load,
    #     'Total wave load (lbf/ft)': load * 0.3048 / 4.4482216152605,
    #     'p1': p1,
    #     'p2': p2,
    #     'p3': p3,
    #     'pu': pu
    # }


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
    # d = kwargs.pop('d', h)
    # Hmax = kwargs.pop('Hmax', None)
    # angle = kwargs.pop('angle', 0)
    # hb = kwargs.pop('hb', h)
    # h_prime = kwargs.pop('h_prime', d)
    # g = kwargs.pop('g', scipy.constants.g)
    # sea_water_density = kwargs.pop('sea_water_density', 1030)
    # assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))
    #
    # B = np.deg2rad(angle)
    # if Hmax is None:
    #     Hmax = 1.8 * H13
    # wave = AiryWave(T13, H13, depth=h)
    # L = wave.L
    # s = 0.75 * (1 + math.cos(B)) * Hmax
    # a_1 = 0.6 + 0.5 * (((4 * math.pi * h / L) / (math.sinh(4 * math.pi * h / L))) ** 2)
    # a_2 = min(
    #     (((hb - d) / (3 * hb)) * ((Hmax / d) ** 2)),
    #     ((2 * d) / Hmax)
    # )
    # a_3 = 1 - (h_prime / h) * (1 - (1 / math.cosh(2 * math.pi * h / L)))
    # p1 = 0.5 * (1 + math.cos(B)) * (a_1 + a_2 * (math.cos(B) ** 2)) * sea_water_density * g * Hmax
    # p2 = p1 / math.cosh(2 * math.pi * h / L)
    # p3 = a_3 * p1
    # if s > hc:
    #     p4 = p1 * (1 - hc / s)
    # else:
    #     p4 = 0
    # hc_star = min(s, hc)
    # pu = 0.5 * (1 + math.cos(B)) * a_1 * a_3 * sea_water_density * g * Hmax
    # P = 0.5 * (p1 + p3) * h_prime + 0.5 * (p1 + p4) * hc_star
    # Mp = (1 / 6) * (2 * p1 + p3) * (h_prime ** 2) + 0.5 * (p1 + p4) * h_prime * hc_star +\
    #      (1 / 6) * (p1 + 2 * p4) * (hc_star ** 2)
    # P_centroid = Mp / P
    # output = pd.DataFrame(data=[
    #     round(P, 3),
    #     round(P * 0.3048 * 0.224808943871, 3),
    #     round(P_centroid, 3),
    #     round(P_centroid / 0.3048, 3),
    #     round(hc_star, 3),
    #     round(hc_star / 0.3048, 3),
    #     round(p1, 3),
    #     round(p1 / 6894.75729, 3),
    #     round(p2, 3),
    #     round(p2 / 6894.75729, 3),
    #     round(p3, 3),
    #     round(p3 / 6894.75729, 3),
    #     round(p4, 3),
    #     round(p4 / 6894.75729, 3),
    #     round(pu, 3),
    #     round(pu / 6894.75729, 3),
    #     round(a_1, 3),
    #     round(a_2, 3),
    #     round(a_3, 3),
    #     round(s, 3),
    #     round(s / 0.3048, 3)
    # ],
    #     index=[
    #         'Total wave load [N/m]',
    #         'Total wave load [lbf/ft]',
    #         'Load centroid [m]',
    #         'Load centroid [ft]',
    #         'hc_star [m]',
    #         'hc_star [ft]',
    #         'p1 [Pa]',
    #         'p1 [psi]',
    #         'p2 [Pa]',
    #         'p2 [psi]',
    #         'p3 [Pa]',
    #         'p3 [psi]',
    #         'p4 [Pa]',
    #         'p4 [psi]',
    #         'pu [Pa]',
    #         'pu [psi]',
    #         'a_1',
    #         'a_2',
    #         'a_3',
    #         'Wave reach [m]',
    #         'Wave reach [ft]'
    #         ],
    #     columns=[
    #         'Value'
    #     ]
    # )
    # return output
