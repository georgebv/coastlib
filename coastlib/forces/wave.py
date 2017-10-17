import numpy as np
import pandas as pd
import scipy.constants

from coastlib.wavemodels.fenton import FentonWave
from coastlib.wavemodels.airy import AiryWave


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
        self.theory = kwargs.pop('theory', 'Fenton')
        self.wave_height = wave_height  # up to user to provide the 1.8Hs value
        self.wave_period = wave_period
        self.depth = depth
        self.current_velocity = kwargs.pop('current_velocity', 0)
        self._rho = kwargs.pop('rho', 1025)
        self._g = kwargs.pop('g', scipy.constants.g)
        self.type = kwargs.pop('element_type', 'vertical cylinder')

        if self.theory == 'Fenton':
            fourier_components = kwargs.pop('fourier_components', 20)
            points = kwargs.pop('points', dict(m=100, ua=100, vert=100))
            convergence = kwargs.pop('convergence', dict(
                maximum_number_of_iterations=40,
                criterion_for_convergence='1.e-4'
            ))
            self.fenton_wave = FentonWave(
                wave_height=wave_height, wave_period=wave_period, depth=depth,
                current_criterion=1, current_velocity=self.current_velocity,
                rho=self._rho, g=self._g, points=points, convergence=convergence,
                fourier_components=fourier_components
            )
            self._x = np.unique(self.fenton_wave.flowfield['X (m)'].values)

        elif self.theory == 'Airy':
            self.airy_wave = AiryWave(
                wave_height=wave_height, wave_period=wave_period, depth=depth, rho=self._rho
            )
            self._dz = kwargs.pop('dz', 0.05)
            self._dx = kwargs.pop('dx', 0.05)
            self._x = np.arange(0, self.airy_wave.L+self._dx, self._dx)

        else:
            raise ValueError(f'Theory {self.theory} not recognized')

        if self.type == 'vertical cylinder':

            # Parse type-specific inputs
            self.cylinder_diameter = kwargs.pop('cylinder_diameter', None)
            self.cylinder_top = kwargs.pop('cylinder_top', None)  # Relative to seabed
            self.cylinder_bottom = kwargs.pop('cylinder_bottom', None)  # Relative to seabed
            self.drag_coefficient = kwargs.pop('drag_coefficient', 1.2)
            self.inertia_coefficient = kwargs.pop('inertia_coefficient', 2)
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            # Calculate wave force on the cylinder and store results
            self.__vertical_cylinder()

        elif self.type == 'sloped cylinder':

            # Parse type-specific inputs
            self.cylinder_diameter = kwargs.pop('cylinder_diameter', None)
            self.cylinder_top = kwargs.pop('cylinder_top', None)  # Relative to seabed
            self.cylinder_bottom = kwargs.pop('cylinder_bottom', None)  # Relative to seabed
            self.drag_coefficient = kwargs.pop('drag_coefficient', 1.2)
            self.inertia_coefficient = kwargs.pop('inertia_coefficient', 2)
            self.slope = kwargs.pop('slope', None)  # Slope V:H
            if not self.slope:
                raise ValueError('<slope> is a mandatory parameter for sloped cylinder')
            assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

            # Calculate wave force on the sloped cylinder
            self.__sloped_cyliner()

        else:
            raise ValueError('Type {} is not supported'.format(self.type))

    def __repr__(self):
        return 'Morrison(wave_height={wh:.2f}, wave_period={wp:.2f}, depth={dep:.2f}, ' \
               'current_velocity={cv:.2f}, type={tp})'.format(wh=self.wave_height,
                                                              wp=self.wave_period,
                                                              dep=self.depth,
                                                              cv=self.current_velocity,
                                                              tp=self.type)

    def __vertical_cylinder(self):

        if self.theory == 'Fenton':
            forces, moments, centroids = [], [], []
            for i, x in enumerate(self._x):
                flowfield = self.fenton_wave.flowfield[
                    (self.fenton_wave.flowfield['X (m)'] == x) &
                    (self.fenton_wave.flowfield['Y (m)'] >= self.cylinder_bottom) &
                    (self.fenton_wave.flowfield['Y (m)'] <= self.cylinder_top)
                    ]
                if len(flowfield) > 1:
                    # integration slice heights
                    dz = flowfield['Y (m)'].values[1:] - flowfield['Y (m)'].values[:-1]
                    # average <u> for each slice
                    u = (flowfield['u (m/s)'].values[1:] + flowfield['u (m/s)'].values[:-1]) / 2
                    # average <du/dt> for each slice
                    ua = (flowfield['du/dt (m/s^2)'].values[1:] + flowfield['du/dt (m/s^2)'].values[:-1]) / 2
                    # center coordinates for each slice (from seabed to slice center)
                    y = (flowfield['Y (m)'].values[1:] + flowfield['Y (m)'].values[:-1]) / 2
                elif len(flowfield) == 1:
                    # integration slice heights
                    dz = np.mean((self.fenton_wave.flowfield['Y (m)'].values[1:] -
                    self.fenton_wave.flowfield['Y (m)'].values[:-1]) / 2)
                    # average <u> for each slice
                    u = flowfield['u (m/s)'].values
                    # average <du/dt> for each slice
                    ua = flowfield['du/dt (m/s^2)'].values
                    # center coordinates for each slice (from seabed to slice center)
                    y = flowfield['Y (m)'].values
                else:
                    dz, u, ua, y = np.nan, np.nan, np.nan, np.nan

                # Wave force
                force = drag_force(
                    velocity=u, drag_coefficient=self.drag_coefficient,
                    face_area=self.cylinder_diameter * dz, rho=self._rho
                ) + \
                    inertia_force(
                        acceleration=ua, volume=(np.pi / 4) * (self.cylinder_diameter ** 2) * dz,
                        inertia_coefficient=self.inertia_coefficient, rho=self._rho
                    )
                moment = force * y
                forces.append(sum(force))
                moments.append(sum(moment))
                centroids.append(sum(moment) / sum(force))

            self.force = pd.DataFrame(data=self._x, columns=['X (m)'])
            self.force['F (N)'] = forces
            self.force['M (N-m)'] = moments
            self.force['Centroid (m, from seabed)'] = centroids

        elif self.theory == 'Airy':
            forces, moments, centroids = [], [], []
            for i, x in enumerate(self._x):
                u, ua, y = [], [], []
                self.airy_wave.dynprop(z=0, t=0, x=x)
                eta = self.airy_wave.S

                z_range = np.arange(
                    self.cylinder_bottom-self.depth,  # cylinder bottom relative to SWL
                    min(eta+self._dz, self.cylinder_top-self.depth+self._dz),  # min of wave crest and cylinder top relative to SWL
                    self._dz
                )
                for z in z_range:
                    if z <= 0:
                        self.airy_wave.dynprop(z=z, t=0, x=x)
                    else:
                        self.airy_wave.dynprop(z=0, t=0, x=x)
                    u.append(self.airy_wave.u + self.current_velocity)
                    ua.append(self.airy_wave.ua)
                    y.append(self.airy_wave.depth + z)
                u = np.array(u)
                ua = np.array(ua)
                y = np.array(y)
                force = drag_force(
                    velocity=u, drag_coefficient=self.drag_coefficient,
                    face_area=self.cylinder_diameter * self._dz, rho=self._rho
                ) + \
                        inertia_force(
                            acceleration=ua, volume=(np.pi / 4) * (self.cylinder_diameter ** 2) * self._dz,
                            inertia_coefficient=self.inertia_coefficient, rho=self._rho
                        )
                moment = force * y
                forces.append(sum(force))
                moments.append(sum(moment))
                centroids.append(sum(moment) / sum(force))

            self.force = pd.DataFrame(data=self._x, columns=['X (m)'])
            self.force['F (N)'] = forces
            self.force['M (N-m)'] = moments
            self.force['Centroid (m, from seabed)'] = centroids

    def __sloped_cyliner(self):
        # TODO : Airy not implemented
        forces, moments, centroids, moments_x, centroids_x = [], [], [], [], []
        for i, x in enumerate(self._x):
            flowfield = self.fenton_wave.flowfield[
                (self.fenton_wave.flowfield['X (m)'] == x) &
                (self.fenton_wave.flowfield['Y (m)'] >= self.cylinder_bottom) &
                (self.fenton_wave.flowfield['Y (m)'] <= self.cylinder_top)
                ]
            # integration slice heights
            dz = flowfield['Y (m)'].values[1:] - flowfield['Y (m)'].values[:-1]
            dz = np.sqrt(((dz / self.slope) ** 2 + dz ** 2))
            # average <u> for each slice
            u = (flowfield['u (m/s)'].values[1:] + flowfield['u (m/s)'].values[:-1]) / 2
            # average <du/dt> for each slice
            ua = (flowfield['du/dt (m/s^2)'].values[1:] + flowfield['du/dt (m/s^2)'].values[:-1]) / 2
            # center coordinates for each slice (from seabed to slice center)
            y = (flowfield['Y (m)'].values[1:] + flowfield['Y (m)'].values[:-1]) / 2
            _x = y / self.slope

            # Wave force
            force = drag_force(
                velocity=u, drag_coefficient=self.drag_coefficient,
                face_area=self.cylinder_diameter * dz, rho=self._rho
            ) + \
                inertia_force(
                    acceleration=ua, volume=(np.pi / 4) * (self.cylinder_diameter ** 2) * dz,
                    inertia_coefficient=self.inertia_coefficient, rho=self._rho
                )
            moment = force * y
            moment_x = force * _x

            forces.append(sum(force))
            moments.append(sum(moment))
            centroids.append(sum(moment) / sum(force))
            moments_x.append(sum(moment_x))
            centroids_x.append(sum(moment_x) / sum(force))
        self.force = pd.DataFrame(data=self._x, columns=['X (m)'])
        self.force['F (N)'] = forces
        self.force['M (N-m)'] = moments
        self.force['Centroid (m, from seabed)'] = centroids
        self.force['M horizontal (N-m)'] = moments_x
        self.force['Centroid horizontal (m, from seabed end)'] = centroids_x


def mcconnell(wave_height, wave_period, depth, clearance,
              deck_width, deck_length, current_velocity=0, theory='Airy', case=2, **kwargs):
    # TODO : QC
    coefficients = pd.DataFrame(
        data=np.array([
            [0.82, 0.84, 0.71, -0.54, -0.35, -0.12, -0.8],
            [0.61, 0.66, 0.71, 0.91, 1.12, 0.85, 0.34]
        ]).T,
        columns=[
            'a',
            'b'
        ],
        index=[
            'Upward vertical forces (seaward beam and deck)',
            'Upward vertical forces (internal beam only)',
            'Upward vertical forces (internal deck, two- and three-dimensional effects)',
            'Downward vertical forces (seaward beam and deck)',
            'Downward vertical forces (internal beam only)',
            'Downward vertical forces (internal deck, two-dimensional effects)',
            'Downward vertical forces (internal deck, three-dimensional effects)',
        ]
    )

    cs = pd.DataFrame(
        data=np.array([
            [1.5, 1.4, 2.2, 1.6, 1.8, 2.1, 1.4],
            [0.5, 0.5, 0.1, 0.4, 0.5, np.nan, 0.65]
        ]).T,
        columns=[
            'C upper',
            'C lower'
        ],
        index=[
            'Upward vertical forces (seaward beam and deck)',
            'Upward vertical forces (internal beam only)',
            'Upward vertical forces (internal deck, two- and three-dimensional effects)',
            'Downward vertical forces (seaward beam and deck)',
            'Downward vertical forces (internal beam only)',
            'Downward vertical forces (internal deck, two-dimensional effects)',
            'Downward vertical forces (internal deck, three-dimensional effects)',
        ]
    )

    def F_vqs(a, b, eta_max, cl, Hs, C, bw, bl, rho=1025, g=scipy.constants.g):
        # p1 = (eta_max - (bh + cl)) * rho * g
        p2 = (eta_max - cl) * rho * g
        f_star_v = bw * bl * p2
        return f_star_v * a * C / ((eta_max - cl) / Hs) ** b, np.sign(a * C / ((eta_max - cl) / Hs) ** b)

    cases = cs.index
    coefficient = coefficients[coefficients.index == cases[case]].values[0]

    if theory == 'Airy':
        wave = AiryWave(wave_height=wave_height, wave_period=wave_period, depth=depth)
        eta_max = wave_height / 2
    elif theory == 'Fenton':
        fourier_components = kwargs.pop('fourier_components', 20)
        wave = FentonWave(
            wave_height=wave_height, wave_period=wave_period, depth=depth,
            current_criterion=1, current_velocity=current_velocity, fourier_components=fourier_components
        )
        eta_max = wave.surface['eta (m)'].values.max() - depth
    else:
        raise ValueError(f'Theory {theory} not recognized')

    force = [
        F_vqs(
            a=coefficient[0], b=coefficient[1], eta_max=eta_max, cl=clearance,
            Hs=wave_height, C=1, bw=deck_width, bl=deck_length
        ),
        F_vqs(
            a=coefficient[0], b=coefficient[1], eta_max=eta_max, cl=clearance,
            Hs=wave_height, C=cs[cs.index == cases[case]].values[0][1], bw=deck_width, bl=deck_length
        ),
        F_vqs(
            a=coefficient[0], b=coefficient[1], eta_max=eta_max, cl=clearance,
            Hs=wave_height, C=cs[cs.index == cases[case]].values[0][0], bw=deck_width, bl=deck_length
        )
        ]
    return force


def goda_1974(wave_height, wave_period, depth, freeboard, wall_height, angle=0, **kwargs):
    """
    Calculates wave load on vertical wall using the Goda (1974) formula
    (Coastal Engineering Manual, VI-5-154)

    Mandatory inputs
    ================
    wave_heigh : float
        Design wave height (m) (<Hdesign> in CEM notation, up to user to multiply by 1.8 to get Hmax)
    wave_period : float
        Wave period (s)
    depth : float
        Water depth at the wall (m) (<d> in CEM notation)
    freeboard : float
        Freeboard (m)
    wall_height : float
        Vertical wall height (m) (<hw> in CEM notation)

    Optional inputs
    ===============
    hs : float (default=depth)
        Water depth at the toe (m) (<hs> in CEM notation)
    hb : float (default=hs)
        Water depth at distance 5Hs seaard from the structure
    angle : float (default=0)
        Angle of wave attack (degrees, 0 - normal to structure)
    l_1, .._2, .._3 : float (default=1 for all)
        Modification factors (tables in CEM) (<lambda_#> in CEM notation)
    g : float (default=scipy.constants.g=9.81)
        Gravity acceleration (m/s^2)
    sea_water_density : float (default=1025)
        Sea water density (kg/m^3)

    Returns
    =======
    A dataframe with: total wave load (N/m), three horizontal pressure components (Pa),
    and a vertical pressure component (Pa)
    """

    l_1 = kwargs.pop('l_1', 1)
    l_2 = kwargs.pop('l_2', 1)
    l_3 = kwargs.pop('l_3', 1)
    hs = kwargs.pop('hs', depth)
    hb = kwargs.pop('hb', hs)
    g = kwargs.pop('g', scipy.constants.g)
    sea_water_density = kwargs.pop('sea_water_density', 1025)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    # initialize wave
    wave = FentonWave(wave_height=wave_height, wave_period=wave_period, depth=hs)

    # alpha parameters
    a_1 = 0.6 + 0.5 * (4 * np.pi * hs / wave.wave_length / np.sinh(4 * np.pi * hs / wave.wave_length)) ** 2
    a_2 = min(
        (hb - depth) / (3 * hb) * (wave_height / depth) ** 2,
        2 * depth / wave_height
    )
    a_3 = 1 - (wall_height - freeboard) / hs * (1 - 1 / np.cosh(2 * np.pi * hs / wave.wave_length))
    a_star = a_2

    # pressure components
    eta_star = 0.75 * (1 + np.cos(np.deg2rad(angle))) * l_1 * wave_height
    p1 = 0.5 * (1 + np.cos(np.deg2rad(angle))) * \
        (l_1 * a_1 + l_2 * a_star * np.cos(np.deg2rad(angle)) ** 2) * sea_water_density * g * wave_height
    if eta_star > freeboard:
        p2 = (1 - freeboard / eta_star) * p1
    else:
        p2 = 0
    p3 = a_3 * p1
    pu = 0.5 * (1 + np.cos(np.deg2rad(angle))) * l_3 * a_1 * a_3 * sea_water_density * g * wave_height

    # load above water
    if eta_star > freeboard:
        load_aw = freeboard * p2 + freeboard * (p1 - p2) * 0.5
    else:
        load_aw = p1 * freeboard * 0.5

    # load under water
    load_uw = (wall_height - freeboard) * p3 + (wall_height - freeboard) * (p1 - p3) * 0.5

    # total wave load
    load = load_aw + load_uw

    return pd.DataFrame(
        data=[load, p1, p2, p3, pu],
        index=['Total wave load (N/m)', 'p1 (Pa)', 'p2 (Pa)', 'p3 (Pa)', 'pu (Pa)'],
        columns=['Value']
    )


# TODO - below are not implemented


def goda_2000(wave_height, wave_period, depth, freeboard, **kwargs):
    """
    Calculates wave load on vertical wall according to Goda (2000) formula
    (Random seas and design of maritime structures, p.134 - p.139)

    Mandatory inputs
    ================
    wave_height : float
        Significant wave height (m)
    wave_period : float
        Wave period (s)
    depth : float
        Water depth at the wall (m)
    freeboard : float
        Freeboard (m)

    Optional inputs
    ===============
    depth_toe : float
        Water depth at structure toe (m)
    wall_height : float
        Vertical wall submerged height (m)
    angle : float
        Angle of wave attack (degrees, 0 - normal to structure)
    hb : float
        Water depth at distance 5H13 seaward from the structure

    Returns
    =======
    A pandas dataframe with pressures, total load, load centroid (above wall footing, i.e. depth d)
    """

    angle = kwargs.pop('angle', 0)
    depth_toe = kwargs.pop('depth_toe', depth)
    wall_height = kwargs.pop('wall_height', depth)
    hb = kwargs.pop('hb', depth_toe)
    g = kwargs.pop('g', scipy.constants.g)
    sea_water_density = kwargs.pop('sea_water_density', 1025)
    assert len(kwargs) == 0, 'unrecognized arguments passed in: {}'.format(', '.join(kwargs.keys()))

    # initialize wave
    wave = FentonWave(wave_height=wave_height, wave_period=wave_period, depth=depth_toe)

    # alpha parameters
    eta_star = 0.75 * (1 + np.cos(np.deg2rad(angle))) * wave_height
    a_1 = 0.6 + 0.5 * ((4 * np.pi * depth_toe / wave.wave_length) /
                       np.sinh(4 * np.pi * depth_toe / wave.wave_length)) ** 2
    a_2 = min(
        (hb - depth) / (3 * hb) * (wave_height / depth) ** 2,
        2 * depth / wave_height
    )
    a_3 = 1 - (wall_height / depth_toe) * (1 - (1 / np.cosh(2 * np.pi * depth_toe / wave.wave_length)))

    # pressure components
    p1 = 0.5 * (1 + np.cos(np.deg2rad(angle))) * \
        (a_1 + a_2 * (np.cos(np.deg2rad(angle)) ** 2)) * sea_water_density * g * wave_height
    p2 = p1 / np.cosh(2 * np.pi * depth_toe / wave.wave_length)
    p3 = a_3 * p1
    if eta_star > freeboard:
        p4 = p1 * (1 - freeboard / eta_star)
    else:
        p4 = 0
    hc_star = min(eta_star, freeboard)
    pu = 0.5 * (1 + np.cos(np.deg2rad(angle))) * a_1 * a_3 * sea_water_density * g * wave_height

    # total load. moment, and centroid
    p = 0.5 * (p1 + p3) * wall_height + 0.5 * (p1 + p4) * hc_star
    mp = (1 / 6) * (2 * p1 + p3) * (wall_height ** 2) + 0.5 * (p1 + p4) * wall_height * hc_star + \
        (1 / 6) * (p1 + 2 * p4) * (hc_star ** 2)
    p_centroid = mp / p
    return pd.DataFrame(
        data=[
            round(p, 3),
            round(mp, 3),
            round(p_centroid, 3),
            round(hc_star, 3),
            round(p1, 3),
            round(p2, 3),
            round(p3, 3),
            round(p4, 3),
            round(pu, 3),
            round(a_1, 3),
            round(a_2, 3),
            round(a_3, 3),
            round(eta_star, 3),
        ],
        index=[
            'Total wave load [N/m]',
            'Total wave force moment at seabed [N-m]'
            'Load centroid [m]',
            'hc_star [m]',
            'p1 [Pa]',
            'p2 [Pa]',
            'p3 [Pa]',
            'p4 [Pa]',
            'pu [Pa]',
            'a_1',
            'a_2',
            'a_3',
            'Wave reach [m]',
        ],
        columns=[
            'Value'
        ]
    )


if __name__ == '__main__':
    mload = Morrison(wave_height=2, wave_period=8, depth=10, current_velocity=0, rho=1028,
                     element_type='vertical cylinder', cylinder_diameter=0.4, cylinder_top=20,
                     cylinder_bottom=0, drag_coefficient=1.2, inertia_coefficient=2)
