import coastlib.coreutils.design_tools as dt
import sys


def atend():
    b = input('Try a new case? [y/n]\n')
    if b == 'n':
        sys.exit()
    elif b != 'y':
        print('Invalid input, please try again\n')
        atend()

while True:
    Hs = float(input('Input wave height [ft]: '))
    Tp = float(input('Input wave period [sec]: '))
    water_elevation = float(input('Input water elevation [ft NAVD88]: '))
    deck_elevation = float(input('Input sea bed elevation at the wall [ft NAVD88]: '))
    bed_elevation = float(input('Input offshore sea bed elevation [ft NAVD88]: '))
    shed_roof_elevation = float(input('Input wall top end elevation [ft NAVD88]: '))

    Hs *= 0.3048
    water_elevation *= 0.3048
    deck_elevation *= 0.3048
    bed_elevation *= 0.3048
    shed_roof_elevation *= 0.3048

    toe_depth = water_elevation - bed_elevation
    wall_depth = water_elevation - deck_elevation
    freeboard = shed_roof_elevation - water_elevation
    wall_height = shed_roof_elevation - deck_elevation

    load = dt.goda_2000(H13=Hs, T13=Tp, h=toe_depth, hc=freeboard, d=wall_depth)
    print()
    print('\nOutput:\n')
    print(load)
    print('\nLoad centroid is given relative to the wall footing\n')
    atend()
