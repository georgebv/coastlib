import coastlib.coreutils.design_tools as dt


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

    load_A_Hs = dt.goda_2000(H13=Hs, T13=Tp, h=toe_depth, hc=freeboard, d=wall_depth)
    print()
    print('\nOutput:\n')
    for key in sorted(load_A_Hs.items(), key=lambda x:x[0]):
        print(str(key[0])+' = '+str(key[1]))
    b = input('\nTry a new case? [y/n]')
    if b != 'y':
        break
