import sys
import time
import pandas as pd
import coastlib.miscutils.convenience_tools as cnt


def print_update(text):
    sys.stdout.write('\r{}'.format(text))
    sys.stdout.flush()

while True:
    break
    try:
        for i in range(100):
            print_update(i)
            time.sleep(0.5)
    except KeyboardInterrupt:
        a = input('r u sure? [y/n]')
        if a == 'y':
            break
        else:
            print('you asked for it')

while True:
    try:
        symbols = input('Give me 4 symbols: ')
        timestep = float(input('Give me a timestep: '))
        while True:
            for sm in symbols:
                print_update(sm)
                time.sleep(timestep)
    except KeyboardInterrupt:
        b = input('\nWant to quit? [y/n]')
        if b == 'y':
            break
        else:
            continue
sys.exit()


data = pd.read_csv('C:\\Users\GRBH.COWI.001\Desktop\GitHub repositories\coastlib\\test\Drawing1.csv')
a = data.values
inter = cnt.intersection(5, a)
for i in range(len(inter)):
    print('Intersection ' + str(i+1) + ': x = ' + str(round(inter[i], 2)))
