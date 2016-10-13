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
        print('Don\'t even try this')


data = pd.read_csv('C:\\Users\GRBH.COWI.001\Desktop\GitHub repositories\coastlib\\test\Drawing1.csv')
a = data.values
inter = cnt.intersection(5, a)
for i in range(len(inter)):
    print('Intersection ' + str(i+1) + ': x = ' + str(round(inter[i], 2)))
