import sys
import time


def print_update(text):
    sys.stdout.write('\r{}'.format(text))
    sys.stdout.flush()

while True:
    try:
        for i in range(100):
            print_update(i)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('Don\'t even try this')
