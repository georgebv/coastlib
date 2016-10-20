import time
import threading
import math
from coastlib.miscutils.convenience_tools import print_update


class ProgressBar(threading.Thread):
    def __init__(self, mutex, text):
        self.text = text
        self.mutex = mutex
        threading.Thread.__init__(self)

    def run(self):
        global progress1
        while progress1 < 100:
            bar_f = math.floor((progress1 / 100) * 20)
            bar_e = 20 - bar_f
            bar = '|' + bar_f * '#' + bar_e * '-' + '|'
            with self.mutex:
                for sym in ['-', '\\', '|', '/']:
                    print_update('{2} {0} {3} [{1}%]'.format(self.text, round(progress1, 0), sym, bar))
                    time.sleep(0.1)
        with self.mutex:
            bar_f = 20
            bar_e = 0
            bar = '|' + bar_f * '#' + bar_e * '-' + '|'
            print_update('! {0} {1} [100%]'.format(self.text, bar))



stdoutmutex = threading.Lock()
progress1 = 0
threads = []
thread = ProgressBar(mutex=stdoutmutex, text='Printing i\'s')
thread.start()
threads.append(thread)

for i in range(0, 100):
    progress1 += 10
    time.sleep(0.1)

for thread in threads:
    thread.join()
print('Over')
