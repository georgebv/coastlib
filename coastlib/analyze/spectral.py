from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np


class Spectrum:

    def __init__(self, series, frequency):
        self.series = series
        self.frequency = frequency
        self.f, self.t, _Sxx = spectrogram(x=self.series, fs=self.frequency)
        self.Sxx = _Sxx.T[0] / 2
        self.fp = self.f[np.argmax(self.Sxx)]
        self.Tp = 1 / self.fp
        self.Hm0 = 4 * np.sqrt(self.moment(0))
        self.Tm = np.sqrt(self.moment(0) / self.moment(2))

    def __repr__(self):
        s = f'     Signal length -> {len(self.series):d}\n' \
            f'Sampling frequency -> {self.frequency:.2f} Hz\n' \
            f'               Hm0 -> {self.Hm0:.2f}\n' \
            f'                Tp -> {self.Tp:.2f} s\n' \
            f'                Tm -> {self.Tm:.2f} s'
        return s

    def moment(self, order):
        _f = np.array([self.f[i+1] - self.f[i] for i in range(len(self.f) - 1)])
        _Sxx = np.array([(self.Sxx[i+1] + self.Sxx[i]) / 2 for i in range(len(self.Sxx) - 1)])
        _ff = np.array([(self.f[i+1] + self.f[i]) / 2 for i in range(len(self.f) - 1)])
        _e = _f * _Sxx * (_ff ** order)
        return np.sum(_e)

    def plot(self):
        plt.pcolormesh(self.f, [0, 1], [self.Sxx, self.Sxx])
        plt.colorbar()

if __name__ == '__main__':
    from coastlib import AiryWave
    frequency = 1  # Hz
    t = np.arange(0, 3600, 1 / frequency)
    etas = []
    for hs, tp in zip(np.random.rayleigh(1, 10), np.random.rayleigh(4, 10)):
        wave = AiryWave(wave_height=hs, wave_period=tp, depth='deep')
        eta = []
        for _t in t:
            wave.dynprop(t=_t, z=0, x=0)
            eta.append(wave.S)
        eta = np.array(eta)
        etas.append(eta)

    seta = []
    for i in range(len(t)):
        _seta = np.array([a[i] for a in etas])
        seta.append(_seta.sum())
    seta = np.array(seta)
    # plt.plot(t, seta)

    sp = Spectrum(series=seta, frequency=frequency)
    print(sp)
    plt.plot(sp.f, sp.Sxx)
    plt.show()
