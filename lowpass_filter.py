import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def butter_lowpass(lowcut, fs, order=4):
    '''バターワースローパスフィルタを設計する関数
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a


# Plotting the frequency response.
fs = 2000 # sampling frequency
cutoff = 100 # cutoff frequency

plt.figure()
# 次数1~4まで繰り返し描画
for i in range(1, 5):
    b, a = butter_lowpass(cutoff, fs, order=i)
    w, h = signal.freqz(b, a)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), label='order='+str(i))
plt.axvline(cutoff, color='k', label='cutoff={}Hz'.format(cutoff))
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.legend(loc='best')
plt.grid()