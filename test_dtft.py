# dtftテスト

import numpy as np
import cmath
import matplotlib.pyplot as plt # 結果表示用
import math
from matplotlib.ticker import (MultipleLocator, AutoLocator)
from scipy import signal

# 離散時間フーリエ変換関数
def DTFT(omega, xn):
    _Xw = np.zeros(len(omega), dtype = np.complex)
    
    for i, w in enumerate(omega):
        for n, xn_i in enumerate(xn):
            _Xw[i] += xn_i * cmath.exp(-1j * w * n)
    
    return _Xw

# nを指定して、シンプソンの公式で逆離散時間フーリエ変換を計算
def InvertWithSimpsonRule(Xomega,omega_array,n):
    lim_n = int(len(Xomega)/2)
    h = abs(omega_array[1] - omega_array[0])
    
    sum = np.zeros(1, dtype=complex)
    for i in range(1, lim_n):
        f1 = (Xomega[2*i - 2] * (cmath.exp((1j) * omega_array[2*i - 2] * n )) )
        f2 = (Xomega[2*i - 1] * (cmath.exp((1j) * omega_array[2*i - 1] * n )) )
        f3 = (Xomega[2*i - 0] * (cmath.exp((1j) * omega_array[2*i - 0] * n )) )
        sum[0] += h*(f1 + 4*f2 + f3)/3

    
    return sum[0] / (2 * cmath.pi)

# nを変化させながら離散時間フーリエ逆変換計算
def InverseDTFT(omega_array, Xw_array, n_array):
    xn = np.zeros(len(n_array), dtype=np.complex)
    for n, n_atom in enumerate(n_array):
        xn[n] = InvertWithSimpsonRule(Xw_array, omega_array, n)
        
    return xn


# 離散時間フーリエ変換データ表示関数
def showDTFTData(omega, Xw, normalize=True, SamplingTime = 1):
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    if normalize:
        plt.plot(omega,Xw)
        plt.xlabel('Normalized angular frequency[rad]')
        plt.ylabel('X($\omega$)')
        ax.xaxis.set_major_locator(MultipleLocator(np.pi))
    else:
        x_data_hz = omega/SamplingTime/(2*np.pi) 
        plt.plot(x_data_hz ,Xw)
        plt.xlabel('Frequency[Hz]')
        plt.ylabel('abs(X($\omega$))')
        
        xtick = x_data_hz[-1]/10
        
        if xtick >= 1:
            ax.xaxis.set_major_locator(MultipleLocator(xtick))
        else:
            ax.xaxis.set_major_locator(AutoLocator())
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.show()

# 離散データ表示用関数
def showDiscreteData(x,y):
    markerline, stemline, baseline = plt.stem(x, y)
    plt.setp(markerline, marker='o', markeredgewidth=2, markersize=8, color='black', markerfacecolor='white', zorder=3)
    plt.setp(stemline, color='black', zorder=2, linewidth=1.0)
    plt.setp(baseline, color='black', zorder=1)
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel('y [n]')
    plt.show()



def butter_lowpass(lowcut, fs, order=4):
    '''バターワースローパスフィルタを設計する関数
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return b, a



# 離散時間データの定義
xd = np.arange(100)
# xd = np.array([0,0,1,1,1,0,0,0])
n = np.arange(len(xd))

# 離散時間データの表示
showDiscreteData(n, xd)

# DEFTの計算に用いる周波数 omega の範囲を指定
omega = np.linspace(0*3.14, 2*3.14, 2020)

# 離散時間フーリエ変換の計算
Xw = DTFT(omega, xd)

# フーリエ変換結果の表示
showDTFTData(omega, Xw.real)

# DEFTの計算に用いる周波数 omega の範囲を指定
omega = np.linspace(-1*3.14, 1*3.14, 2020)

# 離散時間フーリエ変換の計算
Xw = DTFT(omega, xd)

# 離散時間フーリエ逆変換の計算
i_xd = InverseDTFT(omega, Xw, n)

# 逆フーリエ変換結果の表示
showDiscreteData(n, i_xd.real)

