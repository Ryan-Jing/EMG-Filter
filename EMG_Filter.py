import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fft import fft, fftfreq

x = np.linspace(0,10,10000) # fs is 10000/10 = 1000
y = np.sin(2*np.pi*100*x) # frequency is 100 Hz

order=8
f1 = 150
f2 = 250 
sos=signal.butter(order,[f1,f2], btype='bandpass', fs=1000, output='sos') # Creates the filter with type 'band'
databpf = signal.sosfilt(sos,y) # Applies the filter

# Get the fft fo both unfiltered and filtered
N=len(y)
yf=(2/N) * np.abs(fft(y)) 
databpf_f=(2/N) * np.abs(fft(databpf))

# Truncate fft to RHS only
xf=fftfreq(N,1/1000) # ts=1/fs is the 0.1
yf=yf[0:N//2]
databpf_f=databpf_f[0:N//2]
xf=xf[0:N//2]

# Plot them
plt.plot(xf,yf,'--',label='signal')
plt.plot(xf,databpf_f,'k',label='filtered signal')
plt.xlim([98,102])
plt.legend()
plt.show()