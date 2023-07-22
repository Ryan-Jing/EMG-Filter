import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fft import fft, fftfreq

path = 'EMG-Filter/EMG_Datasets/EMG_Datasets.csv'
dataset = pd.read_csv(path)
contractedFilterOutput = 'EMG-Filter/Output_Files/contractFilterOut.csv'
relaxedFilterOutput = 'EMG-Filter/Output_Files/relaxedFilterOut.csv'

# y = np.sin(2*np.pi*100*x) # frequency is 100 Hz
relaxedData= dataset['EMG_Relaxed (mV)'].to_numpy()
contractData = dataset['EMG_Contracted (mV)'].to_numpy()
x = np.linspace(0,10,len(contractData)) # fs is 10000/10 = 1000

order = 8
f1 = 0.1
f2 = 160
f3 = 57
f4 = 63 

sos = signal.butter(order,[f1,f2], btype='bandpass', fs=1000, output='sos') # Creates the filter with type 'band'
databpf = signal.sosfilt(sos,contractData) # Applies the filter

sos = signal.butter(order,[f3,f4], btype='stop', fs=1000, output='sos') # Creates the filter with type 'band'
databpfOne = signal.sosfilt(sos,databpf) # Applies the filter

sos = signal.butter(order,[f1,f2], btype='bandpass', fs=1000, output='sos') # Creates the filter with type 'band'
databpfTwo = signal.sosfilt(sos, relaxedData) # Applies the filter

sos = signal.butter(order,[f3,f4], btype='stop', fs=1000, output='sos') # Creates the filter with type 'band'
databpfThree = signal.sosfilt(sos,databpfTwo) # Applies the filter

# Get the fft fo both unfiltered and filtered
N=len(contractData)
yf=(2/N) * np.abs(fft(contractData)) 
databpf_f = (2/N) * np.abs(fft(databpfOne))

# Truncate fft to RHS only
xf=fftfreq(N,1/1000) # ts=1/fs is the 0.1
yf=yf[0:N//2]
databpf_f = databpf_f[0:N//2]

yfr=(2/N) * np.abs(fft(relaxedData)) 
databpf_fOne = (2/N) * np.abs(fft(databpfThree))
yfr=yfr[0:N//2]
databpf_fOne = databpf_fOne[0:N//2]

xf = xf[0:N//2]

data_contracted = list(zip(xf, databpf_f))
data_relaxed = list(zip(xf, databpf_fOne))

with open(contractedFilterOutput, "w") as file:
    file.write("Time (s),Filtered EMG Contracted\n")
    for row in data_contracted:
        file.write(f"{row[0]},{row[1]}\n")

with open(relaxedFilterOutput, "w") as file:
    file.write("Time (s),Filtered EMG Relaxed\n")
    for row in data_relaxed:
        file.write(f"{row[0]},{row[1]}\n")

def check_termination():
    termination_word = "exit" 
    user_input = input("Type 'exit' to terminate the program: ")
    return user_input.lower() == termination_word.lower()

print(len(xf))
print(len(yf))
print(len(databpf_f))
print(len(databpf_fOne))

plt.figure(figsize=(14, 6))

plt.subplot(1,3,1)
plt.plot(xf,yf,'--',label='signal')
plt.xlim([98,102])
plt.legend()

plt.subplot(1,3,2)
plt.plot(xf,databpf_f,'k',label='filtered contracted signal')
plt.xlim([98,102])
plt.legend()

plt.subplot(1,3,3)
plt.plot(xf,databpf_fOne,'k',label='filtered relaxed signal')
plt.xlim([98,102])
plt.legend()

plt.tight_layout()  # prevent overlapping of subplots


plt.pause(0.1) # so that the program doesnt hold/freeze with plt.show(), can continue check_termination

while True:
    if check_termination():
        break

plt.close()
