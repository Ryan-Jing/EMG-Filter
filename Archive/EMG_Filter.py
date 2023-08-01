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
time = dataset['Time (s)'].to_numpy()
x = np.linspace(0,10,len(contractData)) # fs is 10000/10 = 1000

order = 8
fs = 1000
frequencies_bandpass = [0.1, 450]
frequencies_stop = [57, 63]

sos_bandpass = signal.butter(order, frequencies_bandpass, btype='pass', fs=fs, output='sos')
sos_stop = signal.butter(order, frequencies_stop, btype='stop', fs=fs, output='sos')

datasets = [contractData, relaxedData]
filtered_data = []

for data in datasets:
    databpf = signal.sosfilt(sos_bandpass, data)
    databpf_filtered = signal.sosfilt(sos_stop, databpf)
    filtered_data.append(databpf_filtered)

databpfOne, databpfThree = filtered_data

# Get the fft fo both unfiltered and filtered
N = len(contractData)
yf = (2/N) * np.abs(fft(contractData)) 
databpf_f = (2/N) * np.abs(fft(databpfOne))

# Truncate fft to RHS only
xf = fftfreq(N,1/1000) # ts=1/fs is the 0.1
yf = yf[0:N//2]
databpf_f = databpf_f[0:N//2]

yfr = (2/N) * np.abs(fft(relaxedData)) 
databpf_fOne = (2/N) * np.abs(fft(databpfThree))
yfr = yfr[0:N//2]
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

plt.figure(figsize=(13, 6))

plt.subplot(1,4,1)
plt.plot(time,contractData,label='contracted signal')
plt.legend()

plt.subplot(1,4,2)
plt.plot(time, relaxedData,label='relaxed signal')
plt.legend()

plt.subplot(1,4,3)
plt.plot(xf,databpf_f,label='filtered contracted signal', color = 'orange')
# plt.xlim([0,450])
plt.legend()

plt.subplot(1,4,4)
plt.plot(xf,databpf_fOne,label='filtered relaxed signal', color = 'green')
plt.legend()

plt.tight_layout()  # prevent overlapping of subplots


plt.pause(0.1) # so that the program doesnt hold/freeze with plt.show(), can continue check_termination

while True:
    if check_termination():
        break

plt.close()
