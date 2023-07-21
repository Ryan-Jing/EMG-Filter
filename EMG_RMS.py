import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# print(os.listdir())
path = 'EMG-Filter/EMG_Datasets/EMG_Datasets.csv'

dataset = pd.read_csv(path)
relaxedFilePath = 'relaxedRMS.csv'
contractFilePath = 'contractRMS.csv'

x = []
yRelaxed = []
yContract = []
length = len(dataset['Time (s)'])

def RMS(emgData):
    return np.mean((np.sqrt((emgData ** 2))))

for _, row in dataset.iterrows():
    relaxedEMG = row['EMG_Relaxed (mV)']
    contractEMG = row['EMG_Contracted (mV)']
    x.append(row['Time (s)'])
    yContract.append(RMS(relaxedEMG))
    yRelaxed.append(RMS(contractEMG))

print(len(x))
print(len(yRelaxed))
print(len(yContract))

plt.figure(figsize=(12, 6))  # Adjust the figure size as per your requirement


# plotting the relaxed EMG
plt.subplot(1,2,1)
plt.plot(x, yRelaxed, label='Relaxed EMG RMS')
plt.xlabel('Time (s)')
plt.ylabel('Relaxed EMG')
plt.title('Relaxed EMG RMS Processed')

# plotting the contracted EMG
plt.subplot(1,2,2)
plt.plot(x, yContract, label = 'Contracted EMG RMS', color = 'orange')
plt.xlabel('Time (s)')
plt.ylabel('Contracted EMG')
plt.title('Contracted EMG RMS Processed')

plt.tight_layout()  # To prevent overlapping of subplots
plt.show()

