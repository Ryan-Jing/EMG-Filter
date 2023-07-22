import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# print(os.listdir())
path = 'EMG-Filter/EMG_Datasets/EMG_Datasets.csv'

dataset = pd.read_csv(path)
relaxedFilePath = 'EMG-Filter/Output_Files/relaxedRMS.csv'
contractFilePath = 'EMG-Filter/Output_Files/contractRMS.csv'

x = []
yRelaxed = []
yContract = []
length = len(dataset['Time (s)'])

def RMS(emgData):
    return np.mean((np.sqrt((emgData ** 2))))

for _, row in dataset.iterrows():
    time = (row['Time (s)'])
    relaxedEMG = row['EMG_Relaxed (mV)']
    contractEMG = row['EMG_Contracted (mV)']
    relaxedRMS = RMS(relaxedEMG)
    contractedRMS = RMS(contractEMG)

    x.append(time)
    yContract.append(contractedRMS)
    yRelaxed.append(relaxedRMS)

with open(contractFilePath, "w") as file:
    file.write("Time (s),RMS EMG Contracted\n")
    for time, contractedRMS in zip(x, yContract):
        file.write(f"{time},{contractedRMS}\n")

with open(relaxedFilePath, "w") as file:
    file.write("Time (s),RMS EMG Relaxed\n")
    for time, relaxedRMS in zip(x, yRelaxed):
        file.write(f"{time},{relaxedRMS}\n")


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

