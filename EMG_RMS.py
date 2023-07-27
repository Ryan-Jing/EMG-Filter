import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# print(os.listdir())
path = 'EMG-Filter/EMG_Datasets/EMG_Datasets.csv'
relaxedFilterPath = 'EMG-Filter/Output_Files/relaxedFilterOut.csv'
contractedFilterPath = 'EMG-Filter/Output_Files/contractFilterOut.csv'

dataset = pd.read_csv(path)
filteredContractedDataSet = pd.read_csv(contractedFilterPath)

relaxedFilePath = 'EMG-Filter/Output_Files/relaxedRMS.csv'
contractFilePath = 'EMG-Filter/Output_Files/contractRMS.csv'


x = []
xFilter = []
yRelaxed = []
yContract = []
yFilterContract = []

length = len(dataset['Time (s)'])

def RMS(emgData):
    return np.sqrt((np.mean((emgData ** 2))))

for _, row in dataset.iterrows():
    time = (row['Time (s)'])
    relaxedEMG = row['EMG_Relaxed (mV)']
    contractEMG = row['EMG_Contracted (mV)']
    relaxedRMS = RMS(relaxedEMG)
    contractedRMS = RMS(contractEMG)

    x.append(time)
    yContract.append(contractedRMS)
    yRelaxed.append(relaxedRMS)

# CHECK AND UPDATE THISSSSSS 
for _, row in filteredContractedDataSet.iterrows():
    timeT = (row['Time (s)'])
    first_half_time = timeT[:len(timeT)//2]
    contractFilterEMG = row['Filtered EMG Contracted']
    contractFilterRMS = RMS(contractFilterEMG)

    xFilter.append(first_half_time)
    yFilterContract.append(contractFilterRMS)

with open(contractFilePath, "w") as file:
    file.write("Time (s),RMS EMG Contracted\n")
    for time, contractedRMS in zip(x, yContract):
        file.write(f"{time},{contractedRMS}\n")

with open(relaxedFilePath, "w") as file:
    file.write("Time (s),RMS EMG Relaxed\n")
    for time, relaxedRMS in zip(x, yRelaxed):
        file.write(f"{time},{relaxedRMS}\n")

def check_termination():
    termination_word = "close" 
    user_input = input("Type 'close' to terminate the program: ")
    return user_input.lower() == termination_word.lower()

print(len(x))
print(len(yRelaxed))
print(len(yContract))
print(len(yFilterContract))


plt.figure(figsize=(18, 6))  # Adjust the figure size as per your requirement


# plotting the relaxed EMG
plt.subplot(1,3,1)
plt.plot(x, yRelaxed, label='Relaxed EMG RMS')
plt.xlabel('Time (s)')
plt.ylabel('Relaxed EMG')
plt.title('Relaxed EMG RMS Processed')

# plotting the contracted EMG
plt.subplot(1,3,2)
plt.plot(x, yContract, label = 'Contracted EMG RMS', color = 'orange')
plt.xlabel('Time (s)')
plt.ylabel('Contracted EMG')
plt.title('Contracted EMG RMS Processed')

plt.subplot(1,3,3)
plt.plot(xFilter, yFilterContract, label = 'Contracted Filtered EMG RMS', color = 'orange')
plt.xlabel('Time (s)')
plt.ylabel('Contracted Filtered EMG')
plt.title('Contracted Filtered EMG RMS Processed')

plt.tight_layout()  # To prevent overlapping of subplots
plt.pause(0.1)

while True:
    if check_termination():
        break

plt.close()


