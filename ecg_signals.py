import wfdb
import numpy as np
import matplotlib.pyplot as plt

record_names = ['100', '101', '102']

for record in record_names:
    records = wfdb.rdrecord(record, sampto=15000)
    annotation = wfdb.rdann(record, 'atr', sampto=15000)
    
    data = records.p_signal
    channel1 = data[:, 0]
    channel2 = data[:, 1]
    times = np.arange(len(channel1), dtype=float) / records.fs

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(times, channel1, color='b')
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude (Voltage)')
    plt.title(f'ECG Channel 1 - Record {record}')

    plt.subplot(2, 1, 2)
    plt.plot(times, channel2, color='g')
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude (Voltage)')
    plt.title(f'ECG Channel 2 - Record {record}')

    plt.tight_layout()
    plt.show()


