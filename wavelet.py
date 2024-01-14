import wfdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
import pandas as pd

record_list = ['100', '101', '102']

for record_name in record_list:
    record = wfdb.rdrecord(record_name, sampto=15000)
    annotation = wfdb.rdann(record_name, 'atr', sampto=15000)
    # Extract ECG signal and annotations
    data = record.p_signal
    annotation_indices = annotation.sample
    channel1 = data[:, 0]
    times = np.arange(len(channel1), dtype=float) / record.fs
    signal_length = len(channel1)
    print(pywt.wavelist(kind='discrete'))
    # wavelets = pywt.wavelist(kind='discrete')
    wavelets = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
                'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',
                'bior3.7']
    waveletsLevel = 5
    ecgDataset = np.zeros(((waveletsLevel + 1) * len(wavelets), signal_length))
    channel1 = channel1 - np.mean(channel1)
    minValue = min(channel1)
    maxValue = max(channel1)
    channel1 = (channel1 - minValue) / (maxValue - minValue)
    ecgDataset[0] = channel1

    counterWaveletAssignment = 0
    for i, wavelet in enumerate(wavelets):
        coeffs1 = pywt.wavedec(channel1, wavelet, level=waveletsLevel)
        time_values = [np.linspace(0, signal_length, len(coef),
                                    endpoint=False, dtype=int) for coef in coeffs1]
        print(len(time_values[:][0]), len(time_values[0][:]),
              max(time_values[:][0]), min(time_values[0][:]))
        for j, coef in enumerate(coeffs1):
            coef = coef - np.mean(coef)
            minValue = min(coef)
            maxValue = max(coef)
            coef = (coef - minValue) / (maxValue - minValue)
            # Ensure the dimensions match
            ecgDataset[counterWaveletAssignment][time_values[j]] = coef
            counterWaveletAssignment = counterWaveletAssignment + 1

    stepSize = 500
    iterationNumber = 20
    imageNumber = int(signal_length / stepSize)
    print(len(ecgDataset[0, :]), len(ecgDataset[:, 0]))

    max_plots = 1
    cmap = "viridis"
    counter = 0
    for i in range(0, signal_length - stepSize, iterationNumber):
        newMatrix = np.zeros(((waveletsLevel + 1) * len(wavelets), iterationNumber), dtype=float)
        newMatrix[:, :] = 255 * ecgDataset[:, i:i + iterationNumber]
        counter = counter + 1
        plt.figure(figsize=(8, 8))
        plt.imshow(newMatrix, cmap=cmap)
        plt.tight_layout()
        plt.show()
        if counter >= max_plots:
            break
