import wfdb
import numpy as np
import matplotlib.pyplot as plt

record_names = ['100', '101', '102']

for record in record_names:
    records = wfdb.rdrecord(record, sampto=360)
    annotation = wfdb.rdann(record,'atr',sampto=360)

    data = records.p_signal
    ECG1 = data[:,0]
    ECG2 = data[:,1]

    times = np.arange(len(ECG1),dtype=float)/records.fs

    def plot_time_series(signal, time_points, title):
        plt.plot(time_points,signal)
        plt.title(title)
        plt.xlabel('Time (sec)')
        plt.ylabel('Amp')
        plt.grid(True)


    def discrete_fourier_transform(signal, sampling_frequency):
        N = len(signal)
        k = np.arange(N)
        n = k.reshape(N,1)
        W = np.exp(-2j * np.pi * k * n / N)
        frequencies = k * sampling_frequency / N
        return frequencies, np.dot(W,signal)

    def plot_discrete_fourier_transform(signal, sampling_frequency, title):
        frequencies, fourier_transform = discrete_fourier_transform(signal, sampling_frequency)

        positive_freq_indices = (frequencies >= 0) & (frequencies <= 500)
        plt.plot(frequencies[positive_freq_indices], np.abs(fourier_transform[positive_freq_indices]))
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

    samplingFrequency = records.fs

    plt.figure(figsize=(12,8))

    plt.subplot(2,1,1)
    plot_time_series(ECG1, times, f'Times Series Data-Record Number-{record}')

    plt.subplot(2,1,2)
    plot_discrete_fourier_transform(ECG1, samplingFrequency,f'DFT Coefficients-Record Number-{record}')

    plt.tight_layout()
    plt.show()
