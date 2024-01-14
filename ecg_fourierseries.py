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

    def fourier_coefficients(t, signal, num_terms):
        coeffients = []
        for n in range(num_terms):
            an = 2*np.trapz(signal*np.cos(2*np.pi*n*t),t)
            bn = 2*np.trapz(signal*np.sin(2*np.pi*n*t),t)
            coeffients.append((an, bn))
        return coeffients
    
    def reconstruct_signal(t, coeffients):
        signal = np.zeros_like(t)
        individual_terms = []
        for n, (an, bn) in enumerate(coeffients):
            term = an * np.cos(2*np.pi*n*t) + bn*np.sin(2*np.pi*n*t)
            individual_terms.append(term)
            signal += term
        return signal, individual_terms
    
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

    num_terms = 100
    coefficients = fourier_coefficients(times, ECG1, num_terms)
    reconstructed_signal, individual_terms = reconstruct_signal(times,coefficients)
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(times, ECG1, label='ECG1')
    plt.title(f'Original Signal-Record Number-{record}')
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.legend()
    plt.subplot(3,1,2)
    for n,term in enumerate(individual_terms):
        plt.plot(times, term)

    plt.title(f'Individual Signal-Record Number-{record}')
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(times, reconstructed_signal, label='Our ECG')
    plt.title(f'Reconstructed Signal-Record Number-{record}')
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.legend()
    plt.tight_layout()
    plt.show()


