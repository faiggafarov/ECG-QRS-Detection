import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pywt
import os
import logging
from sklearn.model_selection import train_test_split, KFold

output_base = 'trainingDataset/'
record_list = ['100', '101', '102']

image_log = os.path.join(output_base, 'pathsLabels.txt')
image_logger = logging.getLogger('Image Logger')
image_logger.setLevel(logging.INFO)
image_file_handler = logging.FileHandler(image_log, mode='w')
image_logger.addHandler(image_file_handler)

for record_name in record_list:
    record = wfdb.rdrecord(record_name, sampto=15000)
    annotation = wfdb.rdann(record_name, 'atr', sampto=15000)
    data = record.p_signal
    channel = data[:, 0]
    signal_length = len(channel)

    wavelets = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4',
                'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',
                'bior3.7']
    waveletsLevel = 5

    ecgDataset = np.zeros(((waveletsLevel + 1) * len(wavelets), signal_length))

    channel = channel - np.mean(channel)
    minValue = min(channel)
    maxValue = max(channel)
    channel = (channel - minValue) / (maxValue - minValue)
    ecgDataset[0] = channel
    counterWaveletAssignment = 0

    for i, wavelet in enumerate(wavelets):
        coeffs = pywt.wavedec(channel, wavelet, level=waveletsLevel)

        time_values = [np.linspace(0, signal_length, len(coef),
                                    endpoint=False, dtype=int) for coef in coeffs]

        print(len(time_values[:][0]), len(time_values[0][:]),
              max(time_values[:][0]), min(time_values[0][:]))

        for j, coef in enumerate(coeffs):
            coef = coef - np.mean(coef)
            minValue = min(coef)
            maxValue = max(coef)
            coef = (coef - minValue) / (maxValue - minValue)

            ecgDataset[counterWaveletAssignment][time_values[j]] = coef
            counterWaveletAssignment = counterWaveletAssignment + 1

    stepSize = 500
    iterationNum = 20
    imageNum = int(signal_length / stepSize)
    cmap = "viridis"
    counter = 0

    for i in range(0, signal_length - stepSize, iterationNum):
        newMatrix = 255 * ecgDataset[:, i:i + iterationNum] 
        segment_start = i
        segment_end = i + iterationNum
        is_qrs = any(segment_start <= ann_sample < segment_end for ann_sample in annotation.sample)
        if is_qrs:
            label=1
        else:
            label=0
        if is_qrs:
            folder ='QRS'
        else:
            folder ='notQRS'
        output_path = os.path.join(output_base, folder, f"{record_name}_{counter}.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(newMatrix, cmap=cmap)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        image_logger.info(f"{output_path} - Label: {label}")
        counter += 1
        if counter >= imageNum:
            break

image_paths = []
labels = []

with open(image_log, 'r') as file:
    for line in file:
        path, label_info = line.strip().split(' - Label: ')
        image_paths.append(path)
        labels.append(int(label_info))

split_log = os.path.join(output_base, 'dataset_splitting.txt')
split_logger = logging.getLogger('Split Logger')
split_logger.setLevel(logging.INFO)
split_file_handler = logging.FileHandler(split_log, mode='w')
split_logger.addHandler(split_file_handler)

X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

split_logger.info("Random Splitting:")
split_logger.info("\nTraining Data :")
for i in range(5):
    split_logger.info(f"{X_train[i]} - Label: {y_train[i]}")

split_logger.info("\nTest Data :")
for i in range(5):
    split_logger.info(f"{X_test[i]} - Label: {y_test[i]}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
for train_index, test_index in kf.split(image_paths):
    X_train_fold, X_test_fold = [image_paths[i] for i in train_index], [image_paths[i] for i in test_index]
    y_train_fold, y_test_fold = [labels[i] for i in train_index], [labels[i] for i in test_index]
    split_logger.info(f"\nCross-Validation Fold {fold + 1}:")
    split_logger.info("Training Data :")
    for i in range(5):
        split_logger.info(f"{X_train_fold[i]} - Label: {y_train_fold[i]}")

    split_logger.info("Test Data :")
    for i in range(5):
        split_logger.info(f"{X_test_fold[i]} - Label: {y_test_fold[i]}")
    fold += 1
