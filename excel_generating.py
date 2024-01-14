import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch
from keras.callbacks import CSVLogger
import pandas as pd
import os

base_path = 'trainingDataset/'
qrs_path = os.path.join(base_path, 'QRS')
not_qrs_path = os.path.join(base_path, 'notQRS')

input_shape = (128, 128, 3)

def preprocess_images(directory, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb')
    return generator

def build_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_hypermodel(hp):
    model = Sequential()
    model.add(Conv2D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('kernel_size', values=[3, 5]),
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('units', min_value=64, max_value=256, step=64),
                    activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='model_tuning',
    project_name='QRS_detection'
)

tuner.search_space_summary()

num_classes = 2

model = build_cnn(input_shape, num_classes)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

batch_size = 32 
train_generator = preprocess_images(base_path, batch_size)

model.fit(train_generator, epochs=10, steps_per_epoch=train_generator.samples // batch_size)

model_save_path = 'models/best_model/best_cnn_model.h5'
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
tuner.search_space_summary()
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
tuned_hyperparameters = best_trial.hyperparameters.values

best_trial_metric_names = best_trial.metrics.metrics.keys()
last_metrics = {name: best_trial.metrics.get_last_value(name) for name in best_trial_metric_names}

df = pd.DataFrame({**tuned_hyperparameters, **last_metrics}, index=[0])

excel_file_path = 'cnn_performance_metrics.xlsx'
df.to_excel(excel_file_path, index=False)
print(f"Performance metrics saved to {excel_file_path}")