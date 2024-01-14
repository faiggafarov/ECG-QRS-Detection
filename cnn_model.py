#AI_Project_1_2006A904_Faig-Gafarov_1806A047_BuketNazlı-Yanık_18067039_Osman-ÖĞÜT
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch
import os

# Define dataset paths
base_path = 'trainingDataset/'
qrs_path = os.path.join(base_path, 'QRS')
not_qrs_path = os.path.join(base_path, 'notQRS')

# Define the input shape for the CNN
input_shape = (128, 128, 3)

# Function to preprocess and resize images
def preprocess_images(directory, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb')
    return generator

# CNN model building function
def buildCnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Hyperparameter optimization model building function
def buildHypermodel(hp):
    model = Sequential()
    model.add(Conv2D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('kernel_size', values=[3, 5]),
                     activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('units', min_value=64, max_value=256, step=64),
                    activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Hyperparameter tuning setup
tuner = RandomSearch(
    buildHypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='model_tuning',
    project_name='QRS_detection'
)

# Start the hyperparameter tuning process
tuner.search_space_summary()
# Number of classes (QRS, notQRS)
num_classes = 2

# Build the CNN model 
model = buildCnn(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 32  
train_generator = preprocess_images(base_path, batch_size)
test_generator = preprocess_images(base_path, batch_size)

# Evaluate the best model
test_loss, test_accuracy = model.evaluate(test_generator)
model.fit(train_generator, epochs=10, steps_per_epoch=train_generator.samples // batch_size)

# Save the model
model_save_path = 'best_model/best_cnn_model.h5'
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Print the evaluation results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
