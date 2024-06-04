import tensorflow as tf
from tensorflow import keras
import keras.utils as image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import matplotlib as plt
import numpy as np


test_data_dir = "processed/test"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model = keras.models.load_model('weights/2024-06-03_02-32-00/best_weights.h5')
model.summary()

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
