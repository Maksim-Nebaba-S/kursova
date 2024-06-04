import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from Model import ReluActiovationModel
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import os
from tensorflow import keras
from customCallback import CustomModelCheckpoint
from folder_manager import fresult

import json

directory = f"weights/{fresult}"

epochs = 150
period = 10
batch_size = 32 
target_size = (224, 224)
patience = 7 

train_data_dir = "processed/train"
test_data_dir = "processed/test"
val_data_dir = "processed/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

def count_images_per_class(generator):
   
    labels = generator.classes
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    output = "; ".join(f"{class_name}, {count}" for class_name, count in class_counts.items())
    return class_counts

print("Train images per class:", count_images_per_class(train_generator))
print("Validation images per class:", count_images_per_class(val_generator))
print("Test images per class:", count_images_per_class(test_generator))



class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
num_classes = len(class_names)

#callbacks
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

checkpoint_callback = ModelCheckpoint(f'{directory}/best_weights.h5', monitor='val_loss', save_best_only=True, verbose=1)
 
n_epochs = CustomModelCheckpoint(period = period)
 
reduce_Plate_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)


input_shape = (224, 224, 3)

tf.device("GPU")

model_builder = ReluActiovationModel(input_shape=input_shape, num_classes=num_classes)

model = model_builder.build_model()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.n // batch_size + 1,
    validation_steps=val_generator.n  // batch_size + 1,
    callbacks=[early_stopping_callback, checkpoint_callback, reduce_Plate_lr, n_epochs]
)

model.save(os.path.join(directory,'last_weights.h5'))

epochs_range = range(1, (len(history.history['loss']))+1)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure()
plt.plot(epochs_range, train_loss, 'b', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs_range, train_acc, 'b', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
 
with open(f'{directory}/training_history.json', 'w')  as f:
    json.dump(str(history.history), f)