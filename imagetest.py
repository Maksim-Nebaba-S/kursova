import tensorflow as tf
from keras.models import load_model
import keras.utils as image
import numpy as np
import matplotlib.pyplot as plt


model = load_model('weights/2024-06-03_02-32-00/best_weights.h5')

class_names = ["dog", "horse", "elephant", "butterfly", "hen", "cat", "cow", "sheep", "spider", "squirrel"]

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    return img_array

img_path = 'vivaldi_qfioeCMV5j.png' 
img_array = preprocess_image(img_path)

predictions = model.predict(img_array)
percentage_probabilities = predictions[0] * 100

for class_name, probability in zip(class_names, percentage_probabilities):
    print(f"{class_name}: {probability:.2f}%")

plt.imshow(image.load_img(img_path))
plt.axis('off')  

y_pos = np.arange(len(class_names))
plt.figure(figsize=(10, 8))
plt.barh(y_pos, percentage_probabilities, align='center')
plt.yticks(y_pos, class_names)
plt.xlabel('Probability (%)')
plt.title('Class Probabilities')
plt.gca().invert_yaxis()  
plt.show()