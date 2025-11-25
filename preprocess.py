import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

def preprocess_image(img_path):
   
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)

    return img_array


def preprocess_dataset(folder_path):
    
    processed_images = []
    image_names = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)

            img_array = preprocess_image(img_path)
            processed_images.append(img_array)
            image_names.append(filename)

    return np.array(processed_images), image_names






