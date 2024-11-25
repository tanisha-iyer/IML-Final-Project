import cv2
import time
import mediapipe as mp
import tensorflow as tf
import keras
from keras import layers, models  
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

Holistic(
  static_image_mode=False, #specifies if input should be treated as static or video 
  model_complexity=1, #used to specify the complexity of the pose landmark model 
  smooth_landmarks=True, 
  min_detection_confidence=0.5, 
  min_tracking_confidence=0.5
)
def load_data(base_dir, img_size=(64, 64)):
    images = []
    labels = []
    folders = sorted(os.listdir(base_dir), 
                    key=lambda x: (len(x), x))  
    
    # label mapping dictionary
    label_mapping = {folder: idx for idx, folder in enumerate(folders)}
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        print(f"Loading images from class {folder}")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Read and preprocess image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load {img_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, img_size)
                image = image / 255.0  # Normalize all the images so they're all the same makes the dataset easy to work with
                
                images.append(image)
                labels.append(label_mapping[folder])
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"Total images loaded: {len(images)}")
    print(f"Number of classes: {len(label_mapping)}")
    return np.array(images), np.array(labels), label_mapping

dataset_path = '/Users/user/Desktop/asl_dataset'
images, labels, label_mapping = load_data(dataset_path)
# Reshape images to make them all the same dimension
images = images.reshape(-1, 64, 64, 1)
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])
