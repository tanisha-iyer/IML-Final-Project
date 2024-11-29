import tensorflow as tf
import keras
from keras import layers, models  
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

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
                image = image / 255.0  # Normalize all the images 
                
                images.append(image)
                labels.append(label_mapping[folder])
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"Total images loaded: {len(images)}")
    print(f"Number of classes: {len(label_mapping)}")
    
    return np.array(images), np.array(labels), label_mapping

# Reshape images to make them all the same dimension
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

def build_model(num_classes):
    inputs = keras.Input(shape=(64, 64, 1))
    
    # Data augmentation layers
    x = layers.RandomRotation(0.2)(inputs)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    
    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# Load and prepare data
dataset_path = '/Users/user/Desktop/asl_dataset'
images, labels, label_mapping = load_data(dataset_path)
images = images.reshape(-1, 64, 64, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Get number of classes
num_classes = len(np.unique(labels))
print(f"Number of classes for model: {num_classes}")

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Initialize and compile model
model = build_model(num_classes)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create data generator with augmentation
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda x: data_augmentation(x, training=True)
)

# Train the model with augmentation
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)
# Save the model
model.save('asl_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
