import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import json

# Path to your ASL dataset directory
dataset_path = '/Users/user/Desktop/asl_dataset'  # Modify this path as needed

# Generate the label mapping
folders = sorted(os.listdir(dataset_path), key=lambda x: (len(x), x))
label_mapping = {idx: folder for idx, folder in enumerate(folders)}

# Save the mapping as JSON
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)

print("label_mapping.json has been created.")
model = load_model('asl_model.h5')

# Load the label mapping
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

# Flip the dictionary for easier decoding
idx_to_label = {int(key): value for key, value in label_mapping.items()}

# Define image size
IMG_SIZE = (64, 64)

def preprocess_frame(frame):
    """
    Preprocess the frame for prediction: convert to grayscale, resize, normalize.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

# Start webcam capture
cap = cv2.VideoCapture(0)

print("Starting webcam. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Define a region of interest (ROI) for hand gestures
    roi = frame[100:400, 100:400]  # Adjust as needed
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Preprocess the ROI for prediction
    processed_frame = preprocess_frame(roi)

    # Predict the gesture
    prediction = model.predict(processed_frame)
    predicted_label_idx = np.argmax(prediction)
    predicted_label = idx_to_label[predicted_label_idx]
    confidence = prediction[0][predicted_label_idx]

    # Display the prediction and confidence on the frame
    text = f"{predicted_label} ({confidence:.2f})"
    cv2.putText(frame, text, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("ASL Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
