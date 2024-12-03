import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import sys

# Set proper encoding for printing in different environments
sys.stdout.reconfigure(encoding='utf-8')

# Suppress TensorFlow warnings that aren't critical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
try:
    model = load_model('trained_CNN.keras')
except OSError:
    print("Error: Model file 'trained_CNN.keras' not found.")
    sys.exit(1)

# Load class labels from CSV file
try:
    labels_df = pd.read_csv('GTSRB_dataset/labels.csv', encoding='utf-8')
except FileNotFoundError:
    print("Error: Labels CSV file not found.")
    sys.exit(1)

# Clean up labels to make them ASCII-friendly for display purposes
labels_df['Name'] = labels_df['Name'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
class_labels = labels_df['Name'].tolist()

# Display the loaded labels
print("Loaded class labels:")
print(labels_df['Name'])

# Function to preprocess the frame for prediction
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (32, 32))  # Resize to 32x32
    frame_normalized = frame_resized / 255.0     # Normalize pixel values to range [0, 1]
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension for prediction

# Open the webcam (device ID 0 by default)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam")
    sys.exit(1)

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Preprocess the frame for prediction
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions using the model
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Fetch label
    if 0 <= predicted_class < len(class_labels):
        label = class_labels[predicted_class]
    else:
        label = "Unknown"

    # Display the label on the frame
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame with annotations
    cv2.imshow('Traffic Sign Recognition', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
