#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[10]:


# Load the trained model
model = load_model("sign_language_model1.h5")

# Define label mappings (A-Z)
labels = {i: chr(65 + i) for i in range(26)}

# Load an image for prediction
image_path = "TEST.jpg"
image = Image.open(image_path)
image = image.resize((64, 64))  # Resize to model input size
image = np.array(image) / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(image)
predicted_label = labels[np.argmax(prediction)]

print(f"Predicted Sign: {predicted_label}")


# In[11]:


# Load video
video_path = "B.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 10 == 0:  # Process every 10th frame
        img = cv2.resize(frame, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict sign
        prediction = model.predict(img)
        sign = labels[np.argmax(prediction)]

        # Display prediction
        cv2.putText(frame, f"Predicted: {sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Predicted Sign: {sign}")


# In[ ]:




