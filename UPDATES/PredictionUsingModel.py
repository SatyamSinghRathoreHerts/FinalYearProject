#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_language_model.h5")

# Define labels (A-Z)
labels = {i: chr(65 + i) for i in range(26)}

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




