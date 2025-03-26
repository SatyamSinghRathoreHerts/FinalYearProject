#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import os


# In[13]:


video_path = "ABCs.mp4"
output_folder = "sign_frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(frame_rate) == 0:  # Extract 1 frame per second
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()


# In[ ]:




