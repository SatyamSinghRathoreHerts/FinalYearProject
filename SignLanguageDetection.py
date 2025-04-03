import cv2
import numpy as np
import os
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def dataSetMaker():
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


def modelMaker():

    # Dataset path
    dataset_path = "dataset/"

    # Image preprocessing
    image_size = (64, 64)
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Define CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')  # 26 output classes (A-Z)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_generator, validation_data=val_generator, epochs=10)

    # Save model
    model.save("sign_language_model1.h5")


# In[4]:

def imgPrediction():
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


    # In[5]:



def videoPrediction():
    # Load video
    video_path = "B.mp4"
    cap = cv2.VideoCapture(video_path)
    
    model = load_model("sign_language_model1.h5")
    labels = {i: chr(65 + i) for i in range(26)}
    
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

def main():
    imgPrediction()

main()

