import cv2
import numpy as np
import os
import tensorflow as tf

# Set up the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load the Sign Language Recognition Model
model  = tf.keras.models.load_model('model.h5')

# Sort the data folders in alphabetical order
dataFolders = sorted(os.listdir('SignLanguageRecognitionData/'))

# Delete .DS_Store if present
if dataFolders[0] == '.DS_Store':
    dataFolders.pop(0)
    
print(dataFolders)

while(True):

    # Read from the webcam
    ret, frame = cap.read()

    # If the frame is available
    if ret == True:
        
        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        # Press 'q' to stop the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the frame is unavailable stop the program
    else:
        break

# Release & destroy all resources
cap.release()
cv2.destroyAllWindows()