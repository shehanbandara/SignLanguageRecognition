import cv2
import numpy as np
import os
import tensorflow as tf

# Set up the webcam
cap = cv2.VideoCapture(0)

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
        
        # Create Region Of Interest (ROI)
        cv2.rectangle(frame, (50, 200), (350, 500), (0, 0, 255), 5)
        roi = frame[200:500, 50:350]

        # Prepare the ROI for the prediction
        roi = cv2.resize(roi, (50,50))
        roi = roi / 255

        # Get the prediction
        prediction = model.predict(roi.reshape(1, 50, 50, 3))

        # Get the probability of the prediction being correct
        probability = np.amax(prediction)

        # If the probability of the prediction being correct is greater than 0.9
        if probability > 0.9:
            
            # Get the index of the class with the maximum probability
            classIndex = np.argmax(prediction)

            # Get the letter with the maximum probability
            letter = dataFolders[classIndex]

            # Write the letter with the maximum probability to the frame
            cv2.putText(frame, letter, (200,180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

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