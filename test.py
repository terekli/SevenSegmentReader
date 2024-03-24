import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from video2res import frame2digits

"""
Test the accuracy of the trained model using the test images in "test" folder.

All images are labelled.
"""

folder_path = '/Users/terekli/Desktop/video2num/test/'

roi = np.array([[132.4, 50], [129.2, 215.1], [507.2, 66.1], [496.3, 224.8]]) # 14g, 20psi

def build_testing(folder_path, roi):    

    digits_binary = [] # Store all the digits in every image

    file_list = os.listdir(folder_path)
    
    # Filter out unwanted entries
    filtered_file_list = []
    for f in file_list:
        # Check if the file ends with '.jpg'
        if f.endswith('.jpg'):
            filtered_file_list.append(f)

    file_list = sorted(filtered_file_list, key=lambda x: int(x.split('.')[0]))

    for filename in file_list:
        
        file_path = os.path.join(folder_path, filename)
        frame = cv2.imread(file_path) 
        
        if frame is not None:

            # Extract all the digits from left to right
            digits = frame2digits(frame, roi)
            
            # Save all digits
            for digit in digits:
                digits_binary.append(digit)

    return digits_binary

digits_binary = build_testing(folder_path, roi)

# Load the trained model
model = cv2.ml.SVM_load("model.xml")
model_input = np.array([img.flatten() for img in digits_binary], dtype=np.float32)

_, predicted = model.predict(model_input)

test_label = [0, 0, 0, 4, 5, # 1.jpg
              0, 1, 6, 1, 9, # 2.jpg
              0, 3, 2, 6, 2, # 3.jpg
              0, 3, 4, 4, 8, # 4.jpg
              0, 5, 4, 0, 7, # 5.jpg
              0, 6, 0, 4, 3,] # 6.jpg

plt.plot(test_label, predicted, 'o')
plt.xlabel('Labelled')
plt.ylabel('Predicted')
plt.show()
