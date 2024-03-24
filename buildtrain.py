import os
import cv2
import numpy as np

import pickle

from video2res import frame2digits

"""
Build and save the training set from images in folder "train".
Also include all labels for each training images.
"""

folder_path = '/Users/terekli/Desktop/video2num/train/'

roi = np.array([[132.4, 50], [129.2, 215.1], [507.2, 66.1], [496.3, 224.8]]) # 14g, 20psi

def build_training(folder_path, roi):    

    digits_binary = []

    file_list = os.listdir(folder_path)
    
    # Filter out unwanted entries
    filtered_file_list = []
    for f in file_list:
        # Check if the file ends with '.jpg'
        if f.endswith('.jpg'):
            filtered_file_list.append(f)

    file_list = sorted(filtered_file_list, key=lambda x: int(x.split('.')[0]))

    for filename in file_list:

        # print(filename)
        
        file_path = os.path.join(folder_path, filename)
        frame = cv2.imread(file_path) 
        
        if frame is not None:

            # Extract all the digits binary from left to right
            digits = frame2digits(frame, roi)

            # Save all digits
            for digit in digits:
                digits_binary.append(digit)
            
    return digits_binary

digits_binary = build_training(folder_path, roi)

digits_label = [0,0,0,0,1, # 1.jpg
                0,5,9,2,1, # 2.jpg
                0,9,6,7,9, # 3.jpg
                1,3,1,9,6, # 4.jpg
                1,0,7,7,0, # 5.jpg
                0,7,4,2,5, # 6.jpg
                1,9,7,9,7, # 7.jpg
                0,2,7,0,5, # 8.jpg
                0,5,6,3,2, # 9.jpg
                0,1,1,1,7, # 10.jpg
                0,2,7,2,1, # 11.jpg
                0,1,6,9,2, # 12.jpg
                0,5,6,3,2, # 13.jpg
                1,0,9,2,4, # 14.jpg
                0,0,6,3,8, # 15.jpg
                1,3,0,7,1, # 16.jpg
                0,7,6,8,3, # 17.jpg
                0,0,0,4,6, # 18.jpg
                1,4,2,7,4, # 19.jpg
                1,6,4,7,2, # 20.jpg
                1,4,5,6,4, # 21.jpg
                1,0,0,5,3, # 22.jpg
                2,4,5,1,7, # 23.jpg
                2,2,4,0,6, # 24.jpg
                1,2,6,3,7, # 25.jpg
                0,6,0,4,3, # 26.jpg
                0,0,8,3,7, # 27.jpg
                0,2,4,8,7, # 28.jpg
                0,8,6,3,7] # 29.jpg

with open('digits_binary.pickle', 'wb') as file:
    pickle.dump(digits_binary, file)

with open('digits_label.pickle', 'wb') as file:
    pickle.dump(digits_label, file)