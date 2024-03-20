import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

roi = np.array([[40, 70], [930, 93], [40, 475], [910, 478]])

def video2frame(input_path, output_path, roi):
    
    # Check if output_path exist, if not then create the path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 1
    
    # Retract and process frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform four point transform
        corrected = four_point_transform(frame, roi)
        
        # convert to gray scale
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        
        # Convert to binary
        _, binary = cv2.threshold(gray, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
        
        # Some image processing tick
        # threshold = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        # binary = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        
        # plt.imshow(binary)
        # plt.show()

        # Get contours
        cnts = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Calculate area of each identified contour
        # for i, cnt in enumerate(cnts):
            # area = cv2.contourArea(cnt)
            # print(f"Area of contour {i}: {area}")
        
        # Remove the contour if too small, this reduces noise
        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= 50]
        # Remove the contour if too large, this is the edge
        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) <= 10000]
        # cnts = [cnt for cnt in cnts if not (2500 <= cv2.contourArea(cnt) <= 3000)]

        
        # Calculate area of each identified contour
        # for i, cnt in enumerate(cnts):
        #     area = cv2.contourArea(cnt)
        #     print(f"Area of contour {i}: {area}")
        
        # Draw the binary contour
        binary_cnts = np.zeros_like(binary)
        for c in cnts:
            # area = cv2.contourArea(c)
            # print(f"Area: {area}")
            cv2.drawContours(binary_cnts, [c], -1, (255, 255, 255), -1)
            # plt.imshow(output, cmap='gray')
            # plt.pause(0.5)
        # plt.show()
        
        # Calculate bounding box of each identified segment
        bounding_rect = []
        image_with_rects = corrected.copy()
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(binary_cnts,(x,y),(x+w,y+h),(0, 0, 255), 2)
            bounding_rect.append([x,y,w,h])
            cv2.rectangle(image_with_rects, (x, y), (x+w, y+h), (0, 0, 255), 2)
        image_with_rects_rgb = cv2.cvtColor(image_with_rects, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_with_rects_rgb)
        # plt.show()      
        
        # Get timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Format timestamp
        formatted_timestamp = f"{timestamp:.0f}ms"
    
        # Save output
        frame_filename = os.path.join(output_path, f"frame{frame_count}_{formatted_timestamp}.jpg")
        cv2.imwrite(frame_filename, image_with_rects_rgb)
        
        frame_count += 1

        if not ret:
            break  
        
input_path = '/Users/terekli/Desktop/video2num/data/g15_long/g15_100psi.mp4'
output_path = '/Users/terekli/Desktop/video2num/data/g15_long/g15-100-frame/'
video2frame(input_path, output_path, roi)