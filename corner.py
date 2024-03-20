import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Set Matplotlib to use the Qt5 backend
from matplotlib import pyplot as plt

def get_corner(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read the first frame and process
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB for correct color display with Matplotlib
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Plot and show the first frame
        plt.imshow(frame)
        plt.axis('off')  # Optionally hide axis
        plt.show()

    cap.release()

path = '/Users/terekli/Desktop/video2num/data/g15_long/g15_100psi.mp4'
get_corner(path)
