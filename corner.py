import cv2
from matplotlib import pyplot as plt

"""
Plot the first frame of a video.
Enable manual extraction of the (x, y) coordinate of the four corners of Region of Interest
"""

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

path = ''
get_corner(path)
