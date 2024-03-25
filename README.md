# SevenSegmentReader
Automates extracting and recording numerical data from seven-segment displays in videos, providing time-stamped readings for digital monitoring.

## Background
When working on my PhD thesis, I needed to measure and record the change in mass of a system over time at high sampling frequency (>2Hz). While laboratory digital balance has the required sampling freqncy, its data recording capability is limited to only 0.5Hz. Video capturing the balance display can record the mass reading over time, but extracting the values into numerical data is exhuastingly time demanding. Therefore, I decided to build my own machine vision and machine learning pipeline to process the video recordings.

## Goal
I wanted the code to be extremely efficient in terms of speed with reasonable accuracy, as I have large amount of video files to process.

## Performanc Evaluation
The output of the code is shown below with excellent accuracy. Most importantly, its speed is 120 times faster than counterpart using CNN.
![demo](/output.png)

# Processing Pipeline
1. The user manually records the corner (x, y) coordinates of the Region of Interest.
2. Four point transform is performed on each frame of the video followed by scaling to 100 pixels in height.
3. After transforming to gray scale, aggresive adaptive thresholding to applied to convert the image to binary while minimizing the effect of refleciton and shadow.
4. Contours of all objects are calcualted, and region of the image with small contours are removed to reduce noise and neglect decimal point.
5. Bounding boxes of all contours are obtained, and merged according to overlap in x-coordinate. This efficienctly extract each single digit as a binary map.
6. Each binary map is scaled to 50 * 50 pixels to ensure consistently as shown below.
   ![binary_map](/binary_map.png)
7. A total of 30 images in "train" are used to train a SVM model to identify binary maps.
8. The predicted values are processed into the final output to match the balance display reading.

