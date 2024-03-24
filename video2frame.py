######################################## START OF video2frame ########################################
def video2frame(input_path, output_path):

    import os
    import cv2

    """
    Function:
    Process a video and save each frame.
    Helpful for debugging.

    Input:
    input_path: path of the video to be processed
    output_path: where the processed will be saved

    Output:
    Each frame is saved as a .jpg file with the filename:
    frame{frame_count}_{timestamp}.jpg"

    """
    
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
    
        # Get timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Format timestamp (for debugging)
        formatted_timestamp = f"{timestamp:.0f}ms"
        
        # Save output
        frame_filename = os.path.join(output_path, f"frame{frame_count}_{formatted_timestamp}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

        if not ret:
            break
######################################## END OF video2frame ########################################

input_path = '/Users/terekli/Desktop/video2num/'
output_path = '/Users/terekli/Desktop/video2num/dump/'
video2frame(input_path, output_path)
