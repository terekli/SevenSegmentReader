import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imutils
# from matplotlib import pyplot as plt

######################################## START OF merge_rectangles ########################################
def merge_rectangles(rectangles):
    """
    Function:
    Merges all the rectangles of identified contour according to X domain.
    # Rectangles with overlapping X domain are merged from the bottom left to top right corner.

    Input:
    List of all the rectangles on the screen in the format: (x, y, w, t).
    (x, y): coordinate of the bottom left point
    (w, t): width and thickness

    Output:
    List of merged rectangles in the same format of (x, y, w, t).
    Each represent a single digit, left to right on screen correspond to first to last in the list
    """

    # Filter out rectangles with an area less than 100 pixels
    filtered_rectangles = [rect for rect in rectangles if rect[2]*rect[3] >= 100]
    
    # Sort the filtered rectangles by their starting x-coordinate
    rectangles_sorted = sorted(filtered_rectangles, key=lambda x: x[0])
    
    merged_rectangles = []
    # Check if there are any rectangles to process after filtering
    if not rectangles_sorted:
        return merged_rectangles
    
    current_merge = rectangles_sorted[0]

    for rect in rectangles_sorted[1:]:
        # Check if the current rectangle overlaps with the merge-in-progress
        if rect[0] <= current_merge[0] + current_merge[2]:
            # Update the merge-in-progress to include the current rectangle
            new_x = min(current_merge[0], rect[0])
            new_y = min(current_merge[1], rect[1])
            new_w = max(current_merge[0] + current_merge[2], rect[0] + rect[2]) - new_x
            new_h = max(current_merge[1] + current_merge[3], rect[1] + rect[3]) - new_y
            current_merge = (new_x, new_y, new_w, new_h)
        else:
            # The current rectangle does not overlap, so finalize the current merge
            merged_rectangles.append(current_merge)
            current_merge = rect

    # Add the last merge-in-progress to the list
    merged_rectangles.append(current_merge)

    return merged_rectangles
######################################## END OF merge_rectangles ########################################



######################################## START OF digits2val ########################################
def digits2val(model, digits, num_decimal_places):

    import numpy as np

    """
    Function:
    Process all the digits in a frame and interpret the value

    Input:
    model: trained SVM model
    digits: a list, each element represents a single digit in the form of 50 * 50 array

    Output:
    val: floating point value
    """
    model_input = np.array([img.flatten() for img in digits], dtype=np.float32)

    _, predited = model.predict(model_input)

    # Convert list of digits to a single number
    val = 0
    for digit in predited:
        val = val * 10 + digit

    # Take into account the decimal
    val = val / 10 ** num_decimal_places

    val = np.round(float(val), num_decimal_places)

    return val
######################################## END OF digits2val ########################################



######################################## START OF frame2digits ########################################
def frame2digits(frame, roi, iframe):

    """
    Function:
    Isolate all digits in a frame into 50 * 50 array

    Input:
    frame: the frame
    roi: region of interest, (4,2) array indicating the (x,y) coordinates of the four corners of display

    Output:
    List, each element represents a single digit in the form of 50 * 50 array

    """
    
    digits = [] # Output

    # Perform four point transform
    corrected = four_point_transform(frame, roi)
  
    # Resize and standardize the frame
    scaled = imutils.resize(corrected, height=100)
        
    # convert to gray scale
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=41,C=3)

    # Get contours
    cnts = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Remove the contour if too small or too large
    # Reduce noise and remove the decimals
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= 50]
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) <= 5000]

    # # Used for debugging to visualize the contours
    # colored_image_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    # for _, cnt in enumerate(cnts):
    #     cv2.drawContours(colored_image_rgb, [cnt], -1, color=(255, 0, 0), thickness=1)
    #     plt.imshow(colored_image_rgb)
    #     plt.pause(1)

    # Remove anything outside the identified contour
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, cnts, -1, color=255, thickness=cv2.FILLED)
    inverted_mask = cv2.bitwise_not(mask)
    binary = cv2.bitwise_or(binary, inverted_mask)

    # Calculate bounding box of each identified segment
    bounding_rect = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        bounding_rect.append([x, y, w, h])

    # # Used for debugging to visualize the result
    # binary_3_channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # for rect in bounding_rect:
    #     (x, y, w, h) = rect
    #     cv2.rectangle(binary_3_channel, (x, y), (x+w, y+h), (0, 0, 255), 1)
    # plt.imshow(binary_3_channel)
    # plt.show()

    # Merge bounding boxes to extract single digit
    digits_rectangles = merge_rectangles(bounding_rect)

    # Neglect bounding boxes if height is too small
    digits_rectangles = [arr for arr in digits_rectangles if arr[3] >= 30]  

    # Make output
    for (x, y, w, h) in digits_rectangles:
        # Extract each single digit
        digit = binary[y:y+h, x:x+w]
        # Resize to 50 pixels in height
        digit = imutils.resize(digit, height=50)
        # Add padding to 50 pixels in width
        width = digit.shape[1]
        if width < 50:
            # Calculate how much padding is needed to each side
            padding_left = (50 - width) // 2
            padding_right = 50 - width - padding_left
            # Add padding
            digit = cv2.copyMakeBorder(digit, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=255)
        digits.append(digit)
    
    return digits
######################################## END OF frame2digits ########################################



######################################## START OF video2res ########################################
def video2res(input_path, roi, num_decimal_places):

    """
    Function:
    Process a video and record reading vs timestamp

    Input:
    input_path: path of the video to be processed
    roi: region of interest, (4,2) array indicating the (x,y) coordinates of the four corners of display
    num_decimal_places: how many decimals places exist
    ex: 5.1953 --> 4

    Output:
    res: Panda dataframe. 
    column 1: frame count, column 2: timestamp in ms, column 3: reading

    """
    # Load the trained model
    model = cv2.ml.SVM_load("model.xml")

    # Declare output
    res = []

    iframe = 1

    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    # Retract and process frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Get timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Isoalte each digits
        digits = frame2digits(frame, roi, iframe)

        # Temporary fix, if more than 5 digits are identified then skip this frame
        if len(digits) > 5:
            continue

        # Make predictions
        val = digits2val(model, digits, num_decimal_places)

        # print(iframe, val)

        # Add to output
        res.append([iframe, timestamp, val])

        iframe += 1

        if not ret:
            break 

    return res
######################################## END OF video2res ########################################