import pickle
import cv2
import numpy as np

"""
Train the seven segments recognition model.

Model input: (n * 2500) ndarray.
n: number of digits to recognize.
Each digit is 50 * 50 = 2500

"""

# Load binary images
with open('train_binary.pickle', 'rb') as file:
    binary = pickle.load(file)
train_data = np.array([img.flatten() for img in binary], dtype=np.float32)

# Load label
with open('train_label.pickle', 'rb') as file:
    label = pickle.load(file)
train_label = np.array(label, dtype=np.int32).reshape(-1, 1) 

# Wrap the training data and label in a cv2.ml_TrainData object
train_data = cv2.ml.TrainData_create(train_data, cv2.ml.ROW_SAMPLE, train_label)

# Create SVM model
model = cv2.ml.SVM_create()
model.setType(cv2.ml.SVM_C_SVC)
model.setKernel(cv2.ml.SVM_LINEAR)
model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

# Train the model
model.train(train_data)  # Now only one argument

# Save the model
model.save("model.xml")

