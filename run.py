"""
Process video and outoput result in a list.
Also plot to visualizd the result
"""
import numpy as np
from matplotlib import pyplot as plt

from video2res import video2res

roi = np.array([[132.4, 50], [129.2, 215.1], [507.2, 66.1], [496.3, 224.8]]) # 14g, 20psi
# roi = np.array([[136.3, 80.9], [510.5, 86.1], [138.9, 243.5], [504.0, 242.2]]) # 14g, 100psi

# roi = np.array([[212, 65], [587, 73], [215, 228], [576, 228]]) # 15g, 100psi

# roi = np.array([[142.7, 83.5], [457.6, 82.2], [141.4, 217], [458.2, 222.9]]) # 20g, 80psi

video_path = '/Users/terekli/Desktop/video2num/trimmed/14g_20psi.mov'

res = video2res(video_path, roi, 4)
res = np.array(res)

# column 1: frame count
# column 2: timestamp (ms)
# column 3: reading

plt.plot(res[:,1], res[:,2], 'o')
plt.ylabel('Reading')
plt.show()