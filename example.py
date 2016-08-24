import cv2
import matplotlib.pyplot as plt
import rectify
import pandas as pd
import numpy as np


img1 = cv2.imread('/home/xudong/cam_lidar_data_0819/160744/cam/i00000001_cam1_1471594109.966772_1471594065.290156.jpg')
img2 = cv2.imread('/home/xudong/cam_lidar_data_0819/160744/cam/i00000001_cam2_1471594109.966772_1471594065.258702.jpg')
lidar = pd.read_csv('/home/xudong/cam_lidar_data_0819/160744/lidar/i00000001_1471594109.943019.csv',header=None,sep=' ')
lidar = lidar.values

calibration = rectify.Stereo_Lidar('/home/xudong/calibration/calibration_result_example/')
ret = calibration.rectify((img1,img2,lidar))

# undistorted img1
plt.figure()
plt.imshow(ret[0])
plt.title('left')
plt.waitforbuttonpress()
# undistorted img2
plt.figure()
plt.imshow(ret[1])
plt.title('right')
plt.waitforbuttonpress()

# undistorted depth
plt.figure()
plt.imshow(ret[2])
plt.title('depth')
plt.waitforbuttonpress()

# undistorted disparity
plt.figure()
plt.imshow(ret[3])
plt.title('disparity')
plt.waitforbuttonpress()

# check disparity to see whether textures of two patch are the same or not
rectify.check_disparity(ret[0],ret[1],ret[3])

# check epipolar line
ret[0][:,:,0] = 0
ret[1][:,:,2] = 0
M = np.float32([[1, 0, 0], [0, 1, 0]])
ret[0]= cv2.warpAffine(ret[0], M, (ret[0].shape[1], ret[0].shape[0]))

img = cv2.addWeighted(ret[0],0.4,ret[1],0.6,1)
plt.figure()
plt.title('check epipolar line')
plt.imshow(img)
plt.waitforbuttonpress()






