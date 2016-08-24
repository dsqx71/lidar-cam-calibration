import cv2
import matplotlib.pyplot as plt
import rectify
import pandas as pd


img1 = cv2.imread('/home/xudong/cam_lidar_data_0819/160744/cam/i00000001_cam1_1471594109.966772_1471594065.290156.jpg')
img2 = cv2.imread('/home/xudong/cam_lidar_data_0819/160744/cam/i00000001_cam2_1471594109.966772_1471594065.258702.jpg')
lidar = pd.read_csv('/home/xudong/cam_lidar_data_0819/160744/lidar/i00000001_1471594109.943019.csv',header=None,sep=' ')
lidar = lidar.values

calibration = rectify.Stereo_Lidar('/home/xudong/calibration/result_example/')
ret = calibration.rectify((img1,img2,lidar))

# undistorted img1
plt.figure()
plt.imshow(ret[0])
plt.waitforbuttonpress()
# undistorted img2
plt.figure()
plt.imshow(ret[1])
plt.waitforbuttonpress()

# undistorted depth
plt.figure()
plt.imshow(ret[2])
plt.waitforbuttonpress()

# undistorted disparity
plt.figure()
plt.imshow(ret[3])
plt.waitforbuttonpress()




