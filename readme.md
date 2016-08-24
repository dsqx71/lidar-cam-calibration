###data
 please  refer to following document before preparing data
  * data_example : left/right frame, lidar and config file
  * result_example : 
	* calib_cam_to_cam.txt: camera calibration result
    * calib_cam_to_range.txt : rotation and translation between left camera and lidar
    
###setting
  * chessboard square length : 8 cm
  * chessboard size :  6 x 8 or 8 x 12   
  * chessboard number : 12
  * baseline ： 53 cm
  * camera resolution (before rectifing) ： 768 x 1024
  * camera resolution (after rectifing) : 702 x 945
  

###calibration
  * refer to introduction of [kitti calibration server](http://www.cvlibs.net/software/calibration/index.php)  
 
###rectify example

```
import cv2
import matplotlib.pyplot as plt
import rectify
import pandas as pd
import sys

img1_dir = sys[1]
img2_dir = sys[2]
lidar_dir = sys[3]
calibration_dir = sys[4]

img1 = cv2.imread(img1_dir)
img2 = cv2.imread(img2_dir)
lidar = pd.read_csv(lidar_dir,header=None,sep=' ')
lidar = lidar.values

calibration = rectify.Stereo_Lidar(calibration_dir)
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





```
  