###Pipeline:
 1. prepare calibration data
 2. upload calibration data to [kitti calibration server](http://www.cvlibs.net/software/calibration/index.php)
 3. download the calibration result
 4. use this tool and rectify new data.


###data
 please  refer to following document before preparing data
  * calibration_data_example : left/right frame, lidar and config file
  * calibration_result_example : 
	* calib_cam_to_cam.txt: camera calibration result
    * calib_cam_to_range.txt : rotation and translation between left camera and lidar
    
###setting
  * chessboard square length : 8 cm
  * chessboard size :  6 x 8 or 8 x 12
  * chessboard number : 12
  * baseline ： 57 cm
  * camera resolution (before rectifing) ： 768 x 1024
  * camera resolution (after rectifing) : 702 x 945
  * Fov : 60
  
###calibration
  * refer to introduction of [kitti calibration server](http://www.cvlibs.net/software/calibration/index.php)  
 
###rectify

####Example

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

calibration = rectify.Stereo_Lidar(input_dir = calibration_dir,baseline = 0.57)
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
```

### Outlier: 
some outlier still exist in the undistorted disparity,you need to use stereo matching algorithm to remove them.





  
