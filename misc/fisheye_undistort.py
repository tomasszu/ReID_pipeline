import cv2
import numpy as np

# Load the fisheye image
image = cv2.imread('source_images/fisheye_test/01_fisheye_day_000000.jpg')

K = np.array([[2000.11407173, 0., 700.87320043],
       [  0., 1800.28079025, 500.68920531],
       [  0.,   0.,   1.]])
D = np.array([[-110.91414244e-02],
       [-400.60198728e-03],
       [-300.02912651e-04],
       [ 200.83586453e-05]])

# K = np.array([[541.11407173, 0., 659.87320043],
#        [  0., 541.28079025, 318.68920531],
#        [  0.,   0.,   1.]])
# D = np.array([[-3.91414244e-02],
#        [-4.60198728e-03],
#        [-3.02912651e-04],
#        [ 2.83586453e-05]])


new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (1280, 720), np.eye(3), balance=1)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (1280, 720), cv2.CV_16SC2)
undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

# Display the undistorted image
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Save the undistorted image
# cv2.imwrite('undistorted_image.jpg', undistorted_image)