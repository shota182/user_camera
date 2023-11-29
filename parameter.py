import numpy as np
import cv2

camera_matrix = np.array([[724.50357,   0.     , 313.78345],
                          [0.     , 727.36014, 257.98553],
                          [0.     ,   0.     ,   1.     ]])
[width, height] = [640, 480]
[aperture_width, aperture_height] = [3.6, 2.7]

fovx, fovy, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(camera_matrix, (width, height), aperture_width, aperture_height)

print(f"fovx, fovy = {fovx}, {fovy}")
print(f"focal_length = {focal_length}")
print(f"principal_point = {principal_point}")
print(f"aspect_ratio = {aspect_ratio}")