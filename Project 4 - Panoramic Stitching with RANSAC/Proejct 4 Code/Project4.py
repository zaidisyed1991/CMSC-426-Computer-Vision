import cv2
import numpy as np
from scipy.spatial import distance

## Intro to homographies ##

# Read the images
img1 = cv2.imread("./set1/1.jpg")
img2 = cv2.imread("./set1/2.jpg")
img3 = cv2.imread("./set1/3.jpg")

# Create the transformation matrices
height, width = img1.shape[:2]
angle = -10  # clockwise rotation
M_rotate = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
M_translate = np.float32([[1, 0, 100], [0, 1, 0]])
M_scale = np.float32([[0.5, 0, 0], [0, 0.5, 0]])

# Convert 2x3 affine matrices to 3x3 homography matrices
M_rotate_h = np.vstack([M_rotate, np.array([0, 0, 1])])
M_translate_h = np.vstack([M_translate, np.array([0, 0, 1])])
M_scale_h = np.vstack([M_scale, np.array([0, 0, 1])])

# Apply the transformations using cv2.warpPerspective()
output_size = (1000, 800)
img1_rotated = cv2.warpPerspective(img1, M_rotate_h, output_size)
img2_translated = cv2.warpPerspective(img2, M_translate_h, output_size)
img3_scaled = cv2.warpPerspective(img3, M_scale_h, output_size)

# Save the transformed images
cv2.imwrite("1_rotated.jpg", img1_rotated)
cv2.imwrite("2_translated.jpg", img2_translated)
cv2.imwrite("3_scaled.jpg", img3_scaled)


##  Panoramic Stitching ## 

# 1) Compute SIFT features
def compute_sift_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

# 2) Match features
def find_best_matches(des1, des2, num_best_points=100):  # Add a parameter to specify the number of best points
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Sort matches based on their distance
    sorted_matches = sorted(matches, key=lambda x: x[0].distance)

    # Select the best matches
    best_matches = []
    for m, n in sorted_matches[:num_best_points]:  # Slice the sorted matches to get the best points
        best_matches.append(m)

    return best_matches


# 3) Estimate the homographies
def find_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

# 4) Warp images
def warp_images(img, H, translation=None, output_size=None):
    height, width = img.shape[:2]
    if output_size is None:
        output_size = (width, height)
    if translation is not None:
        H_translation = np.float32([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
        H = np.matmul(H_translation, H)
    warped_img = cv2.warpPerspective(img, H, output_size)
    return warped_img

# Read the images
img1 = cv2.imread("./set1/1.jpg")
img2 = cv2.imread("./set1/2.jpg")
img3 = cv2.imread("././set1/3.jpg")

# Compute SIFT features for each image
kp1, des1 = compute_sift_features(img1)
kp2, des2 = compute_sift_features(img2)
kp3, des3 = compute_sift_features(img3)

# Match features between images
matches_12 = find_best_matches(des1, des2)
matches_23 = find_best_matches(des3, des2)

# Find homographies
H12 = find_homography(kp1, kp2, matches_12)
H32 = find_homography(kp3, kp2, matches_23)

# Warp images
translation = (350, 300)
output_size = (img2.shape[1] + translation[0], img2.shape[0] + translation[1])
img1_aligned = warp_images(img1, H12, translation, output_size)
img3_aligned = warp_images(img3, H32, translation, output_size)
img2_translated = warp_images(img2, np.eye(3), translation, output_size)

# Fuse images
fused_image = np.maximum(np.maximum(img1_aligned, img2_translated), img3_aligned)

# Save the fused image
cv2.imwrite("fused_image.jpg", fused_image)
