import cv2
import numpy as np
import os
from tqdm import tqdm
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import matplotlib.pyplot as plt
import argparse


def decompose_essential_matrix(E, src_pts, dst_pts, K):
    if args.custom_recover_pose:
        R, T = recover_pose_custom(E, src_pts, dst_pts, K)
    else:
        _, R, T, _ = cv2.recoverPose(E, src_pts, dst_pts, K)
    return R, T

# Note:
# To use the cv2.recoverPose() us the command line:
# python FinalProject.py
# To use the custom_recover_pose() function use the command line:
# python FinalProject.py --custom_recover_pose


parser = argparse.ArgumentParser()
parser.add_argument("--custom_recover_pose", help="Use the custom recover pose function", action="store_true")
args = parser.parse_args()


###
def triangulate_points(P1, P2, src_pts, dst_pts):
    points_4d = cv2.triangulatePoints(P1, P2, src_pts.T, dst_pts.T)
    points_3d = points_4d / points_4d[3]
    return points_3d[:3]

def count_points_in_front_of_cameras(R, T, src_pts, dst_pts, K):
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, T)))

    points_3d = triangulate_points(P1, P2, src_pts, dst_pts)
    points_in_front = 0

    for i in range(points_3d.shape[1]):
        Z1 = points_3d[2, i]
        Z2 = np.dot(R[2], points_3d[:, i]) + T[2]

        if Z1 > 0 and Z2 > 0:
            points_in_front += 1

    return points_in_front

def recover_pose_custom(E, src_pts, dst_pts, K):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))
    T1 = U[:, 2].reshape(3, 1)
    T2 = -U[:, 2].reshape(3, 1)

    possible_poses = [(R1, T1), (R1, T2), (R2, T1), (R2, T2)]
    max_points_in_front = 0
    best_pose = (None, None)

    for R, T in possible_poses:
        if np.linalg.det(R) < 0:  # Check if the determinant of R is positive
            continue

        points_in_front = count_points_in_front_of_cameras(R, T, src_pts, dst_pts, K)

        if points_in_front > max_points_in_front:
            max_points_in_front = points_in_front
            best_pose = (R, T)

        if points_in_front == src_pts.shape[0]:
            break

    return best_pose


###

def compute_intrinsic_matrix(fx, fy, cx, cy):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K

fx, fy, cx, cy, _, LUT = ReadCameraModel('./Oxford_dataset_reduced/model')
K = compute_intrinsic_matrix(fx, fy, cx, cy)

def load_and_demosaic_image(filename, LUT):
    img = cv2.imread(filename, flags=-1)
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistorted_image = UndistortImage(color_image, LUT)
    return undistorted_image

def find_keypoint_correspondences(img1, img2, good_match_ratio=0.75):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Create FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to keep only good matches
    good_matches = []
    for m, n in matches:
        if m.distance < good_match_ratio * n.distance:
            good_matches.append(m)

    # Extract locations of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])#.reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])#.reshape(-1, 1, 2)

    return src_pts, dst_pts

def estimate_fundamental_matrix(src_pts, dst_pts, method=cv2.FM_RANSAC, ransac_reproj_threshold=1.0):
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, method, ransacReprojThreshold=ransac_reproj_threshold)
    return F

def recover_essential_matrix(F, K):
    E = np.dot(K.T, np.dot(F, K))
    return E

def decompose_essential_matrix(E, src_pts, dst_pts, K):
    if args.custom_recover_pose:
        R, T = recover_pose_custom(E, src_pts, dst_pts, K)
    else:
        _, R, T, _ = cv2.recoverPose(E, src_pts, dst_pts, K)
    return R, T


def accumulate_transforms(R_list, T_list):
    camera_centers = [np.array([0, 0, 0])]
    R_acc = np.eye(3)
    T_acc = np.zeros((3, 1))
    
    for R, T in zip(R_list, T_list):
        R_acc = R_acc @ R
        T_acc = T_acc + R_acc @ T
        camera_centers.append(T_acc.ravel())
    
    return np.array(camera_centers)



# Main loop
image_folder = './Oxford_dataset_reduced/images'
image_filenames = sorted(os.listdir(image_folder))

R_list = []
T_list = []

for i in tqdm(range(len(image_filenames) - 1)):
    filename1 = os.path.join(image_folder, image_filenames[i])
    filename2 = os.path.join(image_folder, image_filenames[i+1])

    img1 = load_and_demosaic_image(filename1, LUT)
    img2 = load_and_demosaic_image(filename2, LUT)

    src_pts, dst_pts = find_keypoint_correspondences(img1, img2)
    F = estimate_fundamental_matrix(src_pts, dst_pts)
    E = recover_essential_matrix(F, K)
    R, T = decompose_essential_matrix(E, src_pts, dst_pts, K)
    if R is not None and T is not None:
        R_list.append(R)
        T_list.append(T)


camera_centers = accumulate_transforms(R_list, T_list)

def plot_trajectory(camera_centers):

    plt.figure(figsize=(10, 10))
    plt.plot(camera_centers[:, 0], -camera_centers[:, 2], 'o-', markersize=4)
    plt.title("My reconstruction of the trajectory (projected onto 2 dimensions)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.savefig("My reconstruction of the trajectory (projected onto 2 dimensions).pdf")
    plt.show()

plot_trajectory(camera_centers)


def plot_trajectory_3d(camera_centers):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(camera_centers[:, 0], -camera_centers[:, 2], camera_centers[:, 1], 'o-', markersize=4)
    ax.set_title("My reconstruction of the trajectory (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(-70, 70)
    plt.savefig("My reconstruction of the trajectory (3D).pdf")
    plt.show()

plot_trajectory_3d(camera_centers)