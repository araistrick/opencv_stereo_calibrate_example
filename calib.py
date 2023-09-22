# Adapted from https://github.com/bvnayak/stereo_calibration/blob/master/camera_calibrate.py

import argparse
import pickle
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np

pattern_shape = (9, 6)

def detect_calib_points(img, pattern_shape, target_type, subpix_criteria=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if target_type == 'chessboard':
        found, points = cv2.findChessboardCorners(gray, pattern_shape, None)
    elif target_type == 'assym_circles':
        found, points = cv2.findCirclesGrid(gray, pattern_shape, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    else:
        raise ValueError(f"Unrecognized OpenCV Calib Target Type '{target_type}'")

    if found:
        if subpix_criteria is None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        points = cv2.cornerSubPix(gray, points, (11, 11), (-1, -1), criteria)

    return found, points

def extract_detections(frames, target_dim):

    n_points = pattern_shape[0] * pattern_shape[1]

    grid = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]] * target_dim / 1000
    points_3d = np.zeros((len(frames), n_points, 3), np.float32)
    points_3d[:, :, :2] = grid.T.reshape(-1, 2)

    points_2d = np.zeros((len(frames), n_points, 2), dtype=np.float32)
    founds = np.zeros(len(frames), dtype=bool)

    print(f'Running checkerboard detection on [{frames[0]}, ... ]')
    for i, path in enumerate(tqdm(frames)):
        img = cv2.imread(str(path))
        founds[i], p2d = detect_calib_points(img, pattern_shape, target_type='chessboard')
        if founds[i]:
            points_2d[i] = p2d.squeeze()

    return founds, points_2d, points_3d

def viz_calib_pnp_results(frames, pred, gt, out_folder, n=None):

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    
    print(f'Visualizing calib-pnp results to {out_folder.absolute()}')
    for i, path in enumerate(tqdm(frames)):

        if n is not None and i > n:
            break

        plt.figure(figsize=(16, 9))
        plt.imshow(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        plt.scatter(gt[i, :, 0], gt[i, :, 1], c='C0', s=5)
        plt.scatter(pred[i, :, 0], pred[i, :, 1], c='C1', s=3)
        plt.savefig(out_folder/f'{i:04d}.png')
        plt.close()

def viz_all_checkerboard_detections(p2d, backdrop_path, path):

    plt.figure(figsize=(16, 9))
    plt.imshow(plt.imread(backdrop_path))

    pts_flat = p2d.reshape(-1, 2)
    plt.scatter(pts_flat[:, 0], pts_flat[:, 1])

    plt.savefig(path)
    plt.close()

def viz_checkerboard_orderings(frame_pairs, p2d_l, p2d_r, viz_folder, n=None):

    viz_folder.mkdir(exist_ok=True, parents=True)

    def load_and_checkerboard(path, points):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.drawChessboardCorners(img, pattern_shape, points, True)

    print(f'Visualizing checkerboard detection pairs to {viz_folder}')
    for i, (left_path, right_path) in enumerate(tqdm(list(frame_pairs))):

        if n is not None and i > n:
            break
        
        left = load_and_checkerboard(left_path, p2d_l[i])
        right = load_and_checkerboard(right_path, p2d_r[i])
        both = np.hstack([left, right])
        cv2.imwrite(str(viz_folder/left_path.parts[-1]), both)

def compute_stereo_calib(p3d, p2d_l, p2d_r, found_l, found_r, img_shape):

    '''

    Let N=num images, M=num points

    p3d: NxMx3 float32. GT 3D locations of checkerboard points
    p2d_l: NxMx2 float32. 2D detection coords of checkerboard points in left img 
    p2d_r: NxMx2 float32. 2D detection coords of checkerboard points in right img 
    found_l: N bool. True when left image i has complete 2D detections
    found_r: N bool. True when right image i has complete 2D detections

    '''

    # Individually calibrate each camera
    print(f'Calibrating cameras on {found_l.sum()} left and {found_r.sum()} right examples')
    flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
    width, height = img_shape
    assert width > height, 'Image dimensions are probably flipped'
    K_guess = None 

    err_l, K_l, d_l, _, _ = cv2.calibrateCamera(list(p3d[found_l]), list(p2d_l[found_l]), 
        img_shape, K_guess, None, flags=flags)
    err_r, K_r, d_r, _, _ = cv2.calibrateCamera(list(p3d[found_r]), list(p2d_r[found_r]), 
        img_shape, K_guess, None, flags=flags)

    # Calibrate jointly
    both = found_l * found_r
    print(f'Calibrating stereo on {both.sum()} examples')
    stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 500, 1e-5)
    stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT
    err_stereo, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        list(p3d[both]), list(p2d_l[both]), list(p2d_r[both]), 
        K_l, d_l, K_r, d_r, img_shape, flags=stereo_flags, criteria=stereo_criteria)
    
    # Compute Change of Basis matrix
    l2r = np.eye(4)
    l2r[:3, :3] = R
    l2r[:3, -1] = T.squeeze()

    # compute other occasionally useful formats for the intrinsic
    fx, fy = K1[[0, 1], [0, 1]]
    cx, cy = K1[:-1, -1]
    calib_vec = np.array([fx, fy, cx, cy])
    calib_vec_dist = np.hstack([calib_vec, d_l.reshape(-1)])

    #Rectify
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, d1, K2, d2, 
        img_shape, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    calib_dict = {
        'errors': {
            'left': err_l,
            'right': err_r,
            'stereo': err_stereo,
        },
        'left': {
            'K': K1,
            'd': d_l,
            'vec': calib_vec,
            'vec_dist': calib_vec_dist
        },
        'right': {
            'K': K2,
            'd': d_r
        },
        'stereo': {
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'left_to_right': l2r
        },
        'rectify': {
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'roi_left': roi_left,
            'roi_right': roi_right
        },
        'img_shape': img_shape
    }

    return calib_dict

def output_calib_files(calib_dict, folder, prefix='calib'):

    with (folder/f'{prefix}_dict.pickle').open('wb') as f:
        pickle.dump(calib_dict, f)

    # save intrinsic matrix K
    np.save(folder/f'{prefix}_left.npy', calib_dict['left']['K'])

    # save DROID-SLAM format with and without distortion
    with (folder/f'{prefix}_left.txt').open('w') as f:
        f.write(' '.join(calib_dict['left']['vec'].astype(str)))
    with (folder/f'{prefix}_dist_left.txt').open('w') as f:
        f.write(' '.join(calib_dict['left']['vec_dist'].astype(str)))

def process_video(video_folder, args):

    left_frames = np.array(sorted(list((video_folder/'left').iterdir())))
    right_frames = np.array(sorted(list((video_folder/'right').iterdir())))
    
    assert len(left_frames) == len(right_frames)

    if args.n_frame_samples != -1:
        print(f'Sampling --n_frame_samples={args.n_frame_samples} frames, reduction to {args.n_frame_samples/len(left_frames)*100:.2f}% of original')
        sample_idx = np.linspace(0, len(left_frames) - 1, args.n_frame_samples).astype(int)
        left_frames = left_frames[sample_idx]
        right_frames = right_frames[sample_idx]

    height, width = cv2.imread(str(left_frames[0])).shape[:-1]

    found_l, p2d_l, p3d_l = extract_detections(left_frames, target_dim=args.target_grid_mm)
    found_r, p2d_r, p3d_r = extract_detections(right_frames, target_dim=args.target_grid_mm)

    if args.viz:
        viz_folder = video_folder/'viz'
        viz_folder.mkdir(exist_ok=True, parents=True)

        # Visualize checkerboard detections and orderings
        both = found_l * found_r
        viz_checkerboard_orderings(zip(left_frames[both], right_frames[both]), p2d_l[both], p2d_r[both], 
            viz_folder/'checkerboard_detection_pairs', n=args.n_viz)

        # Visualize what parts of the image the checkerboard shows up in
        viz_all_checkerboard_detections(p2d_l, left_frames[0], 
            viz_folder/'checkerboard_point_dist_left.png')
        viz_all_checkerboard_detections(p2d_r, right_frames[0], 
            viz_folder/'checkerboard_point_dist_right.png')

    calib_dict = compute_stereo_calib(p3d_l, p2d_l, p2d_r, found_l, found_r, (width, height))

    data_folder = video_folder/'data'
    data_folder.mkdir(exist_ok=True)
    output_calib_files(calib_dict, data_folder)

    err_l, err_r, err_stereo = calib_dict['errors']['left'], calib_dict['errors']['right'], calib_dict['errors']['stereo']
    print(f"SUMMARY for video {video_folder.parts[-1]} w/ shape={(width, height)}")
    print(f"   Total Calibration Errors: Left={err_l:.3f}, Right={err_r:.3f}, Stereo={err_stereo:.3f}")
    print(f"   distortion {calib_dict['left']['d'].reshape(-1).round(3)}")
    print(calib_dict['left']['K'].astype(int))

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=Path)
    parser.add_argument('--target_grid_mm', type=float, default=24.7)
    parser.add_argument('--n_frame_samples', type=int, default=-1)
    parser.add_argument('--compute_poses', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--n_viz', type=int, default=10)
    args = parser.parse_args()

    process_video(args.input_folder, args)

if __name__ == "__main__":
    main()