import argparse
from pathlib import Path
import pickle
import pdb
import configparser

import cv2
import numpy as np

def hsplit(img):
    new_w = img.shape[1] // 2
    return img[:, :new_w], img[:, new_w:]

def rectify(left, right, calib: dict):    

    assert left.shape == right.shape
    shape = left.shape[:-1][::-1]

    rect = calib['rectify']
    map1x, map1y = cv2.initUndistortRectifyMap(calib['left']['K'], calib['left']['d'], 
        rect['R1'], rect['P1'], shape, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib['right']['K'], calib['right']['d'], 
        rect['R2'], rect['P2'], shape, cv2.CV_32FC1)

    left = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    return left, right

def read_conf_calib(path, res='HD'):

    c = configparser.ConfigParser()
    c.read(path)
    
    baseline = c['STEREO']['Baseline']
    RZ = c['STEREO']['RZ_' + res]
    RY = c['STEREO']['CV_' + res]
    RX = c['STEREO']['RX_' + res]

    T = np.array([-c['STEREO']['Baseline'], 0, 0])
    Rz, _ = cv2.Rodrigues(np.array([0, 0, RZ]))
    Ry, _ = cv2.Rodrigues(np.array([0, RY, 0]))
    Rx, _ = cv2.Rodrigues(np.array([RX, 0, 0]))
    R = np.dot(Rz, np.dot(Ry, Rx)) # Rz*Ry*Rx

    raise NotImplementedError

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=Path)
    parser.add_argument('--calib', type=Path)
    args = parser.parse_args()

    if args.calib.suffix == '.conf':
        calib = read_conf_calib(args.calib)
    elif args.calib.suffix in ['.pkl', '.pickle']:
        with args.calib.open('rb') as f:
            calib = pickle.load(f)
    else:
        raise ValueError(f'Unrecognized path suffix for {args.calib}, {args.calib.suffix=}')

    left_frames = np.array(sorted(list((args.input_folder/'left').iterdir())))
    right_frames = np.array(sorted(list((args.input_folder/'right').iterdir())))

    (args.input_folder/'left_rect').mkdir()
    (args.input_folder/'right_rect').mkdir()

    for l, r in zip(left_frames, right_frames):
        left = cv2.imread(str(l))
        right = cv2.imread(str(r))
        left_rect, right_rect = rectify(left, right, calib)
        cv2.imwrite(str(args.input_folder/'left_rect'/l.name), left_rect)
        cv2.imwrite(str(args.input_folder/'right_rect'/r.name), right_rect)

if __name__ == "__main__":
    main()