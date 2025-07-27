
# OpenCV Stereo Calibrate Example

I made these in 2022 when I couldnt find a solid of example of how to use the opencv-python mono/stereo calibration tools. I used them for a project which was ultimately discontinued. They achieved OK calibration but you may want to dial in the particular settings options used in calib.py. There is no guarantee they are correct or good, but maybe this repository will still be useful you.

## Data Collection

I captured video from a stereo camera. You can save this as either a set of png frames or an mp4 file. If your camera outputs both left and right as a single image, or as separate images, either is fine and we will handle that below. 

You should also collect a set of calibration images to data/calib/raw/\*.png. Each frame should show an unobstructed view of the standard opencv chessboard pattern. If part of the checkerboard is out of frame, this is not ideal, but my code will handle it ok IIRC. The chessboard must be flat - I recommend printing out `pattern.png` and using a thin layer of glue to attach it to something rigid, ideally flat glass as it will not deform. Then, to collect the calibration images, mount your camera on a tripod and hold the chessboard at many different locations and orientations, covering the entire field of view. For each image, hold the chessboard still and take a photo - this avoids motion blur or rolling shutter. 

`mkdir data`

If your stereo data is horizontally stacked (IE each image contains both a left and right view) you can use the following ffmpeg command to split them:
```
cd data/calib
mkdir left right
ffmpeg -pattern_type glob -i "raw/*.png" -filter_complex "[0]crop=iw/2:ih:0:0[left];[0]crop=iw/2:ih:ow:0[right]" -map "[left]" left/%04d.png -map "[right]" right/%04d.png
```

If your data is an mp4 you can either adapt the command above or use some other ffmpeg command to unpack it into data/left/*.png and data/right/*.png

To get absolute scale, you will also need to measure the sidelength of a single chessboard square in mm. The best way to do this is to measure several squares using a caliper or ruler, then divide by the number of squares. This is necessary because each printer might print the chessboard at slightly different scale. 

## Calibrate

`python calib.py data/calib --target_dim_mm $MY_MEASUREMENT`

This will create calib files in many formats, including pickle, txt, and npy matrices. 

## Rectify

`python rectify.py data/test --calib data/calib/data/calib_dict.pickle`

This will create data/test/left_rect/*.png and data/test/right_rect/*.png containing the rectified images. 
