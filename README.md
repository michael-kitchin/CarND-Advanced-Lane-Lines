**Advanced Lane Finding Project**

## Writeup

### Introduction

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Environment

Minimum execution environment is undefined.

Project was developed using the following environment:

| Category | Item        |
|----------|-------------|
| OS       | Windows 10 |
| CPU      | Intel i7/6800k |
| RAM      | 64GB |
| GPU      | nVidia GTX 1060 |
| VRAM     | 6GB |
| Storage  | SATA SSD |

[//]: # (Image References)

[this]: https://github.com/michael-kitchin/CarND-Advanced-Lane-Lines
[cal10_input]: process_output/calibration_images/by_name/calibration10/calibration10_input.png
[cal10_undistorted]: process_output/calibration_images/by_name/calibration10/calibration10_undistorted.png
[cal10_transformed]: process_output/calibration_images/by_name/calibration10/calibration10_transformed.png
[all_input]: media/all_input.png
[all_undistorted]: media/all_undistorted.png
[all_transformed]: media/all_transformed.png
[cal1_input]: calibration_images/calibration1.jpg
[process_py]: process.py
[project_1]: process_output/test_videos/project_video.mp4
[challenge_1]: process_output/challenge_videos/challenge_video.mp4
[challenge_2]: challenge_videos/harder_challenge_video.mp4

[straight1_input]: process_output/test_images/filter/by_name/straight_lines1/straight_lines1_input.png
[straight1_undistorted]: process_output/test_images/filter/by_name/straight_lines1/straight_lines1_undistorted.png
[straight1_sharpened]: process_output/test_images/filter/by_name/straight_lines1/straight_lines1_sharpened.png
[straight1_transformed]: process_output/test_images/filter/by_name/straight_lines1/straight_lines1_transformed.png
[straight1_filtered]: media/straight1_filtered.png
[straight1_output]: process_output/test_images/fit/by_name/straight_lines1/straight_lines1_output.png 
[test1_input]: process_output/test_images/filter/by_name/test1/test1_input.png 
[test1_filtered]: media/test1_filtered.png
[test1_output]: process_output/test_images/fit/by_name/test1/test1_output.png 
[test4_input]: process_output/test_images/filter/by_name/test4/test4_input.png 
[test4_filtered]: media/test4_filtered.png
[test4_output]: process_output/test_images/fit/by_name/test4/test4_output.png 
[test5_input]: process_output/test_images/filter/by_name/test5/test5_input.png 
[test5_filtered]: media/test5_filtered.png
[test5_output]: process_output/test_images/fit/by_name/test5/test5_output.png 

[fr440_input]: media/project_video_000440_input.png
[fr440_undistorted]: media/project_video_000440_undistorted.png
[fr440_sharpened]: media/project_video_000440_sharpened.png
[fr440_transformed]: media/project_video_000440_transformed.png
[fr440_combined]: media/project_video_000440_combined_binary.png
[fr440_output]: media/project_video_000440_output.png
[fr440_result]: media/project_video_000440_result.png
[fr440_filtered]: media/frame_440_filtered.png

[fr400_input]: media/project_video_000400_input.png
[fr400_combined]: media/project_video_000400_combined_binary.png
[fr400_output]: media/project_video_000400_output.png
[fr400_filtered]: media/frame_400_filtered.png

[image_filter_matrix]: media/image_filter_matrix_1.png

### Execution

Implementation is within the [process_py](process.py) script, automating all project capabilities.
 
Supported arguments (defaults are observed, best values):

| Argument | Description | Default / Best Value |
|:-------:|-------------|----------------------|
| `--calibration-image-path` | Calibration image path. | `./calibration_images` |
| `--test-image-path` |  Test image path. | `./test_images` |
| `--test-video-path` |  Test video path. | `./test_videos` |
| `--challenge-video-path` |  Challenge video path. | `./challenge_videos` |
| `--process-output-path` |  Output path for generated images/videos. | `./process_output` |
| `--transform-offset-px` |  Edge offset pixels for transformed images/frames. | `200` |
| `--transform-source-px-multipliers` |  Input image/frame transformation shape (mask), as image percentages. Requires four (4) points (ul/ur/ll/lr). | `[[45.31, 63.61], [55.00, 63.61], [20.86, 91.81], [81.41, 91.81]]` |
| `--y-meters-per-px` |  Y-axis meters-per-pixels for lane curvature/position computation. | `0.04166666666666666666666666666667` |
| `--x-meters-per-px` | X-axis meters-per-pixels for lane curvature/position computation.  | `0.00528571428571428571428571428571` |
| `--video-frame-fit-interval` |  Video frame interval between full-frame (windowed search) edge detection. | `50` |
| `--video-frame-save-interval` |  Video frame interval between saving frame filter/fit reference images. | `40` |
| `--fit-num-windows` |  Number of windows for full-frame edge detection. | `9` |
| `--fit-margin-px` | Pixel margin for full-frame edge detection. | `100` |
| `--fit-min-px` | Minimum number of pixels for windowed edge detection. | `50` |
| `--refit-margin-px` | Minimum number of pixels for non-windowed edge detection. | `100` |
| `--process-test-images` | Process test images (True/False)? | `True` |
| `--process-test-videos` | Process test videos (True/False)?  | `True` |
| `--process-challenge-videos` |  Process challenge videos (True/False)? | `True` |
| `--filter-sobel-kernel-size` | Kernel size for sobel filters (all). | `11` |
| `--line-average-frames` | Number of previous lane lines to average. | `10` |
| `--line-average-prune` | Full lane line recalculation interval. | `100` |

Example execution (Windows, w/in JetBrains IntelliJ):
```
C:\Tools\Anaconda3\envs\bc-project-gpu-1\python.exe C:\Users\mcoyo\.IntelliJIdea2017.3\config\plugins\python\helpers\pydev\pydev_run_in_console.py 57142 57143 E:/Projects/Work/Learning/CarND/CarND-Advanced-Lane-Lines/process.py
Running E:/Projects/Work/Learning/CarND/CarND-Advanced-Lane-Lines/process.py
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['E:\\Projects\\Work\\Learning\\CarND\\CarND-Advanced-Lane-Lines', 'E:/Projects/Work/Learning/CarND/CarND-Advanced-Lane-Lines'])
Python 3.5.4 | packaged by conda-forge | (default, Dec 18 2017, 06:53:03) [MSC v.1900 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
Args: Namespace(calibration_image_path='./calibration_images', challenge_video_path='./challenge_videos', filter_sobel_kernel_size=11, fit_margin_px=100, fit_min_px=50, fit_num_windows=9, line_average_frames=10, line_average_prune=100, process_challenge_videos=True, process_output_path='./process_output', process_test_images=True, process_test_videos=True, refit_margin_px=100, test_image_path='./test_images', test_video_path='./test_videos', transform_offset_px=200, transform_source_px_multipliers='[[45.31, 63.61], [55.00, 63.61], [20.86, 91.81], [81.41, 91.81]]', video_frame_fit_interval=50, video_frame_save_interval=40, x_meters_per_px=0.005285714285714286, y_meters_per_px=0.041666666666666664)
Calibrating camera...
Processing: calibration1.jpg
Processing: calibration10.jpg
Processing: calibration11.jpg

[...]

Camera calibrated
Processing test images...
Processing: straight_lines1.jpg
Processing: straight_lines2.jpg
Processing: test1.jpg

[...]

Test images processed
Processing test videos...
Processing: project_video.mp4
[MoviePy] >>>> Building video ./process_output/test_videos/project_video.mp4
[MoviePy] Writing video ./process_output/test_videos/project_video.mp4
  0%|          | 0/1261 [00:00<?, ?it/s]
  0%|          | 1/1261 [00:00<14:31,  1.44it/s]
  0%|          | 2/1261 [00:01<14:57,  1.40it/s]

[...]
```

---

## Rubric Points

### [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) are discussed individually with respect to the implementation.

---

### 1. Writeup / README

#### 1.1 Provide a Writeup / README that includes all the rubric points and how you addressed each one (...).  

_The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled._

See: GitHub [repo][this].

---

### 2. Camera Calibration

#### 2.1. Briefly state how you computed the camera matrix and distortion coefficients (...).

_OpenCV functions or other methods were used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository (note these are 9x6 chessboard images, unlike the 8x6 images used in the lesson). The distortion matrix should be used to un-distort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is Included in the writeup (or saved to a folder)._

Camera calibration is provided by the `process_camera_calibration` function, as follows:
1. For each (assumed to be) chessboard image in an input folder:
    1. Read into numpy array using `matplotlib.image.mpimg()` to ensure RGB color space
    1. Convert to grayscale via `cv2.cvtColor()`
    1. Find 9x6 chessboard interior corners via `cv2.findChessboardCorners()`; If found:
        1. Refine corners using `cv2.cornerSubPix()`
        1. Render corners over input image using `cv2.drawChessboardCorners()`
        1. Create/update distortion coefficient/matrix using `cv2.calibrateCamera()`
        1. Un-distort input image using `cv2.undistort()`
        1. For reference/verification, create a transform matrix and transform to common interior coordinates
        
Examples follow (calibration image #10). This example is particularly interesting due to circular artifacts generated in the un-distorted image as a by-product of radial correction.

Input:

![cal10_input]

Un-distorted:

![cal10_undistorted]

Transformed:

![cal10_transformed]

All images, input:

![all_input]

All images, un-distorted:

![all_undistorted]

All images, transformed:

![all_transformed]

Calibration image #1 was unusuable, probably due to its outer edges being clipped as shown:

![cal1_input]

---

### 3. Pipeline (Test Images)

Test image processing is provided by the `process_image_files` function, delegating to `filter_street_image`, `fit_lane_lines` and other functions common to both image and video processing.

#### 3.1. Provide an example of a distortion-corrected image.

_Distortion correction that was calculated via camera calibration has been correctly applied to each image. An example of a distortion corrected image should be included in the writeup (or saved to a folder) and submitted with the project._

Calibration coefficients/matrices produced by `process_camera_calibration` are applied to image frames in the `filter_street_image` function, delegated to by `process_video_frame`.

Straight lines image #1 (input):

![straight1_input]

Straight lines image #1 (un-distorted):

![straight1_undistorted]
  

#### 3.2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image (...).

_A method or combination of methods (i.e., color transforms, gradients) has been used to create a binary image containing likely lane pixels. There is no "ground truth" here, just visual verification that the pixels identified as part of the lane lines are, in fact, part of the lines. Example binary images should be included in the writeup (or saved to a folder) and submitted with the project._

A filtering scheme was established as follows:
1. Reflectance- (red/green/blue, hue/saturation/lightness) and gradient-based (x/y/magnitude/directional Sobel) filters were applied.
1. Filters were individually refined for utility in _some subset_ of test images.
1. Performance was documented in the matrix shown below (x = useful, ? = marginal, <blank> = not useful (or) degrading)
1. Image/filter set groupings were identified, as shown:

Gaussian blurring was applied to all gradient-based (Sobel) filters in an attempt to reduce noise pixels.

Image filter matrix (colors = groups, highlights = best filters w/in group):

![image_filter_matrix]

Summarized as follows:
1. Straight #1, Straight #2 (straight, open lanes; asphalt)
    1. Red, Green, Blue
    1. Saturation, lightness (not shown in matrix)
    1. X, magnitude, and directional Sobel
1. Test #1 (curved, open lanes; concrete)
    1. Red, Green, Blue
    1. Saturation and lightness
1. Test #2, Test #3 (curved open lanes; asphalt)
    1. Red, Green, Blue
    1. Saturation, lightness, hue
    1. X, magnitude, and directional Sobel
1. Test #4, Test #6 (curved, partly shaded lanes; asphalt/concrete transition)
    1. Red, Green, Blue
    1. Saturation
1. Test #5 (curved, very shaded lanes; asphalt/concrete transition)
    1. Red, Green, Blue
    1. X and directional Sobel
    
At this point it was speculated sets of filters could be logically aligned with these groups (filters in a group &-ed, all groups |-ed). Attempts at implementation were limited by project time constraints, specifically due to:
* Fine-tuning required for gradient-based filters in circumstances such as group #4 (above)
* Deconflicting permutations of filters categorically common to multiple groups (e.g., red for all groups) but requiring different thresholds, Sobel filter kernels, etc. for exclusive subsets

Instead, the following was implemented in the `filter_street_image` function to maximize best- and average-case performance:

```python
combined_bin_image[(np.sum([red_bin_image, grn_bin_image, blu_bin_image,
                            sat_bin_image, hue_bin_image, lgt_bin_image,
                            (x_bin_image | y_bin_image),
                            (mag_bin_image | dir_bin_image)], axis=0) > 2)] = 1
```  

...with all reflectance-based filters summed with x/y and magnitude/directional Sobel filters, the latter two halved in significance to minimize their degrading effects (as configured) on outlier sections such as concrete overpasses. Once summed, only pixels with a score of 3 or greater would be forwarded for line detection/fitting.

It is speculated picking suitable combinations of filters would be a good task for a neural network-driven pre-filtering strategy, as follows:
1. Preemptively: 
    1. Determine a fixed number of distinct, manageable variations of lane imagery (e.g., by lane curvature/position/surface, sun angle/cloud cover, fraction/density of shadow)
    1. Manually establish general- to best-case filter combinations for each variation
    1. Manually classify examples of each image variation according to these filter combinations 
    1. Train/validate/test neural networks on these and more examples
1. On board the vehicle:
    1. Classify and apply filter combinations in order of confidence (e.g., top-N selections)
    1. Rank and select results based on fitted lane line conformance with previous, near-term results

The following examples show the breadth of filter results across multiple conditions. The "combined_binary" images represent results forwarded for lane detection/fitting.

Straight lines image #1 (filtered):

![straight1_filtered]

Test image #1 (filtered):

![test1_filtered]

Test image #4 (filtered):

![test4_filtered]

Test image #5 (filtered):

![test5_filtered]

#### 3.3 Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image (...).

_OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view". Transformed images should be included in the writeup (or saved to a folder) and submitted with the project._

Perspective transformation was implemented in the `transform_street_image` function. This function computes or re-uses previously-computed to/from transformation matrices and applies them to input images.

Straight lines image #1 (sharpened):

![straight1_sharpened]

Straight lines image #1 (transformed):

![straight1_transformed]

#### 3.4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

_Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial). Example images with line pixels identified and a fit overplotted should be included in the writeup (or saved to a folder) and submitted with the project._

For image and video processing:
* Principal lane line detection/fitting was implemented in the `fit_lane_lines` function based on a histogram-driven, sliding-window algorithm provided with project source material. 

For video processing, specifically:
* Adjustments to previously-established lane line fits was implemented in the `refit_lane_lines` function, based on a margin-expansion algorithm also provided with project source material.
* Fitted lane lines are discarded whenever (a) there are usable, existing ones and (b) new ones cross the frame vertical center line. This was implemented in the `process_video_frame` function.
* Fitted lane lines are averaged, frame-by-frame for 10 (configurable) frames. This was implemented in the `process_video_frame` function and `RunningMean` class. 

The following input/output image pairs illustrate lane detection/fitting under different conditions, as follows:
 * Red pixels: left lanes
 * Blue pixels: right lanes
 * Green boxes: detection windows
 * Yellow lines: fitted results

Straight lines image #1 (input):

![straight1_input]

Straight lines image #1 (output):

![straight1_output]

Test image #1 (input):

![test1_input]

Test image #1 (output):

![test1_output]

Test image #4 (input):

![test4_input]

Test image #4 (output):

![test4_output]

Test image #5 (input):

![test5_input]

Test image #5 (output):

![test5_output]

#### 3.5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

_Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle. For the position of the vehicle, you may assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset you're looking for. As with the polynomial fitting, convert from pixels to meters._

Lane position estimation is implemented in the `get_center_diff_in_m` function and is based on project source material. Position estimation relies principally on input image dimensions and previously-established left and right lane fits. 

```python
def get_center_diff_in_m(input_image, left_fit, right_fit):
    """Get distance from lane center in meters."""
    width_in_px = input_image.shape[1]
    height_in_px = input_image.shape[0]
    bottom_left = get_curve_pixels(left_fit, height_in_px)
    bottom_right = get_curve_pixels(right_fit, height_in_px)
    lane_center = int(bottom_left + (bottom_right - bottom_left) / 2.0)
    output_diff = int(width_in_px / 2.0) - lane_center
    return output_diff
```

Lane curvature estimation is implemented in the `get_curve_radius_in_m` function and is based on project source material. Curvature estimation relies principally on generated lane line pixels.

```python
def get_curve_radius_in_m(curve_x_pixels, curve_y_pixels, meters_per_px):
    """Get curve radius in meters from supplied x/y pixels."""
    curve_scaled_pixels = np.polyfit(curve_y_pixels * meters_per_px[0], curve_x_pixels * meters_per_px[1], 2)
    output_radius = ((1 + (2 * curve_scaled_pixels[0] * np.max(curve_y_pixels) * meters_per_px[0]
                           + curve_scaled_pixels[1]) ** 2) ** 1.5) / np.absolute(2 * curve_scaled_pixels[0])
    return output_radius
```

Both functions also rely on on estimated pixel/meter conversion ratios for X and Y axes, also based on project source material.

Results of both functions should be averaged for best performance but were not in this implementation due to project time constraints.

#### 3.6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

_The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified. An example image with lanes, curvature, and position from center should be included in the writeup (or saved to a folder) and submitted with the project._

This was not implemented for test image processing but was for video frames. 

Project video frame #400 (result):

![fr440_result]

---

### 4. Pipeline (Video)

Video processing is provided by the `process_video_files` function, delegating to `process_video_frame` (providing most of the video pipeline), then `filter_street_image`, `fit_lane_lines` and other functions common to both image and video processing.

#### 4.1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (...).

_The image processing pipeline that was established to find the lane lines in images successfully processes the video. The output here should be a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The pipeline should correctly map out curved lines and not fail when shadows or pavement color changes are present. The output video should be linked to in the writeup and/or saved and submitted with the project._

See [project_video.mp4][project_1] for final project video. This proved to be a successful application of this pipeline.

See [challenge_video.mp4][challenge_1] for challenge video #1. This was not as successful as the project video but illuminating. Overall performance was poor due to filter configurations uncalibrated for heavy shadows across (mostly) the left lane line. 

The final challenge video ([harder_challenge_video.mp4][challenge_2]) proved unreadable by the chosen `VideoFileClip` API, as shown:
```
Processing: harder_challenge_video.mp4
Traceback (most recent call last):
  File "C:\Users\mcoyo\.IntelliJIdea2017.3\config\plugins\python\helpers\pydev\pydev_run_in_console.py", line 53, in run_file
    pydev_imports.execfile(file, globals, locals)  # execute the script
  File "C:\Users\mcoyo\.IntelliJIdea2017.3\config\plugins\python\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "E:/Projects/Work/Learning/CarND/CarND-Advanced-Lane-Lines/process.py", line 892, in <module>
    execute_process()

[...]

OSError: [WinError 6] The handle is invalid
Backend Qt4Agg is interactive backend. Turning interactive mode on.
PyDev console: using IPython 6.2.1
Python 3.5.4 | packaged by conda-forge | (default, Dec 18 2017, 06:53:03) [MSC v.1900 64 bit (AMD64)] on win32
```

This was not investigated further due to project time constraints.

---

### 5. Discussion

#### 5.1. Briefly discuss any problems / issues you faced in your implementation of this project (...).

_Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail._

As discussed above, the principal challenge (as approached) was filter selection and configuration. 

Application of additional filter variations/types, convolutions, and other techniques all offer potential improvements for this capability. 
