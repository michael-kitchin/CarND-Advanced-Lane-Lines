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

### Execution

Implementation is within the [process_py](process.py) script, automating all project capabilities.
 
Supported arguments (defaults are observed, best values):

| Argument | Description | Default / Best Value |
|:-------:|-------------|----------------------|
| `--calibration-image-path` | Calibration image path. | `./calibration_images` |

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

Camera calibration is automated by the `process_camera_calibration` function, as follows:
1. For each (assumed to be) chessboard image in an input folder:
    1. Read into a numpy array using `matplotlib.image.mpimg()` to ensure RGB color space
    1. Covert to grayscale via `cv2.cvtColor()`
    1. Find 9x6 chessboard interior corners via `cv2.findChessboardCorners()`; If found:
        1. Refine corners using `cv2.cornerSubPix()`
        1. Render corners over input image using `cv2.drawChessboardCorners()`
        1. Create/update distortion coefficient/matrix using `cv2.calibrateCamera()`
        1. Un-distort input image using `cv2.undistort()`
        1. For reference/verification, create a transform matrix and transform to common interior coordinates
        
Example follow (calibration image #10). This example is particularly interesting due to circular artifacts generated in the un-distorted image as a by-product of radial correction.

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

#### 3.1. Provide an example of a distortion-corrected image.

_Distortion correction that was calculated via camera calibration has been correctly applied to each image. An example of a distortion corrected image should be included in the writeup (or saved to a folder) and submitted with the project._

#### 3.2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image (...).

_A method or combination of methods (i.e., color transforms, gradients) has been used to create a binary image containing likely lane pixels. There is no "ground truth" here, just visual verification that the pixels identified as part of the lane lines are, in fact, part of the lines. Example binary images should be included in the writeup (or saved to a folder) and submitted with the project._

#### 3.3 Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image (...).

_OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view". Transformed images should be included in the writeup (or saved to a folder) and submitted with the project._

#### 3.4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

_Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial). Example images with line pixels identified and a fit overplotted should be included in the writeup (or saved to a folder) and submitted with the project._

#### 3.5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

_Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle. For the position of the vehicle, you may assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset you're looking for. As with the polynomial fitting, convert from pixels to meters._

#### 3.6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

_The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified. An example image with lanes, curvature, and position from center should be included in the writeup (or saved to a folder) and submitted with the project._

---

### 4. Pipeline (Video)

#### 4.1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (...).

_The image processing pipeline that was established to find the lane lines in images successfully processes the video. The output here should be a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The pipeline should correctly map out curved lines and not fail when shadows or pavement color changes are present. The output video should be linked to in the writeup and/or saved and submitted with the project._

---

### 5. Discussion

#### 5.1. Briefly discuss any problems / issues you faced in your implementation of this project (...).

_Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail._