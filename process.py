import io
import os
from collections import deque

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

calibration_image_path = './calibration_images'
test_image_path = './test_images'
test_video_path = './test_videos'
challenge_video_path = './challenge_videos'

output_dir_name = 'process_output'
calibration_image_output_path = '{}/calibration_images'.format(output_dir_name)
test_image_output_path = '{}/test_images'.format(output_dir_name)
test_video_output_path = '{}/test_videos'.format(output_dir_name)
challenge_video_output_path = '{}/challenge_videos'.format(output_dir_name)

# Filter constants/counters
distortion_mtx = None
distortion_coeff = None
default_offset_px = 200

# [dst_ul, dst_ur, dst_ll, dst_lr]
# Longer view (straighter roads)
default_source_px_multipliers = [[45.31, 63.61], [55.00, 63.61], [20.86, 91.81], [81.41, 91.81]]
# Shorter view (curvier roads)
# default_source_px_multipliers = [[42.58, 66.39], [57.97, 66.39], [20.86, 91.81], [81.41, 91.81]]

prev_to_transform = None
prev_from_transform = None
y_meters_per_px = 30.0 / 720.0
x_meters_per_px = 3.7 / 700.0

# Video constants/counters
curr_video_frame_ctr = 0
video_frame_fit_interval = 50
video_frame_save_interval = 40
curr_video_base_file_name = None
curr_video_output_path = None
prev_left_fit = None
prev_right_Fit = None
left_line_mean = None
right_line_mean = None


# Simple class for maintaining running averages
class RunningMean:
    def __init__(self, max_size=10, max_prunes=100):
        """Basic ctor"""
        self.max_size = max_size
        self.max_prunes = max_prunes
        self.prune_ctr = 0
        self.cache = deque()
        self.sum = None
        self.curr_mean = None

    def __len__(self):
        """Length operator"""
        return len(self.cache)

    def update_mean(self, new_value):
        """Update running mean"""
        self.cache.append(new_value)
        if self.sum is None:
            self.sum = np.copy(new_value)
        else:
            self.sum = np.add(self.sum, new_value)
        if self.prune():
            self.prune_ctr += 1
            if 0 < self.max_prunes < self.prune_ctr:
                self.prune_ctr = 0
                self.recalc()
        return self.build_mean()

    def clear(self):
        """Clear everything"""
        self.prune_ctr = 0
        self.cache.clear()
        self.sum = None
        self.curr_mean = None

    def recalc(self):
        """"Recalc running sum from values
        (addresses accumulated precision loss)"""
        self.sum = None
        for value in self.cache:
            if self.sum is None:
                self.sum = np.copy(value)
            else:
                self.sum = np.add(self.sum, value)

    def prune(self):
        """Discard old values"""
        result = False
        while len(self.cache) > self.max_size:
            result = True
            self.sum = np.subtract(self.sum, self.cache.popleft())
        return result

    def build_mean(self):
        """Recalc current mean"""
        self.curr_mean = (self.sum / float(len(self.cache)))
        return self.get_mean()

    def get_mean(self):
        """Gets current mean"""
        return self.curr_mean


def setup():
    if not os.path.isdir(output_dir_name):
        os.makedirs(output_dir_name)


def save_image(input_image, output_path, output_type, output_name, output_ext):
    # Create directories
    type_path = '{}/by_type/{}'.format(output_path, output_type)
    if not os.path.isdir(type_path):
        os.makedirs(type_path)
    name_path = '{}/by_name/{}'.format(output_path, output_name)
    if not os.path.isdir(name_path):
        os.makedirs(name_path)

    # Save files
    mpimg.imsave('{}/{}_{}.{}'.format(type_path, output_name, output_type, output_ext), input_image)
    mpimg.imsave('{}/{}_{}.{}'.format(name_path, output_name, output_type, output_ext), input_image)


def save_figure(output_path, output_type, output_name, output_ext):
    # Create directories
    type_path = '{}/by_type/{}'.format(output_path, output_type)
    if not os.path.isdir(type_path):
        os.makedirs(type_path)
    name_path = '{}/by_name/{}'.format(output_path, output_name)
    if not os.path.isdir(name_path):
        os.makedirs(name_path)

    # Save files
    plt.savefig('{}/{}_{}.{}'.format(type_path, output_name, output_type, output_ext))
    plt.savefig('{}/{}_{}.{}'.format(name_path, output_name, output_type, output_ext))


def get_figure_data(input_figure):
    with io.BytesIO() as mem_file:
        input_figure.canvas.draw()
        input_figure.savefig(mem_file, format='PNG')
        mem_file.seek(0)
        return np.asarray(Image.open(mem_file))


def get_new_figure(image_size):
    output_figure = plt.figure()
    figure_dpi = output_figure.get_dpi()
    output_figure.set_size_inches(float(image_size[1]) / float(figure_dpi),
                                  float(image_size[0]) / float(figure_dpi))
    return output_figure


def process_camera_calibration(input_image_path,
                               output_image_path):
    # termination criteria
    sub_pix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    global distortion_mtx
    global distortion_coeff

    distortion_mtx = None
    distortion_coeff = None

    # Step through the list and search for chessboard corners
    for input_file_name in os.listdir(input_image_path):
        print('Processing: {}'.format(input_file_name))
        base_file_name = input_file_name.replace('.jpg', '')

        input_image = mpimg.imread('{}/{}'.format(input_image_path, input_file_name))
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        curr_result, found_corners = cv2.findChessboardCorners(gray_image, (9, 6), None)

        # If found, add object points, image points
        if curr_result == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray_image, found_corners, (11, 11), (-1, -1), sub_pix_criteria)
            imgpoints.append(found_corners)

            # Draw and display the corners
            input_image = cv2.drawChessboardCorners(input_image, (9, 6), found_corners, curr_result)

            # Calibrate
            curr_result, distortion_mtx, distortion_coeff, rotation_vec, translation_vec = \
                cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

            # Undistort
            undistorted_image = undistort_image(input_image, distortion_mtx, distortion_coeff)

            # Transform (for reference)
            source_px = np.float32([found_corners[0], found_corners[8], found_corners[-9], found_corners[-1]])
            image_size = (undistorted_image.shape[1], undistorted_image.shape[0])
            to_transform, from_transform = get_perspective_transform(image_size, source_px)
            transformed_image = transform_image(input_image, to_transform)

            save_image(input_image, output_image_path, 'input', base_file_name, 'png')
            save_image(undistorted_image, output_image_path, 'undistorted', base_file_name, 'png')
            save_image(transformed_image, output_image_path, 'transformed', base_file_name, 'png')


def get_curve_pixels(poly_fit, plot_axis):
    poly_func = np.poly1d(poly_fit)
    return poly_func(plot_axis)


def get_lr_curve_pixels(left_fit, right_fit, input_image):
    ploty_axis = np.linspace(0, input_image.shape[0] - 1, input_image.shape[0])
    left_fitx_line = get_curve_pixels(left_fit, ploty_axis)
    right_fitx_line = get_curve_pixels(right_fit, ploty_axis)
    return left_fitx_line, right_fitx_line, ploty_axis


def get_curve_radius_in_m(fit_pixels, plot_axis, meters_per_px):
    fit_scaled = np.polyfit(plot_axis * meters_per_px[0], fit_pixels * meters_per_px[1], 2)
    output_radius = ((1 + (2 * fit_scaled[0] * np.max(plot_axis) * meters_per_px[0] + fit_scaled[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * fit_scaled[0])
    return output_radius


def get_center_diff_in_m(input_image, left_fit, right_fit):
    width_in_px = input_image.shape[1]
    height_in_px = input_image.shape[0]
    bottom_left = get_curve_pixels(left_fit, height_in_px)
    bottom_right = get_curve_pixels(right_fit, height_in_px)
    lane_center = int(bottom_left + (bottom_right - bottom_left) / 2.0)
    output_diff = int(width_in_px / 2.0) - lane_center
    return output_diff


def undistort_image(input_image,
                    distortion_mtx,
                    distortion_coeff):
    return cv2.undistort(input_image, distortion_mtx, distortion_coeff, None, distortion_mtx)


def transform_image(input_image,
                    perspective_transform):
    image_size = (input_image.shape[1], input_image.shape[0])
    return cv2.warpPerspective(input_image, perspective_transform, image_size,
                               flags=cv2.INTER_CUBIC)


def sharpen_image(input_image):
    return cv2.addWeighted(input_image, 1.5, cv2.GaussianBlur(input_image, (0, 0), 3), -0.5, 0)


def get_perspective_transform(image_size,
                              source_px,
                              offset_px=default_offset_px):
    dst_ul = [offset_px, (offset_px / 2)]
    dst_ur = [image_size[0] - offset_px, (offset_px / 2)]
    dst_lr = [image_size[0] - offset_px, image_size[1] - (offset_px / 2)]
    dst_ll = [offset_px, image_size[1] - (offset_px / 2)]
    dest_px = np.float32([dst_ul, dst_ur, dst_ll, dst_lr])
    source_px_arr = np.float32(source_px)

    to_transform = cv2.getPerspectiveTransform(source_px_arr, dest_px)
    from_transform = cv2.getPerspectiveTransform(dest_px, source_px_arr)

    return to_transform, from_transform


def transform_street_image(input_image,
                           source_px_multipliers=default_source_px_multipliers,
                           to_transform=None,
                           from_transform=None):
    image_size = (input_image.shape[1], input_image.shape[0])

    if to_transform is None \
            or from_transform is None:
        source_px = []
        for item in default_source_px_multipliers:
            source_px.append([round((item[0] / 100.0) * image_size[0]),
                              round((item[1] / 100.0) * image_size[1])])
        to_transform, from_transform = get_perspective_transform(image_size, source_px, default_offset_px)

    return transform_image(input_image, to_transform), to_transform, from_transform


def abs_sobel_thresh(gray_image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel_image = np.absolute(sobel_image)
    scaled_sobel_image = np.uint8(255 * abs_sobel_image / np.max(abs_sobel_image))
    binary_sobel_image = np.zeros_like(scaled_sobel_image)
    binary_sobel_image[(scaled_sobel_image >= thresh[0]) & (scaled_sobel_image <= thresh[1])] = 1

    return binary_sobel_image


def mag_thresh(gray_image, sobel_kernel=3, mag_thresh=(0, 255)):
    sobel_x_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel_image = np.sqrt(np.power(sobel_x_image, 2) + np.power(sobel_y_image, 2))
    scaled_sobel_image = np.uint8(255 * abs_sobel_image / np.max(abs_sobel_image))
    binary_sobel_image = np.zeros_like(scaled_sobel_image)
    binary_sobel_image[(scaled_sobel_image >= mag_thresh[0]) & (scaled_sobel_image <= mag_thresh[1])] = 1

    return binary_sobel_image


def dir_threshold(gray_image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobel_x_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    arctan_sobel_image = np.arctan2(np.absolute(sobel_y_image), np.absolute(sobel_x_image))
    binary_sobel_image = np.zeros_like(arctan_sobel_image, dtype=np.uint8)
    binary_sobel_image[(arctan_sobel_image >= thresh[0]) & (arctan_sobel_image <= thresh[1])] = 1

    return binary_sobel_image


def hls_threshold(img, channel_id='h', thresh=(0, 255)):
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel_id == 'h':
        input_data = hls_image[:, :, 0]
    elif channel_id == 'l':
        input_data = hls_image[:, :, 1]
    else:
        input_data = hls_image[:, :, 2]

    binary_image = np.zeros_like(input_data)
    binary_image[(input_data >= thresh[0]) & (input_data <= thresh[1])] = 1

    return binary_image


def rgb_threshold(img, channel_id='r', thresh=(0, 255)):
    if channel_id == 'r':
        input_data = img[:, :, 0]
    elif channel_id == 'g':
        input_data = img[:, :, 1]
    else:
        input_data = img[:, :, 2]

    binary_image = np.zeros_like(input_data)
    binary_image[(input_data >= thresh[0]) & (input_data <= thresh[1])] = 1

    return binary_image


def hsv_threshold(img, channel_id='h', thresh=(0, 255)):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if channel_id == 'h':
        input_data = hsv_image[:, :, 0]
    elif channel_id == 's':
        input_data = hsv_image[:, :, 1]
    else:
        input_data = hsv_image[:, :, 2]

    binary_image = np.zeros_like(input_data)
    binary_image[(input_data >= thresh[0]) & (input_data <= thresh[1])] = 1

    return binary_image


def filter_street_image(input_image,
                        distortion_mtx,
                        distortion_coeff,
                        to_transform=None,
                        from_transform=None,
                        save_file_path=None,
                        save_file_name=None):
    # Undistort
    undistorted_image = undistort_image(input_image, distortion_mtx, distortion_coeff)

    # Sharpen
    sharpened_image = sharpen_image(undistorted_image)

    # Transform
    transformed_image, to_transform, from_transform = \
        transform_street_image(sharpened_image, default_source_px_multipliers,
                               to_transform=to_transform, from_transform=from_transform)

    # Choose a Sobel kernel size
    gray_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY)

    # Sobel kernel size
    kernel_size = 11

    # Apply each of the thresholding functions
    red_bin_image = rgb_threshold(transformed_image, 'r', thresh=(230, 255))
    grn_bin_image = rgb_threshold(transformed_image, 'g', thresh=(205, 255))
    blu_bin_image = rgb_threshold(transformed_image, 'b', thresh=(200, 255))
    sat_bin_image = hls_threshold(transformed_image, 's', thresh=(150, 255))
    lgt_bin_image = hls_threshold(transformed_image, 's', thresh=(150, 255))
    hue_bin_image = hls_threshold(transformed_image, 'h', thresh=(20, 40))
    x_bin_image = abs_sobel_thresh(gray_image, orient='x', sobel_kernel=kernel_size, thresh=(20, 120))
    x_bin_image = cv2.GaussianBlur(x_bin_image, (0, 0), 3)
    y_bin_image = abs_sobel_thresh(gray_image, orient='y', sobel_kernel=kernel_size, thresh=(20, 120))
    y_bin_image = cv2.GaussianBlur(y_bin_image, (0, 0), 3)
    mag_bin_image = mag_thresh(gray_image, sobel_kernel=kernel_size, mag_thresh=(20, 200))
    mag_bin_image = cv2.GaussianBlur(mag_bin_image, (0, 0), 3)
    dir_bin_image = dir_threshold(gray_image, sobel_kernel=kernel_size, thresh=(0.001, 0.001 + (np.pi / 24.0)))
    dir_bin_image = cv2.GaussianBlur(dir_bin_image, (0, 0), 3)

    # Combine results
    combined_bin_image = np.zeros_like(dir_bin_image)
    combined_bin_image[(np.sum([red_bin_image, grn_bin_image, blu_bin_image,
                                sat_bin_image, hue_bin_image, lgt_bin_image,
                                (x_bin_image | y_bin_image),
                                (mag_bin_image | dir_bin_image)], axis=0) > 2)] = 1

    if not save_file_path is None \
            and not save_file_name is None:
        save_image(input_image, save_file_path, 'input', save_file_name, 'png')
        save_image(undistorted_image, save_file_path, 'undistorted', save_file_name, 'png')
        save_image(sharpened_image, save_file_path, 'sharpened', save_file_name, 'png')
        save_image(transformed_image, save_file_path, 'transformed', save_file_name, 'png')
        save_image(gray_image, save_file_path, 'gray', save_file_name, 'png')
        save_image(red_bin_image, save_file_path, 'red_binary', save_file_name, 'png')
        save_image(grn_bin_image, save_file_path, 'grn_binary', save_file_name, 'png')
        save_image(grn_bin_image, save_file_path, 'blu_binary', save_file_name, 'png')
        save_image(sat_bin_image, save_file_path, 'sat_binary', save_file_name, 'png')
        save_image(lgt_bin_image, save_file_path, 'lgt_binary', save_file_name, 'png')
        save_image(hue_bin_image, save_file_path, 'hue_binary', save_file_name, 'png')
        save_image(x_bin_image, save_file_path, 'x_binary', save_file_name, 'png')
        save_image(y_bin_image, save_file_path, 'y_binary', save_file_name, 'png')
        save_image(mag_bin_image, save_file_path, 'mag_binary', save_file_name, 'png')
        save_image(dir_bin_image, save_file_path, 'dir_binary', save_file_name, 'png')
        save_image(combined_bin_image, save_file_path, 'combined_binary', save_file_name, 'png')

    return combined_bin_image, transformed_image, undistorted_image, to_transform, from_transform


def fit_lane_lines(input_image,
                   num_windows=9,
                   margin_px=100,
                   min_px=50,
                   build_fit_image=True,
                   save_file_path=None,
                   save_file_name=None):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    lower_histogram = np.sum(input_image[int(input_image.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    work_image = np.dstack((input_image, input_image, input_image)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint_px = np.int(lower_histogram.shape[0] / 2)
    base_leftx_px = np.argmax(lower_histogram[:midpoint_px])
    base_rightx_px = np.argmax(lower_histogram[midpoint_px:]) + midpoint_px

    # Set height of windows
    window_height_px = np.int(input_image.shape[0] / num_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero_image = input_image.nonzero()
    nonzero_y_image = np.array(nonzero_image[0])
    nonzero_x_image = np.array(nonzero_image[1])
    # Current positions to be updated for each window
    curr_leftx_px = base_leftx_px
    curr_rightx_px = base_rightx_px
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = input_image.shape[0] - (window + 1) * window_height_px
        win_y_high = input_image.shape[0] - window * window_height_px
        win_xleft_low = curr_leftx_px - margin_px
        win_xleft_high = curr_leftx_px + margin_px
        win_xright_low = curr_rightx_px - margin_px
        win_xright_high = curr_rightx_px + margin_px
        # Draw the windows on the visualization image
        cv2.rectangle(work_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(work_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y_image >= win_y_low) & (nonzero_y_image < win_y_high) &
                          (nonzero_x_image >= win_xleft_low) & (nonzero_x_image < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y_image >= win_y_low) & (nonzero_y_image < win_y_high) &
                           (nonzero_x_image >= win_xright_low) & (nonzero_x_image < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > min_px:
            curr_leftx_px = np.int(np.mean(nonzero_x_image[good_left_inds]))
        if len(good_right_inds) > min_px:
            curr_rightx_px = np.int(np.mean(nonzero_x_image[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzero_x_image[left_lane_inds]
    lefty = nonzero_y_image[left_lane_inds]
    rightx = nonzero_x_image[right_lane_inds]
    righty = nonzero_y_image[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if build_fit_image:
        work_image[nonzero_y_image[left_lane_inds], nonzero_x_image[left_lane_inds]] = [255, 0, 0]
        work_image[nonzero_y_image[right_lane_inds], nonzero_x_image[right_lane_inds]] = [0, 0, 255]

        # Overplot
        left_fitx_line, right_fitx_line, ploty_axis = \
            get_lr_curve_pixels(left_fit, right_fit, input_image)

        fig = get_new_figure(input_image.shape)
        plt.imshow(work_image)
        plt.plot(left_fitx_line, ploty_axis, color='yellow')
        plt.plot(right_fitx_line, ploty_axis, color='yellow')
        plt.xlim(0, input_image.shape[1])
        plt.ylim(input_image.shape[0], 0)

        fit_image = get_figure_data(fig)
        plt.close()

        if not save_file_path is None \
                and not save_file_name is None:
            # Save input image
            save_image(input_image, save_file_path, 'input', save_file_name, 'png')
            save_image(fit_image, save_file_path, 'output', save_file_name, 'png')
        return left_fit, right_fit, fit_image
    else:
        return left_fit, right_fit, input_image


def refit_lane_lines(input_image,
                     left_fit,
                     right_fit,
                     margin=100,
                     build_fit_image=True,
                     save_file_path=None,
                     save_file_name=None):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = input_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = \
        ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                      left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                            left_fit[1] * nonzeroy + left_fit[
                                                                2] + margin)))

    right_lane_inds = \
        ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                      right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                             right_fit[1] * nonzeroy + right_fit[
                                                                 2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if build_fit_image:
        # Create an image to draw on and an image to show the selection window
        work_image = np.dstack((input_image, input_image, input_image)) * 255
        window_img = np.zeros_like(work_image)

        # Color in left and right line pixels
        work_image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        work_image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Overplot
        left_fitx_line, right_fitx_line, ploty_axis = \
            get_lr_curve_pixels(left_fit, right_fit, input_image)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx_line - margin, ploty_axis]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx_line + margin,
                                                                        ploty_axis])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx_line - margin, ploty_axis]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx_line + margin,
                                                                         ploty_axis])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        weighted_work_image = cv2.addWeighted(work_image, 1, window_img, 0.3, 0)

        fig = get_new_figure(input_image.shape)
        plt.imshow(weighted_work_image)
        plt.plot(left_fitx_line, ploty_axis, color='yellow')
        plt.plot(right_fitx_line, ploty_axis, color='yellow')
        plt.xlim(0, input_image.shape[1])
        plt.ylim(input_image.shape[0], 0)

        fit_image = get_figure_data(fig)
        plt.close()

        if not save_file_path is None \
                and not save_file_name is None:
            # Save input image
            save_image(input_image, save_file_path, 'input', save_file_name, 'png')
            save_image(fit_image, save_file_path, 'output', save_file_name, 'png')

        return left_fit, right_fit, fit_image
    else:
        return left_fit, right_fit, input_image


def process_image_files(input_image_path,
                        output_image_path):
    global prev_to_transform
    global prev_from_transform

    for input_file_name in os.listdir(input_image_path):
        print('Processing: {}'.format(input_file_name))
        base_file_name = input_file_name.replace('.jpg', '')

        # Read & re-write (for reference)
        input_image = mpimg.imread('{}/{}'.format(input_image_path, input_file_name))

        # Filter
        combined_bin_image, transformed_image, undistorted_image, prev_to_transform, prev_from_transform = \
            filter_street_image(input_image,
                                distortion_mtx,
                                distortion_coeff,
                                to_transform=prev_to_transform,
                                from_transform=prev_from_transform,
                                save_file_path='{}/filter'.format(output_image_path),
                                save_file_name=base_file_name)

        # Fit
        fit_lane_lines(combined_bin_image,
                       save_file_path='{}/fit'.format(output_image_path),
                       save_file_name=base_file_name)


def process_video_frame(input_image):
    global video_frame_fit_interval
    global prev_to_transform
    global prev_from_transform
    global prev_left_fit
    global prev_right_fit
    global curr_video_frame_ctr
    global curr_video_base_file_name
    global curr_video_output_path
    global left_line_mean
    global right_line_mean

    output_image_path = None
    base_file_name = None

    if video_frame_save_interval > 0 \
            and curr_video_frame_ctr % video_frame_save_interval == 0:
        output_image_path = curr_video_output_path
        base_file_name = '{}_{:0>6d}'.format(curr_video_base_file_name, curr_video_frame_ctr)

    combined_bin_image, transformed_image, undistorted_image, prev_to_transform, prev_from_transform = \
        filter_street_image(input_image,
                            distortion_mtx,
                            distortion_coeff,
                            to_transform=prev_to_transform,
                            from_transform=prev_from_transform,
                            save_file_path='{}/filter'.format(output_image_path),
                            save_file_name=base_file_name)

    if prev_left_fit is None or prev_right_fit is None \
            or curr_video_frame_ctr % video_frame_fit_interval == 0:
        prev_left_fit, prev_right_fit, fit_image = \
            fit_lane_lines(combined_bin_image,
                           save_file_path='{}/fit'.format(output_image_path),
                           save_file_name=base_file_name)
    else:
        prev_left_fit, prev_right_fit, fit_image = \
            refit_lane_lines(combined_bin_image,
                             prev_left_fit,
                             prev_right_fit,
                             save_file_path='{}/fit'.format(output_image_path),
                             save_file_name=base_file_name)

    # Overplot
    left_fitx_line, right_fitx_line, ploty_axis = \
        get_lr_curve_pixels(prev_left_fit, prev_right_fit, transformed_image)

    left_fitx_line = left_line_mean.update_mean(left_fitx_line)
    right_fitx_line = right_line_mean.update_mean(right_fitx_line)

    # Create an image to draw the lines on
    color_warp_image = np.zeros_like(undistorted_image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx_line, ploty_axis]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx_line, ploty_axis])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp_image, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped_image = cv2.warpPerspective(color_warp_image, prev_from_transform,
                                         (transformed_image.shape[1], transformed_image.shape[0]))

    # Combine the result with the original image
    output_image = cv2.addWeighted(undistorted_image, 1, unwarped_image, 0.3, 0)

    center_diff = get_center_diff_in_m(input_image, prev_left_fit, prev_right_fit)
    cv2.putText(output_image, 'Position: {: >4.2f}m ({})'.format(abs(center_diff * x_meters_per_px),
                                                                 ('left' if center_diff < 0.0 else 'right')),
                (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    left_radius_in_m = get_curve_radius_in_m(left_fitx_line, ploty_axis,
                                             (y_meters_per_px, x_meters_per_px))
    right_radius_in_m = get_curve_radius_in_m(right_fitx_line, ploty_axis,
                                              (y_meters_per_px, x_meters_per_px))
    cv2.putText(output_image,
                'Curvature: {: >4.2f}m (left), {: >4.2f}m (right)'.format(left_radius_in_m, right_radius_in_m),
                (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    if not output_image_path is None \
            and not base_file_name is None:
        save_file_path = '{}/output'.format(output_image_path)
        save_file_name = base_file_name

        # Save input image
        save_image(input_image, save_file_path, 'input', save_file_name, 'png')
        save_image(output_image, save_file_path, 'output', save_file_name, 'png')

    curr_video_frame_ctr = curr_video_frame_ctr + 1
    return output_image


def process_video_files(input_video_path,
                        output_video_path):
    if not os.path.isdir(output_video_path):
        os.makedirs(output_video_path)

    global prev_to_transform
    global prev_from_transform
    global prev_left_fit
    global prev_right_fit
    global curr_video_frame_ctr
    global curr_video_base_file_name
    global curr_video_output_path
    global left_line_mean
    global right_line_mean

    for input_file_name in os.listdir(input_video_path):
        print('Processing: {}'.format(input_file_name))
        base_file_name = input_file_name.replace('.mp4', '')

        prev_to_transform = None
        prev_from_transform = None
        prev_left_fit = None
        prev_right_fit = None
        curr_video_frame_ctr = 0
        curr_video_base_file_name = base_file_name
        curr_video_output_path = output_video_path
        left_line_mean = RunningMean()
        right_line_mean = RunningMean()

        input_video_clip = VideoFileClip('{}/{}'.format(input_video_path, input_file_name))
        fit_video_clip = input_video_clip.fl_image(process_video_frame)
        fit_video_clip.write_videofile('{}/{}.mp4'.format(output_video_path, base_file_name), audio=False)


def execute_process(calibrate_camera=True,
                    test_images=True,
                    test_videos=True,
                    challenge_videos=True):
    setup()

    if calibrate_camera:
        print('Calibrating camera...')
        process_camera_calibration(calibration_image_path, calibration_image_output_path)
        print('...Camera calibrated')

    if test_images:
        print('Processing test images...')
        process_image_files(test_image_path, test_image_output_path)
        print('...Test images processed')

    if test_videos:
        print('Processing test videos...')
        process_video_files(test_video_path, test_video_output_path)
        print('...Test videos processed')

    if challenge_videos:
        print('Processing challenge videos...')
        process_video_files(challenge_video_path, challenge_video_output_path)
        print('...Challenge videos processed')


execute_process()
print('Done!')
