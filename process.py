import os

import cv2
import matplotlib.image as mpimg
import numpy as np

calibration_image_path = './camera_cal'
test_image_path = './test_images'

output_dir_name = 'process_output'
calibration_output_dir_name = (output_dir_name + '/calibration')
test_output_dir_name = (output_dir_name + '/test')

default_offset_px = 200
# [dst_ul, dst_ur, dst_ll, dst_lr]
default_source_px_multipliers = [[45.31, 63.61], [55.00, 63.61], [20.86, 91.81], [81.41, 91.81]]


def setup():
    if (not os.path.isdir(output_dir_name)):
        os.mkdir(output_dir_name)


def save_image(input_image, output_path, output_type, output_name, output_ext):
    if (not os.path.isdir(output_path)):
        os.mkdir(output_path)
    if (not os.path.isdir(output_path + '/by_type')):
        os.mkdir(output_path + '/by_type')
    type_path = (output_path + '/by_type/' + output_type)
    if (not os.path.isdir(type_path)):
        os.mkdir(type_path)
    if (not os.path.isdir(output_path + '/by_name')):
        os.mkdir(output_path + '/by_name')
    mpimg.imsave(type_path + '/' + output_name + '_' + output_type + '.' + output_ext, input_image)
    name_path = (output_path + '/by_name/' + output_name)
    if (not os.path.isdir(name_path)):
        os.mkdir(name_path)
    mpimg.imsave(name_path + '/' + output_name + '_' + output_type + '.' + output_ext, input_image)


def calibrate_camera(input_image_path,
                     output_image_path):
    # termination criteria
    sub_pix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    distortion_mtx = None
    distortion_coeff = None
    is_example_saved = False

    # Step through the list and search for chessboard corners
    for input_file_name in os.listdir(input_image_path):
        print('Processing: {}'.format(input_file_name))
        base_file_name = input_file_name.replace('.jpg', '')

        input_image = mpimg.imread(input_image_path + '/' + input_file_name)
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
            perspective_transform = get_perspective_transform(image_size, source_px)
            transformed_image = transform_image(input_image, perspective_transform)

            if not is_example_saved:
                is_example_saved = True
                save_image(input_image, output_image_path, 'input', base_file_name, 'png')
                save_image(undistorted_image, output_image_path, 'undistorted', base_file_name, 'png')
                save_image(transformed_image, output_image_path, 'transformed', base_file_name, 'png')

    return distortion_mtx, distortion_coeff


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
    return cv2.getPerspectiveTransform(np.float32(source_px), dest_px)


def transform_street_image(input_image,
                           source_px_multipliers=default_source_px_multipliers,
                           perspective_transform=None):
    image_size = (input_image.shape[1], input_image.shape[0])

    if perspective_transform is None:
        source_px = []
        for item in default_source_px_multipliers:
            source_px.append([round((item[0] / 100.0) * image_size[0]),
                              round((item[1] / 100.0) * image_size[1])])
        perspective_transform = get_perspective_transform(image_size, source_px, default_offset_px)

    return transform_image(input_image, perspective_transform)


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


def process_test_images(input_image_path,
                        output_image_path,
                        distortion_mtx,
                        distortion_coeff):
    for input_file_name in os.listdir(input_image_path):
        print('Processing: {}'.format(input_file_name))
        base_file_name = input_file_name.replace('.jpg', '')

        # Read & re-write (for reference)
        input_image = mpimg.imread(input_image_path + '/' + input_file_name)
        save_image(input_image, output_image_path, 'input', base_file_name, 'png')

        # Undistort
        undistorted_image = undistort_image(input_image, distortion_mtx, distortion_coeff)
        save_image(undistorted_image, output_image_path, 'undistorted', base_file_name, 'png')

        # Sharpen
        sharpened_image = sharpen_image(undistorted_image)
        save_image(sharpened_image, output_image_path, 'sharpened', base_file_name, 'png')

        # Transform
        transformed_image = transform_street_image(sharpened_image, default_source_px_multipliers)
        save_image(transformed_image, output_image_path, 'transformed', base_file_name, 'png')

        # Choose a Sobel kernel size
        gray_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY)
        save_image(gray_image, output_image_path, 'gray', base_file_name, 'png')

        # Sobel kernel size
        kernel_size = 3

        # Apply each of the thresholding functions
        red_bin_image = rgb_threshold(transformed_image, 'r', thresh=(230, 255))
        save_image(red_bin_image, output_image_path, 'red_binary', base_file_name, 'png')

        grn_bin_image = rgb_threshold(transformed_image, 'g', thresh=(205, 255))
        save_image(grn_bin_image, output_image_path, 'grn_binary', base_file_name, 'png')

        blu_bin_image = rgb_threshold(transformed_image, 'b', thresh=(200, 255))
        save_image(grn_bin_image, output_image_path, 'blu_binary', base_file_name, 'png')

        sat_bin_image = hls_threshold(transformed_image, 's', thresh=(150, 255))
        save_image(sat_bin_image, output_image_path, 'sat_binary', base_file_name, 'png')

        lgt_bin_image = hls_threshold(transformed_image, 's', thresh=(150, 255))
        save_image(lgt_bin_image, output_image_path, 'lgt_binary', base_file_name, 'png')

        hue_bin_image = hls_threshold(transformed_image, 'h', thresh=(20, 40))
        save_image(hue_bin_image, output_image_path, 'hue_binary', base_file_name, 'png')

        x_bin_image = abs_sobel_thresh(gray_image, orient='x', sobel_kernel=kernel_size, thresh=(20, 120))
        save_image(x_bin_image, output_image_path, 'x_binary', base_file_name, 'png')

        y_bin_image = abs_sobel_thresh(gray_image, orient='y', sobel_kernel=kernel_size, thresh=(20, 120))
        save_image(y_bin_image, output_image_path, 'y_binary', base_file_name, 'png')

        mag_bin_image = mag_thresh(gray_image, sobel_kernel=kernel_size, mag_thresh=(20, 200))
        save_image(mag_bin_image, output_image_path, 'mag_binary', base_file_name, 'png')

        dir_bin_image = dir_threshold(gray_image, sobel_kernel=kernel_size, thresh=(0.001, 0.001 + (np.pi / 24.0)))
        save_image(dir_bin_image, output_image_path, 'dir_binary', base_file_name, 'png')

        combined_bin_image = np.zeros_like(dir_bin_image)
        combined_bin_image[
            np.sum([red_bin_image, grn_bin_image, blu_bin_image,
                    sat_bin_image, hue_bin_image, lgt_bin_image,
                    x_bin_image & y_bin_image,
                    mag_bin_image & dir_bin_image], axis=0) > 2] = 1
        save_image(combined_bin_image, output_image_path, 'combined_binary', base_file_name, 'png')


def execute_process():
    setup()

    print('Calibrating camera...')
    distortion_mtx, distortion_coeff = calibrate_camera(calibration_image_path, calibration_output_dir_name)
    print('...Camera calibrated')

    print('Processing test images...')
    process_test_images(test_image_path, test_output_dir_name, distortion_mtx, distortion_coeff)
    print('...Test images processed')


execute_process()
print('Done!')
