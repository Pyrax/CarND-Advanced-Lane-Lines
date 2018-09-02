## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_undistorted.jpg "Undistorted calibration image"
[image2]: ./output_images/test1_undistorted.jpg "Undistorted real scenario"
[image3]: ./output_images/colored_thresholds.jpg "Colored thresholds"
[image4]: ./output_images/thresholded_image.jpg "Binary thresholded image"
[image5]: ./output_images/transform_unwarped.jpg "Unwarped image"
[image6]: ./output_images/transform_warped.jpg "Warped image"
[image7]: ./output_images/lane_fitting.jpg "Lane fitting"
[image8]: ./output_images/curvature_formula.jpg "Curvature formula"
[image9]: ./output_images/result.jpg "Result"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Camera Calibration

The code for this step can be found in the IPython notebook located at "[./src/camera_calibration.ipynb](src/camera_calibration.ipynb)".

To calibrate the camera I first determined that the chessboard pattern in the calibration images have 9x6 inside corners.
Then, I prepared the "object points" which will be the (x, y, z) coordinates of the corners in the world. As I assumed
that the chessboard is fixed on the (x, y) plane at z=0, the object points are identical for every calibration image.
Now, I searched for the positions of these corners in the images which are the "image points". These are used to create
a mapping of the real-world object points to the image space for all calibration images where all inside corners
were detected.

From the available calibration images 17 out of 20 were successfully used for the mapping.

Finally, I calibrated the camera and calculated the distortion coefficients through the `cv2.calibrateCamera()` function
and saved them to a serialized file with pickle for later usage.
Undistorting an image leads to this result:

![Undistorted calibration image][image1]

However, the process of undistorting an image from a real scenario also shows that it might lead to a slight (for most 
cases neglectable) information loss on the image as the following image demonstrates:

![Undistorted real scenario][image2]

Here, both rear lights from the white car are visible on the original image but one is cut off in the undistorted image.

### Pipeline (single images)

Steps I used to experiment with different parameters and algorithms to obtain different outputs are available in the
"lane_finding" notebook at "[./src/lane_finding.ipynb](src/lane_finding.ipynb)" while the results of the final pipeline
(which is "[./src/Pipeline.py](src/Pipeline.py)") have been produced by the "pipelined_lane_finding" notebook at 
"[./src/pipelined_lane_finding.ipynb](src/pipelined_lane_finding.ipynb)".

#### 1. Example of a distortion-corrected image.

Camera distortion is applied by first loading the undistortion coefficients from the previously stored pickle file at
"[./src/calibration.pickle](src/calibration.pickle)" and then calling the `cv2.undistort()` function on an image. For 
an example see the real scenario image from the camera calibration section.

#### 2. Creating a thresholded binary image through color transforms and gradients

The next step in the pipeline is to create a binary image through various thresholds. I decided to convert the image
to the HLS color space and to grayscale in order to create color thresholds. From the HLS image the S-channel is used
for thresholding with a range of `(170, 255)` which is combined using bitwise-or with a threshold on the grayscale image 
with limits of `(200, 255)`. The purpose of the grayscale threshold is mainly to detect the white lanes more confidently 
while the S-channel picks up the yellow lanes.
Additionally, I combined the result so far with a gradient in x-orientation which is calculated using the
Sobel-operator. The origin of what pixels are detected by which threshold are shown here:

![Colored thresholds][image3]

It shows that colored thresholds generally pick up near lanes at the bottom of the image while the gradient adds
information to the distance because the color looses density and becomes less distinct there.

The following picture represents the thresholded image in binary format how it is used in the pipeline:

![Binary thresholded image][image4]

It is done in the `preprocess_image()` function of the "[./src/Pipeline.py](pipeline)" in lines 44-55.

#### 3. Perspective transform

Then, I transformed the perspective of the image with a matrix. The matrix is calculated through image source and 
destination points with the `cv2.getPerspectiveTransform()` function. I determined the source points by observing images
with straight road lines.
For the destination points, I chose y-values equal to the top and bottom of the original images and x-values to be at
1/4 and 3/4 of the width because lanes would optimally be centered in the left and right halves.
It results in the following mapping for our `(1280, 720)` shaped images:

| Source             | Destination   | 
|:------------------:|:-------------:|
| 593.15594, 450.0   | 320.0, 0.0    | 
| 690.84406, 450.0   | 320.0, 720.0  |
| 1120.0, 720.0      | 960.0, 720.0  |
| 210.0, 720.0       | 960.0, 0.0    |

Applying it in the examined images confirms that lines are roughly parallel in the warped image:

![Unwarped image][image5] ![Warped image][image6]

In the "[./src/Pipeline.py](pipeline)" it is executed by the functions `warp()`, `unwarp()`, `transform_perspective()`
and the `get_transform_params()` function which calculates the points.

#### 4. Identifying lane-line pixels and fitting their positions with a polynomial

Now, I identified the lane pixels through by calculating a histogram of the binary warped image where the peaks of each 
half of the image is used as a starting position for a sliding window search. Then I average the x-coordinates of the 
detected pixels to fit a second order polynomial through the average. 

![Lane fitting][image7]

I wrote a class called "[./src/LaneDetector.py](LaneDetector)" to perform these operations. In the file 
"[./src/LaneFindingStrategies.py](LaneFindingStrategies.py)" there are the `SlidingWindowStrategy` and 
`PolySearchStrategy` classes which describe the method how the lane pixels are identified from which I already described 
the former. The "PolySearchStrategy" on the other side is used for videos where an initial sliding window search gets 
a first polynomial function and then for subsequent images those polynomial coefficients are used to search around the 
already fitted function for new lane pixels.

#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center

Finally, I calculated the radius of curvature and the position of the vehicle by taking the x-coordinate of the lanes 
at the bottom of the image (y=max). Assuming the camera is mounted at the center of the car, the position of the vehicle 
is simply the difference between the middle and the actual lane position.
The radius of curvature boils down to this formula (again assuming that the camera is mounted at the center of the car): 

![Curvature formula][image8]

Those calculations are performed in the "[./src/Metrics.py](Metrics)" class where I also convert the units from pixels 
to meters.

#### 6. Result

The result of all the previous steps is demonstrated in this image:

![Result][image9]

### Pipeline (video)

Here's a [link to my video result](./output_videos/project_video.mp4).

### Discussion
