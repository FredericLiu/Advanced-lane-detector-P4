**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the source code file 'CameraCali.py'.  

The source code defined three functions: cameraCali(), cal_undistort() and testCali().

cameraCali() firstly used glob to load all chessboard images (except one for test), then find all corners in real world and in 2D plane image, separately as imgpoints and objpoints.

cal_undistort() used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  Then saved mtx and dist into a file called 'wide_dist_pickle.p'. For the rest of this project, I load the pickle file once, then used cv2.undistort() for each frame in the algrithom.

caliTest() use one test image to see if the calibration algrithom works properly. The test result is shown as following:

![caliTest](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/camera_cal/test_result.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![caliTest](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/camera_cal/test_undist.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 33-42 in `pipelineClass.py`, and the thresholding functions are defined at lines 108-171 in 'pipelineClass.py'). 

I used threholdings on sobelx, sobely, magnitude, direction and color, all combined these five images to detect the lanes.

I will demonstrate the thresholded test images later in section 6.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in lines 301-320 in `pipelineClass.py`
 
corners_unwarp() has two features:

1. undistort the input image using pickled mtx and dist coeffects, via cv2.undistort.

2. transform the image using cv2.warpPerspective, from source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[210,720],[1100,720],[595,450],[690,450]])
dst = np.float32([[200,720], [1000,720], [200,0], [1000,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 720      | 200, 720      | 
| 1100, 720     | 1000, 720     |
| 595, 450      | 200, 0        |
| 690, 450      | 1000, 0       |

I verified that my perspective transform was working as expected by drawing the warped top-down image and verify that the lines appear parallel in the warped image.
I will demonstrate the thresholding and transforming result of the test images later in section 6.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I defined three functions to identify lane pixels and fit polynomial, FindingBsae(), firstSliding() and skipSliding() at lines 173-278 in 'pipelineClass.py'

FindingBase() used histogram to find the two base point of two lanes.

firstSliding() used sliding window to find the lane pixels, based on the base point from FindingBase(), finally fit the pixels with a ploynomial

skipSliding() is used for the case that you has find a valid polynomial in last frame, then for current frame, the algorithm will use skipSliding() to find the polynomial for 

current frame. the core thinking is find pixels around the polynomial fitted in last frame, as the new lane pixels for current frame.

And in the pipeline, I designed the strategy including smoothing, reset and sanity check, shown as bellow code:

```python
if(self.reset == True):
	leftx_base, rightx_base = self.FindingBase(binary_warped)
	leftx, lefty, rightx, righty,left_fit, right_fit, out_img = self.firstSliding(binary_warped, leftx_base, rightx_base, margin = 100, minpix = 50, nwindows = 9, window_height= 80)
	self.recent_leftfit.append(left_fit)
	self.recent_rightfit.append(right_fit)            
	self.smoothing()            
elif(self.reset == False):
	leftx, lefty, rightx, righty,left_fit, right_fit, out_img= self.skipSliding(binary_warped, margin=100, left_fit=self.best_leftfit, right_fit=self.best_rightfit)           
	sanityCheck = self.sanityCheck(left_fit,right_fit)
	if (sanityCheck == True):
		self.recent_leftfit.append(left_fit)
		self.recent_rightfit.append(right_fit)            
		self.smoothing()
	else:
		reset_counter += 1 
		if (reset_counter == 3):
			self.reset = True 
			reset_counter = 0
```
The smoothing(). sanityCheck() are defined at lines 77-106 in pipelineClass.py

I will demonstrate the results that using this pipeline over the test images later in section 6.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 280-299 in `pipelineClass.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 357-378 in my code in `pipelineClass.py` in the function `plotTestImage()`. 

I will demonstrate the results that using this pipeline over the test images as below:

![analysis_test1](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/output_images/analysis_test1.png)
![analysis_test2](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/output_images/analysis_test2.png)
![analysis_test3](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/output_images/analysis_test3.png)
![analysis_test4](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/output_images/analysis_test4.png)
![analysis_test5](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/output_images/analysis_test5.png)
![analysis_test6](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/output_images/analysis_test6.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/FredericLiu/Advanced-lane-detector-P4/blob/master/output_images/lanes_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have discuss some of my pipeline and algorithm above in the writeup.
Besides that, there are also some problem need to be settled in the future:

a. I caculated the radius of curvature but didn't use them to check the sanity, because sometimes the curvature are quite different, especially when on the straight lane.

b. The current algorithm performed quite good on project video, but for the challenge vedios, it's result is very bad. I assume this is because the threholding can't find the correct lanes, but I tried to tune the thresholds for a few days, still can't get the acceptible result. In the future I need to find some reference about this topid to learn.
