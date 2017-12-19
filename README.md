# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
    * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
The Project
---

All of the code for the project is contained in the Jupyter notebook `Vehicle_Detection.ipynb` 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The hog features are extracted from 'get_hog_features' function defined in the lesson_functions.py

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
In the section "Train a classifier". I have collected all the training data and classified in to cars and noncars. Using sklearn StandardScaler I normalized each set of data for mean 0 and variance. Then I extracted the features using hog, gradient and histogram. I have split my training set to 80/20. I have used Linear SVM to classify. I was able to get 98.7 accuracy. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled "Method for Using Classifier to Detect Cars in an Image" I adapted the method `find_cars` from the lesson materials. The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction. This is fast compared to calculating hog each time.

I have struggled to get the scale and the ystart and yend parameters to work correctly.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The test images are shown in the notebook. I have tried RGB ang got 97%. Changing it to HLS gave 98.7% accuracy. Normalizing the data increased the accuracy. pixel per cell is 8. Its very slow. Would be interesting to increase the patch ans see the accuracy

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I have used the same pipeline; the only difference being previous images are saved and added to the heat map to better detect the vehicle

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Even the accuracy of the SVM is 98.7, The classifier didn't perform very well on the images. The parameters scale, pixel patch are very hard to tune. Getting multiple ROI with scale  and combing it in the pipeline is very slow.




