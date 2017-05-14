## **Vehicle Detection Project**

The goals and steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to a HOG feature vector.
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use a trained Linear SVM classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./images/randomvehicles.png
[image1]: ./images/randomnonvehicles.png
[image2]: ./images/hogfeatures.png
[image3]: ./images/slidingwindows.png
[image4]: ./images/bboxes.png
[image5]: ./images/heatboxes.png
[image6]: ./images/detection.png
[image7]: ./images/vehicledetector.gif

### Overview

The objective of this project is to identify and track vehicles using computer vision and machine learning techniques from road images taken from the `vehicles` and `non-vehicles` data sets. I used such the histogram of oriented gradients (HOG), color histograms, and spatial binning techniques to extract features from the data. I then used linear support vector machines (SVM) classifier and a sliding window technique and thresholded heat maps to extract windows in the image to identify if the image window represented a vehicle or not. Furthermore, I removed duplicate detections and false positives at the end of the image processing pipeline and displayed bounding boxes on video frames.


### Loading in the data

First, I loaded in the data from the `vehicles` and `non-vehicles` labeled data sets from the [GTI Vehicle Image Database](http://www.gti.ssr.upm.es/data/Vehicle_database.html). The data is later further divided into training, testing, and validation sets. Here are some random sample vehicle and non-vehicle images.

![alt text][image0]
![alt text][image1]

### Color Histogram

The `color_hist` function takes an image and computes the color histogram of features given a particular number of bins and pixels intensity range, and returns the concatenated feature vector. For number of bins I used `nbins=32`.

### Spatial Binning of Color

The `bin_spatial` function takes in an image, a color space conversion, and the resolution you would like to convert it to, and returns a feature vector. Essentially spatial binning scales down an image to a lower resolution while preserving the relevant features. I selected a scaled down size of `size=(32, 32)`.

### Histogram of Oriented Gradients (HOG)

As Dalal and Triggs describes in their paper, the image window is divided into small spatial regions ('cells') which accumulate a local 1D histogram of graident directions or edge orientations over the pixels of the cell. The combined histogram entries form the representation. The local responses are contrast-normalized for better invariance to illumination and shadowing. Contrast-normalization is completing by accumulating a measure of local histogram 'energy' over larger spatial regions ('blocks') and using the results to normalize all of the cells in the block. The normalized descriptor blocks are referred to as the histogram of oriented gradient descriptors.

I used the scikit-image package built in `hog` function to extract Histogram of Oriented Gradient features. The scikit-image `hog` function takes in a single color channel or grayscaled image as input, as well as various parameters. These parameters include `orientations`, `pixels_per_cell` and `cells_per_block`.

The number of orientations is specified as an integer, and represents the number of orientation bins that the gradient information will be split up into in the histogram. Typical values are between 6 and 12 bins.

The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. This parameter is passed as a 2-tuple and cells are commonly chosen to be square.

The cells_per_block parameter is also passed as a 2-tuple, and specifies the local area over which the histogram counts in a given cell will be normalized. Block normalization  generally leads to a more robust feature set.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is a sample `vehicle` and `non-vehicle` images using the `YCrCb` color space and HOG parameters of `cmap='gray'`, `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` which are also the final parameter values that I used:

![alt text][image2]

### Sliding Window Search

First, I convolved a small window to extract windows from the test image. The window width, window height, and window overlap are parameters for the sliding window algorithm. I removed the bottom band showing the car's hood and the top half showing the sky from the search space. I tried several window sizes, overlaps, and search areas and ultimately arrived at the following parameter values.  
* `xy_window = (96, 96)`
* `xy_overlap = (0.75, 0.75)`
* `y_start_stop = [400, 600]`
* `xy_window = [None, None]`

The following image depicts all of the search window positions used for vehicle detection.  

![alt text][image3]

### Linear SVM (Support Vector Machines) Classifier

Before feeding in the data to the classifer, I had to prepare the data. To avoid having mny alogrithm simply lassify everything as belonging to the majority class, I made sure that there was an roughly equivalent amount of `vehicle` and `non-vehicle` images in the data sets. To avoid problems due to the ordering of the data, I randomly shuffled the data. To avoid overfitting and to improve model generalization, I split the data into training, testing, and validation sets using `train_test_split` from `sklearn.model_selection`. To prevent individual features or sets of features from dominating the response of my classifier I normalized features to zero mean and unit variance.

I selected Linear SVM because they are relatively fast to train while still retaining very high accuracy for this data set. Linear SVM classifiers can be tuned by varying hyper-parameters C: `[0.8, 0.9, 1.0, 1.1, 1.2]`, penalty: `l2`, and loss function `hinge` or `squared hinge` and optimizing for the highest accuracy.
These are the best parameters I found for my Linear SVM classifier which resulted in a extremely high `0.9901` validation set accuracy:
* `C = 0.08`
* `penalty = 'l2'`
* `loss = 'hinge'`

### Multiple Detections and False Positives

The following image is a test image with overlapping detections. The goal of this section is to build heat map from these detections in order to combine overlapping detections and remove false positives. First, I initialize a heat map image copy with equal dimensions. The I add "heat" (+=1) (`add_heat`) for all pixels within windows where a positive detection is reported by my classifier. Then I impose a threshold (`apply_threshold`) at the "hot" parts of the heat map to reject false positives.

![alt text][image4]

In the pipeline, I integrate a heat map over several frames of video to combine overlapping detections so that areas of multiple detections get "hot" (where the cars are), while transient false positives stay "cool". Then, I can simply threshold the heat map to reject false positives. The following image is the thresholded heat map showing 2 cars.

![alt text][image5]

In addition to `add_heat` and `apply_threshold`, I also use a custom `HeatMapQueue` class to store the last 25 heat maps to smooth out the predicted bounding boxes. The sum of the last 25 heat maps are passed to `apply_threshold`. I also create a custom `VehicleDetector` class that bundles together all of the methods. This class accepts road images and outputs annotated images annotated with bounding boxes.

The following image shows the resulting bounding boxes formed from the thresholded heat maps on 6 test images.

![alt text][image6]

---

### Discussion

This project was a great introduction in vehicle detection and tracking from front-facing camera images. I used color histograms, histograms of oriented gradients (HOG), and spatial binning to extract features from road images. Next, I applied a sliding windows technique to select out windows to consider for vehicle detection. Last, I applied a thresholded heat map to remove duplicate detections and false positives.

The following is a short snippet of the output video after applying my image processing pipeline to `project_video.mp4`.

![alt text][image7]

In the full video, there are many false positives. I expect that I could reduce the false positive rate by using more powerful or non-linear classifiers such as SVMs with sigmoid, rbf, or polynomial kernels, random forests, gradient boosting, or even convolutional neural networks.

I wonder how a convolutional neural network would perform for vehicle detection. I expect that it would result in higher accuracies and fewer false positives, but require much longer training times.

I read in the CarND Slack channel that deep learning methods like [YOLO](https://pjreddie.com/darknet/yolo/) and [SSD](https://arxiv.org/abs/1512.02325) have performed very well for car and traffic sign detection and tracking. I would like to personally try out these methods and combine my image pipelines from traffic sign detection, advanced lane line detection, and vehicle detection.

### References

N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 886-893 vol. 1.
[http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
