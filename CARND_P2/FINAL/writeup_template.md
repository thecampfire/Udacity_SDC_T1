# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/1.png "Traffic Sign 1"
[image2]: ./writeup_images/2.png "Traffic Sign 2"
[image3]: ./writeup_images/3.png "Traffic Sign 3"
[image4]: ./writeup_images/4.png "Traffic Sign 4"
[image5]: ./writeup_images/5.png "Traffic Sign 5"
[image6]: ./writeup_images/images_per_class.png "Class distribution"
[image7]: ./writeup_images/before_norm.png "Before Normalization"
[image8]: ./writeup_images/after_norm.png "After Normalization"
[image9]: ./writeup_images/geometric.png "Geometric Transformations"
[image10]: ./writeup_images/filled.png "Filled up classes"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many examples there are in each class:

![alt text][image6]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The project description suggested, at a minimum, normalizing the images, which I did. I seemed to get better results by using [0, 1] as the normalization interval instead of [-1, 1]

I then used histogram normalization to enhance the image (levels, contrast, etc.)

Here are some images before normalization, and after.

![alt text][image7]
![alt text][image8]


I tried using the initial data set to reach over 93% accuracy, but it looked like there was no chance of doing that without either a more complicated architecture, or more data.

At first, I decided to enhance the data in a few ways:
* doing random, geometric transformations
* making sure each class has at least a few thousand data points. 

### Geometric Transformations

I used random rotation, scaling and warping, on each image.

Here is an example of an original image and an augmented image:

![alt text][image9]

I found an interesting idea on the Web around flipping signs in order to generate additional data points by using the original data. I took the code and provided attribution to the author. 

### Filling up each class

I chose 3,000 as the minimum number of data points for each class. After flipping images, for classes that still had under 3,000 data points, I randomly used a geometric transformation, more or less the same number of times, on each original image within the data class. 

The difference between the original data set and the augmented data set is the following ... 

![alt text][image10]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with LeNet and implemented the concatenation idea in the LeCun paper provided. This yielded good results. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400    |
| RELU					|												|
|Fully connected   		| Concatenated Output: layer 2 + layer 3 convolutions, 1x1x800												|
|Dropout  						|0.5 probability						|
|Logits                 | 43 classes                                    |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I started with some of the basic LeNet hyperparameters, such as a Learning Rate of 0.001. I increased the batch size to 256 as the model used was not so complex that it required a lot of memory.

I trained the model for 50 epochs. The model seems to deliver a validation set accuracy of over 93% within the first few epochs. I tried running 35 epochs and 50 epochs, and it looked like there was a difference of about 1% accuracy between the two, so I went ahead with the 50 epoch model. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 96.1%
* test set accuracy of 94.5%

I initially tried implementing an Inception model based on GoogleNet. That model took a very long time to train on the initial data set, and it did not deliver great results.

I then started generating more data and it looked like the Inception model would have taken too long to run. 

I instead tried LeNet again and after seeing that it performed quite well on the augmented data set, tried implementing the concatenation technique demonstrated in the paper suggested.  
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians      		| Pedestrians  								100%	| 
| No Passing     			| No Passing 								100%		|
| Turn Right Ahead					| Turn Right Ahead					72%						|
| 30 km/h	      		| 30 km/h					 			100%	|
| Beware of Snow			| Children Crossing      					53%		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. It looks like it mistook the last sign for a Pedestrians sign, but it's not overly certain. Feeding it a more reasonable number of signs, maybe 100, would tell us more about what the true accuracy of the model is. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![Pedestrians][image2]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Pedestrians   								| 
| 0.0     				| Road Work 									
| 0.0					| General Caution											|
| 0.0	      			| Right of Way					 				|
| 0.0				    | Traffic Signals      							|


For the second image the results are quite similar:

![No Passing][image3]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| No Passing   								| 
| 0.0     				| All the rest 									


For the third image:

![Turn Right][image4]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.72       			| Turn Right Ahead   								| 
| 0.28    				| Ahead Only 									
| 0.0					| All the rest									|


For the fourth image:

![30 km/h][image1]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| 30 km/h   								| 
| 0.0     				| All the rest 									


The prediction for the fifth image was wrong but not by a lot:

![Beware of Ice / Snow][image5]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.53       			| Children Crossing   							| 
| 0.38     				| Beware of Ice / Snow                			|
| 0.09					| Slippery Road									|
| 0.0	      			| Right of Way					 				|
| 0.0				    | Double Curve      							|