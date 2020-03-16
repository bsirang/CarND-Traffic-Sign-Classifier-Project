# **Traffic Sign Recognition**

---

## Project Summary
The goal of this project was to create a convolutional neural network (CNN) in
Tensorflow, which is based on the LeNet architecture, to correctly classify
images of German traffic signs.

The goals can be broken down as follows:
* Load the training / validation / test datasets
* Explore and visualize the datasets
* Train and validate, and test a CNN on the given data
* Test the model on new data found online
* Analyze the results of the model on the data found online


[//]: # (Image References)

[image1]: ./examples/histograms.png "Histograms"
[image2]: ./examples/training_visualization.png "Training Visualization"
[image3]: ./examples/online_signs.png "Traffic Signs Found Online"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This writeup will make references to the "notebook", which is referring to the Jupyter Notebook titled: `Traffic_Sign_Classifier.ipynb`

### Data Set Summary & Exploration
Here are some statistics on the input dataset. More details can be found in the notebook.
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization via Histograms

The training and validation datasets were plotted as histograms of the traffic sign classes. This helps give a rough sense of the distribution of different traffic sign types. A quick observation shows that some classes had less than 250 occurrences in the test data, while others were closer to 2000.

You can observe that the shape of the two histograms roughly match each other. This is a good indication that the validation data set is a good representation of the training set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing

##### Grayscale
When designing the model architecture, I designed it such that the input depth can be variable. I did this so that I can try both RGB (depth=3) and grayscale (depth=1) to see which performed better. In the end, grayscale seemed to perform better, so I went ahead and did a grayscale conversion as part of the preprocessing.

Here is an example of four traffic sign images (chosen at random) before and after grayscaling. I implemented grayscaling by simply averaging the values along the color dimension. This was done by specifying the axis to average using numpy's average function: `np.average(x, axis=(3,))`.

![alt text][image2]

##### Normalization
After grayscale conversion, I normalized the data. Each pixel is one byte, so they range from 0-255. Instead, we want the values to range from [-1 1], so that they are centered around zero (and have approximately equal variance). This can be done simply by shifting down by 128 and then dividing by 128 (i.e. `(x - 128.0) / 128.0`)


#### 2. Model Architecture

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale / normalized image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, output depth of 16 (outputs 28x28x16) 	|
| RELU					|	Activation	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output depth of 64 (outputs 10x10x64) |
| RELU					|	Activation	|
| Fully connected		| outputs 480	|
| Fully connected		| outputs 280	|
| Fully connected		| outputs 43 (43 classifications, one-hot encoded)	|


#### 3. Training Pipeline

The training pipeline can be described as follows:

CNN output logits -> Softmax cross entropy -> Reduce mean function -> Adam optimizer

1. CNN outputs 43 logits, one for each one-hot encoded classifier type
1. A softmax is performed, and then cross entropy is calculated
1. The optimization function is defined as minimization of mean of cross entropy
1. Finally, this optimization function is passed to the Adam optimizer, which is a better performing derivative of stochastic gradient descent.

#### 4. Approach
An iterative approach was taken, using the LeNet architecture as a starting point. This approach was chosen to build up intuition about how the various hyper-parameters and network structures affect the results.

##### Starting Point
Initially, the LeNet architecture from the MNIST digits classification lab was used with the minimal changes required to operate output 43 classes and take inputs with a depth of 3 (for RGB color). This model reached just under 90% accuracy on the validation sets. The learning rate was tweaked to see how it would effect the results. It was determined that cutting the learning rate in half from `0.001` to `0.0005` improved performance, so the learning rate was kept here.

Next, the images were converted to grayscale and the architecture was adapted to take in an input depth of 1. The idea was to find out empirically whether or not color data helps or hurts the performance of the network. The accuracy was observed to be slightly higher so grayscale conversion was added to the preprocessing stage.

##### Modified Architecture
Next, the LeNet architecture was modified in an attempt to increase accuracy. The output depths of the first two convolutional layers were increased to 16 and 64 respectively. The intuition here is that unlike digits, there's a lot more potential features to extract, and a lot more output classes (43 vs. 10). The second layer's output depth is considerably larger than the first layer. This was done to mimic the LeNet architecture. The performance of the network was observed to go up.

##### Activation
Both `RELU` and `Sigmoid` activation functions were tested. Training with `Sigmoid` would result in slower training times without any noticeable performance increase. For this reason, `RELU` was chosen.

##### Max Pooling
The second max pooling layer was removed entirely. The intuition was that because the pooling layer is primarily to save on computation not increase performance, it could be removed because the training time without it is still relatively low on modern GPU hardware.

##### Dropouts
Finally, to avoid overfitting on the training data, dropouts were added for the 3rd and 4th fully connected layers. After some research, it was decided that it's better to apply dropout layers to the fully connected layers rather than the convolution layers. Intuitively this made sense to me because of the weight-sharing aspect of convolutional layers. Special care was taken to ensure that the dropouts only happen during training and not during validation and testing since dropouts are only valuable during training.

Initially, keep probabilities of `0.5` were chosen for both layers. After observing that the training accuracy would hit `1.0` while the validation accuracy was still below `0.95`, the probabilities were reduced until the training and validation accuracy were tracking fairly well.

##### Epochs
The number of epochs were chosen such that the training terminated after the validation accuracy leveled off. This was done to mitigate overfitting.


My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.950
* test set accuracy of 0.941

### Test a Model on New Images

#### 1. German Traffic Signs Found Online

Here are five German traffic signs that I found on the web (and cropped/scaled to 32x32):

![alt text][image3]

By visual inspection, I anticipated the second image (ice/snow ahead) to be troublesome due to the angled perspective and the white-padding added to maintain the original aspect ratio. The rest of the images seemed reasonable for the model to predict accurately.

#### 2. Results

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Double Curve      		| Slippery Road   									|
| Ice/Snow	| Speed Limit 60 km/h	|
| Bicycle Crossing | General Caution |
| Children Crossing	| Double Curve |
| Yield | Yield	|


The model was only able to correctly guess 1 out of 5 (20%) of these images. Some time was spent troubleshooting to determine if there was some kind of mistake in pre-processing these images, which caused their inputs to come through differently than the training set.

At the time of writing it's not yet determined what is the cause of the performance difference between these signs and the original input data sets.

#### 3. Prediction Certainty

The probability for the yield sign was 1, which was the only sign that was correctly predicted. Despite not predicting the other signs, the confidence values were low enough such that the prediction can be tossed out. The top 5 predictions are all listed at the bottom of the Jupyter notebook.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .032        			| Double Curve	|
| .157    				| Ice/Snow	|
| .332					| Bicycle Crossing	|
| .362	      			| Children Crossing	|
| 1.00				    | Yield	|
