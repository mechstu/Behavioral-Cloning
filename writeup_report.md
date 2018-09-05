# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I employed the NVDIA model that is discussed in the course lecture. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was trained initially on 5 epochs, but this loss did not reduce much on the last 2 epochs, so the training was reduced to 3 epochs to avoid over fitting. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving clockwise on the track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVDIA network. It is a proven network so I decided to go with it.

I experimented with the data collected and employed various subsampling techniques to get the optimal data set to train my model. Details of this procedure are explained below.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I reduced the number of epochs from 5 to 3 while training the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, initially, at the left turn right after the bridge. To improve the driving behavior in these cases, I sub-sampled the data so that the distribution of steering angles is closer to normal. This fixed the left turn issue, but the car was not turning right. I had to go back and add some more right turn data since the car was not turning right. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The model has 5 convolution layers:
Layer 1: 24 filters, kernal size: 5x5, strides 2, Activation RELU
Layer 2: 36 filters, kernal size: 5x5, strides 2, Activation RELU
Layer 3: 48 filters, kernal size: 5x5, strides 2, Activation RELU
Layer 4: 64 filters, kernal size: 3x3, strides 1, Activation RELU
Layer 5: 64 filters, kernal size: 3x3, strides 1, Activation RELU

This is followed by a flattening layer and then 4 fully connected layers. These layers have 100, 50, 10 and 1 outputs respectively. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Before normalizing, the images are cropped from 160 input rows to 80 output rows. This removes unwanted parts of the image, like the sky and the hood of the car. 


#### 3. Creation of the Training Set & Training Process

I initially drove 6 laps, 3 anti-clockwise and 3 clockwise. I added some recovery from the side of the road data as well. I trained the model using this data set, and the vehicle did not do well in the curves. Looking at the histogram of the shows that the; data had a lot of samples of straight line driving that was skewing the model to be trained in straight line driving. 

<img src="./examples/All_Driving_dist.JPG?raw=true">

To remedy this, the data was sub-sampled so that the distribution is near normal for the center steering angles. This would train the model to drive in curves as well as straight lines equally well. The histogram of the subsampled data is shown below.

<img src="./examples/Normalized_Driving_dist.JPG?raw=true">

Even after this adjustment, the vehicle was not performing in right turns. I added an extra lap in the clockwise direction. This added more right turn data to train the model to work better in right turns. The resulting distribution of the augmented data looks like this.

<img src="./examples/Normalized_Driving_augmented_dist.JPG?raw=true">

After the collection process, I had 8696 data points. I then preprocessed this data by cropping 55 rows from the top and 25 rows from the bottom. Then I normalized the pixels.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
