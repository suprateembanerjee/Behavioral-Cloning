# **Behavioral Cloning** 

## Based on NVIDIA End-to-End Learning model [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The project uses the simulator developed by Udacity as present in the repository: https://github.com/udacity/self-driving-car-sim
The dataset for training has also been generated using the same.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/arch.PNG "Model Visualization"
[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/flipped_1.png "Straight Image"
[image4]: ./examples/flipped.png "Flipped Image"
[image5]: ./examples/loss.png "Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.py for creating a video of the output data (using moviepy)
* video.mp4 containing the output video itself

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with a combination of 3x3 filters and 5x5 filters and depths between 24 and 64 (model.py lines 79-83). This is identical to the model proposed by NVIDIA. Some of the layers use striding while others don't.

The model includes RELU layers to introduce nonlinearity (code lines 79-83), and the data is normalized in the model using a Keras LayerNormalization layer (code line 78). Cropping has also been incorporated into the network itself, to ensure a better model by eliminating areas of non-importance. My research suggests the validation loss increased when cropping is also used towards the bottom of the image, and so it has been skipped.

#### 2. Attempts to optimally fit the model

The model generalizes well by default, as a result of strided convolutions and well augmented data, and does not need additional measures to reduce overfitting. Augmentation steps included flipping the training data, as well as incorporating images from all three cameras. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Increasing the speed made poorer models wobbly, but eventually I was able to stabilize the model even at maximum speed available in the simulator.

#### 3. Model parameter tuning

Checkpointing using validation loss(model.py line 90) and Early Stopping with a patience of 10 (model.py line 91) has been implemented to ensure we save the best model that has been trained.

The model used an Adam optimizer, optimized using Exponential Decayed learning rate (model.py lines 93-94) to ensure better convergence.

Huber Loss has been incorporated (model.py line 95) because it performed better than other experimented loss functions for Regression Models such as MSE.

20% of the training data has been used for validation, and shuffling of data has been incorporated inside the fit() call (model.py line 99).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road using a correction factor of 0.2. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design a convolutional model that could work effectively for my use case.

My first step was to use a convolution neural network model similar to the NVIDIA End-To-End Model for Self Driving Cars. I thought this model might be appropriate because it has already been optimized for self driving cars and fit my use case accurately. I initially tried training using much larger models such as Inception_V3 (22M parameters) and NASNetLarge (85M parameters). However, they failed to produce satisfactory results when used in my application.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low loss on the training set and a low loss on the validation set. This implied that the model was underfitting, and not training.

Throughout the process of debugging the program, I discovered much about regression models: that classification-specific metrics such as Accuracy don't work for Regression, and final activation functions need to be changed from softmax and ReLU to linear. I having little experience with regression models before found these observations profound, and getting these concepts cleared helped me optimize the model.

To combat the underfit model, I eliminated all the dropout layers I had initially designed for it.

Then I optimized the initial learning rate for the Exponential Decay, to make it suit my model much better.

The final step was to run the simulator to see how well the car was driving around track one. The initial underfit models produced a constant steering angle of -25 degrees, before I had changed the final activation function from ReLU to Linear. Afterwards, tuning more parameters such as initial learning rate helped my model get more robust.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road at the maximum speed possible.

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-88) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Cropping2D     	| eliminates irrelevant data such as sky and environment 	|
| Lambda 					|	resizes the input to fit the model's requirement											|
| LayerNormalization	      	| axis = 1  				|
| Convolution 5x5     	| 2x2 stride, activation = 'relu', filters = 24	|
| Convolution 5x5     	| 2x2 stride, activation = 'relu', filters = 36	|
| Convolution 5x5     	| 2x2 stride, activation = 'relu', filters = 48	|
| Convolution 3x3     	| 1x1 stride, activation = 'relu', filters = 64	|
| Convolution 3x3     	| 1x1 stride, activation = 'relu', filters = 64	|
| Flatten     	|  	|
| Fully Connected      	| 100,	activation = 'relu'		|
| Fully Connected      	| 50,	activation = 'relu'		|
| Fully Connected      	| 10,	activation = 'relu'		|
| Fully Connected      	| 1,	activation = 'linear'		|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it ever finds itself not positioned in the center. I made selective recordings of only when the vehicle moves towards the center and not when the vehicle moves away from the center. This ensured the vehicle never was trained to move away from the center. I also generated one batch of training data by driving the vehicle in reverse in the same circuit, thereby negating the bias towards left that was induced by the circuit, which had more left turns than right turns.

I did not repeat this process on track two, since it was inversely affecting my model. Since my model does not exactly account for varying heights, the difference in elevation confused the model during training and I found it increased the Huber Loss by approximately 0.003 points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

After the collection process, I had  data points. I had included preprocessing steps in my Netowrk Layers, such as Cropping and Normalization, so I did not implement any such measures externally.

I observed that shuffling the data externally decreased performance as compared to enabling shuffle inside model.fit() (model.py line 99), so I implemented it there itself. I have also implemented a validation split of 20% inside model.fit().

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 on average as evidenced by the Early Stopping Algorithm. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

The Loss variations over epochs is visible through the following visualization:
![alt text][image5]
