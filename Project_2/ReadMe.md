## Project 2:

### Disclaimer:
We initially started on an ML4Science project, in Olaf Blanke's lab. We spent the majority of the project time span on it, and started the road recognition 3 weeks before the deadline. You will find in this repository a report for the first part, explaining what we did. We believe that we managed to finish the road recognition challenge properly, but it was important to us to include some of the works we did in the first part of the project. 

### Description:
We introduce our take on a **segmentation task**, using a **convolutional neural nets** for a classification of roads on Earth satellite images. In one case, we made use of dilated convolutional layers to increase the receptive field of our model without giving up on high resolution which could reduce the accuracy. Using a random data augmentation, we were able to obtain results significantly higher by increasing the dataset size. For purpose of comparison, we constructed two different models based on different architectures.

### Data:
We are provided with two datasets, one for training and one for evaluation of the model.

The training dataset consists of 400x400 pixels satellite images extracted from Google Maps along with ground-truths images for training where each pixel is labelled as either a road or background. The number of training examples is a 100. 

The evaluation set contains 50 images of 608x608 pixels, on which predictions are made on 16x16 patches all over the image. We then use a threshold to determine whether the patch should be classified as a road or a background.

### Methods and Models:

### Data Augmentation:
Data augmentation is a recurring trick used in image classification and segmentation tasks as it allows the model to be trained on datasets multiple times their original size while starting with the same data. Data augmentations uses transformations such as rotations or rolling in order to present to the model images that, despite coming from the same original, are considered as two separate data points and offer two distinct learning outcomes. Here is a list of the every transformation available in our project:

<ol>
  <li>Identity,</li>
  <li>Flip,</li>
  <li>Rotate,</li>
  <li>Contrast,</li>
  <li>AutoContrast,</li>
  <li>Roll,</li>
  <li>Add Noise</li>
  <li>Brightness,</li>
  <li>Sharpness.</li>
</ol>


### U-net:
We implemented state of the art neural network architecture (proposed by Ronneberger et al.) used for semantic image segmentations, in order to segment roads and perform classification on each pixel as background or road.


### Fractal net:
An other model available is the fractal model (proposed by Gustav Larsson et al.). It is originally designed for image classification, but with some modifications, we managed to use it in the context of image segmentation. It is arranged in a sequence of blocks, containing each convolutional and joining layers. A pooling operation is performed between each.


### How to run:
Using the ML_project2.ipynb notebook file run the cell containing run() without arguments to launch training for our best model: U-net with dilated bottleneck architecture.
The default parameters for run are (train = True, use_fractal = False, augment = False, augment_random = False, augment_factor = 7, dilation = True, display_preds = True)

* Parameters relative to the choice of the model and training:
> Train:Wheter to train the model or not, if false the model launches prediction directly from our best model file bestmodel.h5 if unet is used or Fractal_model.h5 if fractal is used

> use_fractal: if true uses fractal network, if false uses Unet

Parameters relative to the data augmentation :
> augment: if true augments the training data

> augment_random: if true uses random data augmentation, if false uses the non random data augmentation

> augment_factor: relative to random data augmentation, which is the number of folds training set is increased by random augmentation

* Parameter for visualization of the results:
> display_preds: whether to display predictions on some images after predictions or not

### Files:

* **run.py:** allows to initialize different training parameters train and evaluate the model and make predictions.
* **ML_project2.ipynb:** Notebook to run the code and get image classifications.
* **preprocessing.py:** Functions to load training images, convert them in array form, prepare for training by augmenting and building train/test/val sets
* **UNet_model.py:** Functions to build U-net architecture
* **fractal_net.py:** Functions to build the fractal architecture 
* **proj2_helpers.py:** Various helper functions allowing  mainly to load test images, make predictions and make a csv submission.


### Libraries and Progamming language:
- [Python](https://www.python.org)
- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)

For neural networks:
- [Keras](https://keras.io)
- [Tensorflow](https://www.tensorflow.org/?hl=fr)

For image processsing:
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [Scipy](https://www.scipy.org)
