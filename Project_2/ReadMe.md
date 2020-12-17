## Project 2:

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
We implemented state of the art neural network architecture (proposed by Ronneberger et al.) used for semantic image segmentations, in order to predict segment roads and perform classificaiton on each pixel as background or road.


### Fractal net:
the fractal model is constructed by sequence of blocks, containing each a convolution and joining layers, between which a pooling operation is done. In our case, this model consists of 4 blocks with 3 convolutional layers each in sequence. It begins with 16 filters in the first layer, a number multiplied by 2 from blocks to blocks. This model return the probability of a patch to be either 1 or 0.

The interesting fact of fractal net over any other construction resides in its ability to transition from shallow to deep during training. Therefore, it allows for a rather quick answer when in "shallow mode" and a more precise answer when in "deep mode". Furthermore, it has proven to be really effective even without data augmentation, which is very costly computationally. All these facts lead us to think about fractal net as an appropriate model for our project. Although, it is originally designed for image classification tasks, we have made an adaptation using patched images.


### How to run:\
Using the ML_project2.ipynb notebook file run the cell containing run() without arguments to launch training for our best model: U-net with dilated bottleneck architecture.\
The default parameters for run are (train = True, use_fractal = False, augment = False, augment_random = False, augment_factor = 7, dilation = True, display_preds = True)\
Train:Wheter to train the model or not, if false the model launches prediction directly from our best model file bestmodel.h5 if unet is used or Fractal_model.h5 if fractal is used\
use_fractal:if true uses fractal network, if false uses Unet\
augment:if true augments the training data\
augment random:if true uses random data augmentation, if false uses the non random data augmentation\
augment factor: relative to random data augmentation, which is the number of folds training set is increased by random augmentation\
display_preds: whether to display predictions on some images after predictions or not\

###Files:\

## run.py\
allows to initialize different training parameters train and evaluate the model and make predictions.\
## ML_project2.ipynb\
Notebook to run the code and get image classifications.\
## preprocessing.py\
Functions to load training images, convert them in array form, prepare for training by augmenting and building train/test/val sets\
## UNet_model.py\
Functions to build U-net architecture\
## fractal_net.py\
Functions to build the fractal architecture\
## proj2_helpers.py\
Various helper functions allowing  mainly to load test images, make predictions and make a csv submission.\



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
