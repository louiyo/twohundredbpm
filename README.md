# CS-433 Course - 2020/2021
*Participants : Anas Machraoui, Camil Hamdane, Nathan Girard.*

## What's there ?
In this repository you can find the first project of the course of Machine learning (CS-433) of epfl. The second project of the semester is to be released soon.

## Project 1 :

### Description :
The goal of this project is to predict whether a particule is a boson or not depending on specific features. The dataset we analyize are the training and test set, of size (250000,30) and (568238,30) respectively. 

### Architecture :
The project is coded with Python3 using the numpy library only, and seaborn for visualization.

### Implementations :
For the computer to learn the prediction model, we give a variety of methods including **Least Squares** (with (stochastic) gradient descent), **ridge regression**, and **logistic regression**. These are the basic implementations :

- least squares GD(y, tx, initial w, max iters, gamma)
- least squares SGD(y, tx, initial w, max iters, gamma)
- least squares(y, tx)
- ridge regression(y, tx, lambda )
- logistic regression(y, tx, initial w, max iters, gamma)
- reg logistic regression(y, tx, lambda , initial w, max iters, gamma)

However these methods are not enough to achieve good performance. First we noticed that depending on the value of the *PRI_jet_num* feature of the data, some of the other features are irrelevant for predicting the label of this particle (we found that for the jet num zero, less than twenty out of 30 are defined, whereas 29 are in model 3). Therefore we *split* the dataset in *4*, already sorting the examples by their PRI_jet_num value. The algorithm then learns 4 different model.

Thus we added the **preprocessing** part, which *standardizes* the data, performs *polynomial expansion* on the features, removes *useless features* and replaces non defined data by the median of the corresponding feature when necessary. 

In addition, **cross-validation** is implemented in order to find the best hyper-parameter values depending on the Machine Learning method you want to use. 

### How to run the algorithm :
All the files of the project contains the different methods named above, such that running the run.py file allows your computer to learn the model with the training data. Then it predicts the labels of the examples in the test dataset. 

There is an alternative way to run the code : the Jupyter Notebook (Project.ipynb). With this tool, one can cross-validate the hyper-parameters values, compute the prediction model (ridge regression by default) and visualize training accuracy for example.

### Libraries and Programming language:
- [Numpy](https://numpy.org)
- [Python](https://www.python.org)
- [Seaborn](https://seaborn.pydata.org/)

## Project 2:

### Description:
We introduce our take on a **segmentation task**, using a **convolutional neural nets** for a classification of roads on Earth satellite images. In one case, we made use of dilated convolutional layers to increase the receptive field of our model without giving up on high resolution which could reduce the accuracy. Using a random data augmentation, we were able to obtain results significantly higher by increasing the dataset size. For purpose of comparison, we constructed two different models based on different architectures.

### Data:
We are provided with two datasets, one for training and one for evaluation of the model.

The training dataset consists of 400x400 pixels satellite images extracted from Google Maps along with ground-truths images for training where each pixel is labelled as either a road or background. The number of training examples is a 100. 

The evaluation set contains 50 images of 608x608 pixels, on which predictions are made on 16x16 patches all over the image. We then use a threshold to determine whether the patch should be classified as a road or a background.

### Methods and Models:

#### Data Augmentation:
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


#### U-net:
U-Net is an architecture for semantic segmentation. It consists of a contracting path and an expansive path. The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for down-sampling. At each down-sampling step we double the number of feature channels. Every step in the expansive path consists of an up-sampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers. 


#### Fractal net:
the fractal model is constructed by sequence of blocks, containing each a convolution and joining layers, between which a pooling operation is done. In our case, this model consists of 4 blocks with 3 convolutional layers each in sequence. It begins with 16 filters in the first layer, a number multiplied by 2 from blocks to blocks. This model return the probability of a patch to be either 1 or 0.

The interesting fact of fractal net over any other construction resides in its ability to transition from shallow to deep during training. Therefore, it allows for a rather quick answer when in "shallow mode" and a more precise answer when in "deep mode". Furthermore, it has proven to be really effective even without data augmentation, which is very costly computationally. All these facts lead us to think about fractal net as an appropriate model for our project. Although, it is originally designed for image classification tasks, we have made an adaptation using patched images.


### How to run:

There is two different way of running the code. The first method consists of doing it on your computer (in local), but you should be aware that neural networks are very **costly computationally**. Hence, if you don't have a powerful computer (with a decent GPU), you should use the second method. You can either choose to run everything, from loading of the data to the construction of the submission file, by using this command in your terminal (make sure you are in the right repository):

<pre><code> python3 run.py -args </code></pre>

The args correspond to the arguments of the method run, which will do everything for you. In this manner, you can either choose to train the model, use the fractal model (by default it will use the U-net), and upscale the training data to the test size. However, we don't recomment to upscale to test size for training of models, as it needs then more time to compute.

An other way of running the code can be the **Google Colab**, that we put at your disposition in the link below. To this extent, wou will be allowed to make use of an hosted server of Google, provided with a fine GPU:

**link :** To be Done

Also, to use the GPU, you have to enter the **runtime** menu, and then click on **modifying runtime type**. After that, you should be able to select **GPU** for running the code. 


### Libraries and Progamming language:
- [Python](https://www.python.org)
- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)

For neural networks:
- [Keras](https://keras.io)
- [Tensorflow](https://www.tensorflow.org/?hl=fr)

For image processsing:
- [Pillow](https://pillow.readthedocs.io/en/stable/)

