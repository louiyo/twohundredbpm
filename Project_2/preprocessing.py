from data_augmentation import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TEST_IMG_SIZE = 608
TRAIN_IMG_SIZE = 400

def preprocess(root_dir='./training/',
               ratio=0.15,
               save_imgs = False,
               augment = True,
               augment_random = True,
               augment_factor = 6):

    images_dir = root_dir + 'images/'
    gt_dir = root_dir + 'groundtruth/'
    files = os.listdir(images_dir)

    images = [PIL.Image.open(images_dir + img) for img in files]
    gt_images = [PIL.Image.open(gt_dir + img) for img in files]

    if augment:
        if (augment_random):
            imgs_aug, gt_imgs_aug = randaug(images, gt_images,augment_factor)
        else:
            imgs_aug, gt_imgs_aug = non_randaug(images,gt_images)

        X_train, X_test, Y_train, Y_test = train_test_split(imgs_aug, gt_imgs_aug,
                                                            test_size=ratio, random_state=2020)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(images, gt_images,
                                                            test_size=ratio, random_state=2020)

    X_train = np.stack(X_train, axis = 0)
    X_test = np.stack(X_test, axis = 0)
    Y_train = np.stack(Y_train, axis = 0) / 255.0
    Y_test = np.stack(Y_test, axis = 0) / 255.0
    Y_train[Y_train>=0.55]=1
    Y_train[Y_train<0.55]=0
    Y_test[Y_test>=0.55]=1
    Y_test[Y_test<0.55]=0
    return X_train,X_test,Y_train,Y_test
