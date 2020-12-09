from data_augmentation import *
from sklearn.model_selection import train_test_split


def load_image(directory):
    return mpimg.imread(directory)


def preprocess(root_dir='./training/'):
    images_dir = root_dir + 'images/'
    gt_dir = root_dir + 'groundtruth/'
    files = os.listdir(images_dir)

    images = [PIL.Image.open(images_dir + img) for img in files]
    gt_images = [PIL.Image.open(gt_dir + img) for img in files]
    rdaug = RandAugment(12, 12)

    imgs_aug = []
    gt_imgs_aug = []

    for (img, gt_img, file_, i) in zip(images, gt_images, files, range(len(images))):
        print("processing image ", i, "/", len(images))
        #for j in range(3):
        #    print(j)
        #    img, gt_img = rdaug.augment((img, gt_img))
        imgs_aug.append(img_to_array(img))
        gt_imgs_aug.append(img_to_array(gt_img))
        #    img.save(images_dir + str(i) + file_)
        #    gt_img.save(gt_dir+ str(i) + file_)

    imgs_aug = np.asarray([img_to_array(img_) for img_ in imgs_aug])
    gt_imgs_aug = np.asarray([img_to_array(img_) for img_ in gt_imgs_aug])
    #if(divide_set == False):
    #    return imgs_aug, gt_imgs_aug
    #else:
    #    X_train, Y_train = imgs_aug, gt_imgs_aug
    #                                                        test_size=ratio, random_state=0)
    X_train,X_test,Y_train,Y_test=train_test_split(imgs_aug,gt_imgs_aug, test_size=0.2, random_state=0)
    #X_train, Y_train = imgs_aug, gt_imgs_aug
    X_train = np.stack(X_train, axis = 0)
    Y_train = np.stack(Y_train, axis = 0)/255.0
    X_test=np.stack(X_test, axis = 0)
    Y_test=np.stack(Y_test, axis = 0)/255.0
    print("size ytest",Y_test.shape)
    print("size xtest",X_test.shape)
    Y_train[Y_train>0.5]=1
    Y_train[Y_train<0.5]=0
    Y_test[Y_test>0.5]=1
    Y_test[Y_test<0.5]=0
    return X_train,X_test,Y_train,Y_test
