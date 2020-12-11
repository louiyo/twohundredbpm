from data_augmentation import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from proj2_helpers import img_float_to_uint8

TEST_IMG_SIZE = 608
TRAIN_IMG_SIZE = 400

def pad_single(img):
    size_diff = int((TEST_IMG_SIZE - img.shape[0]) / 2)
    return np.pad(img, pad_width = ((size_diff, size_diff), 
                (size_diff, size_diff), (0,0)), mode='symmetric')
   

# A voir comment utiliser ce machin

def preprocess(root_dir='./training/',
               divide_set=True,
               ratio=0.1,
               upscale_to_test_size=False,
               save_imgs = False):
    images_dir = root_dir + 'images/'
    gt_dir = root_dir + 'groundtruth/'
    files = os.listdir(images_dir)

    images = [PIL.Image.open(images_dir + img) for img in files]
    gt_images = [PIL.Image.open(gt_dir + img) for img in files]
    rdaug = RandAugment(5, 12, upscale_to_test_size)
    
    imgs_aug = []
    gt_imgs_aug = []

    for (img_, gt_img_, file_, i) in zip(images, gt_images, files, range(len(images))):
        print("processing image ", i+1, "/", len(images))
            
        if(upscale_to_test_size):
            imgs_aug.append(pad_single(img_to_array(img_)))
            gt_imgs_aug.append((pad_single(img_to_array(gt_img_))))
        else: 
            imgs_aug.append(img_to_array(img_))
            gt_imgs_aug.append(img_to_array(gt_img_))
            
        for j in range(1):
            img, gt_img = img_.copy(), gt_img_.copy()
            img, gt_img = rdaug.augment((img, gt_img))
            img, gt_img = img_to_array(img), img_to_array(gt_img)
            
            if(upscale_to_test_size):
                img = pad_single(pad_single(img)) 
                gt_img = pad_single(pad_single(gt_img)) 
            if(save_imgs):
                array_to_img(img).save(images_dir + 'augmented' + str(j) + file_)
                array_to_img(gt_img).save(gt_dir + 'augmented' + str(j) + file_)

    if(divide_set == False):
        imgs_aug = np.stack(imgs_aug, axis = 0)
        gt_imgs_aug = np.stack(gt_imgs_aug, axis = 0) / 255.0
        gt_imgs_aug[Y_train>0.5]=1
        gt_imgs_aug[Y_train<0.5]=0
        return (imgs_aug, gt_imgs_aug)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(imgs_aug, gt_imgs_aug,
                                                            test_size=ratio, random_state=2020)
        X_train = np.stack(X_train, axis = 0)
        X_test = np.stack(X_test, axis = 0)
        Y_train = np.stack(Y_train, axis = 0) / 255.0
        Y_test = np.stack(Y_test, axis = 0) / 255.0
        return X_train, X_test, Y_train, Y_test
        print("size ytest",Y_test.shape)
        print("size xtest",X_test.shape)
        Y_train[Y_train>0.5]=1
        Y_train[Y_train<0.5]=0
        Y_test[Y_test>0.5]=1
        Y_test[Y_test<0.5]=0
        return X_train,X_test,Y_train,Y_test
