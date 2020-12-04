from data_augmentation import *
from sklearn.model_selection import train_test_split

TEST_IMG_SIZE = 608
TRAIN_IMG_SIZE = 400

def preprocess(root_dir='./training/',
               patches=False,
               divide_set=True,
               ratio=0.1,
               upscale_to_test_size=True,
               save_imgs = False):
    images_dir = root_dir + 'images/'
    gt_dir = root_dir + 'groundtruth/'
    files = os.listdir(images_dir)

    images = [PIL.Image.open(images_dir + img) for img in files]
    gt_images = [PIL.Image.open(gt_dir + img) for img in files]
    rdaug = RandAugment(7, 12, upscale_to_test_size)

    imgs_aug = []
    gt_imgs_aug = []

    for (img_, gt_img_, file_, i) in zip(images, gt_images, files, range(len(images))):
        print("processing image ", i+1, "/", len(images))
        for j in range(3):
            img, gt_img = img_.copy(), gt_img_.copy()
            img, gt_img = rdaug.augment((img, gt_img))
            img, gt_img = img_to_array(img), img_to_array(gt_img)
            
            
            if(upscale_to_test_size):
                size_diff = TEST_IMG_SIZE - img.shape[0]
                if(not (size_diff == TEST_IMG_SIZE - gt_img.shape[0])):
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                img = np.pad(img, pad_width = ((0, size_diff), (0, size_diff), (0,0)))
                gt_img = np.pad(gt_img, pad_width = ((0, size_diff), (0, size_diff), (0,0)))
            
            imgs_aug.append(img)
            gt_imgs_aug.append(gt_img)
            if(save_imgs):
                img.save(images_dir + 'augmented' + str(j) + file_)
                gt_img.save(gt_dir + 'augmented' + str(j) + file_)

    """imgs_aug = np.asarray([img_to_array(img_) for img_ in imgs_aug])
    gt_imgs_aug = np.asarray([img_to_array(img_) for img_ in gt_imgs_aug])"""
    # CLEAN CA SI CA MARCHE
    
    if(divide_set == False):
        imgs_aug = np.round(np.stack(imgs_aug, axis = 0) / 255)
        gt_imgs_aug = np.round(np.stack(gt_imgs_aug, axis = 0) / 255)
        return (imgs_aug, gt_imgs_aug)
    else:
        print("ok épic...")
        X_train, X_test, Y_train, Y_test = train_test_split(imgs_aug, gt_imgs_aug,
                                                            test_size=ratio, random_state=2020)
        X_train = np.round(np.stack(X_train, axis = 0) / 255)
        X_test = np.round(np.stack(X_test, axis = 0) / 255)
        Y_train = np.round(np.stack(Y_train, axis = 0) / 255)
        Y_test = np.round(np.stack(Y_test, axis = 0) / 255)
        return X_train, X_test, Y_train, Y_test