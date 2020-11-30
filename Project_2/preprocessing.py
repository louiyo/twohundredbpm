from data_augmentation import *


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

    for img, gt_img in images, gt_images:
        img, gt_img = rdaug.augment((img, gt_img))
        imgs_aug.append(img)
        gt_imgs_aug.append(gt_img)

    X = np.asarray(imgs_aug)
    Y = np.expand_dims(np.asarray(gt_imgs_aug), axis=3)
