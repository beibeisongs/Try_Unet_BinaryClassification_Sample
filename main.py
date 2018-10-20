# encoding=utf-8
# Date: 2018-10-18
# Reference from: https://github.com/zhixuhao/unet
# Theory Reference: https://blog.csdn.net/u012931582/article/details/70215756
# Error Correction Reference: https://github.com/zhixuhao/unet/issues/45


from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

from model import unet
from model import ModelCheckpoint


""" Description: 

    The combination of the values of the array represents the RGB in labeled datas
"""
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]
# Note: [[128 128 128] [128 0 0] [192 192 128] [128  64 128] [60 40 222] [128 128 0] [192 128 128] [ 64 64 128] [64 0 128] [64 64 0] [0 128 192] [0 0 0]]
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):
        img = img / 255
        mask = mask[:, :, :, 0] if(len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0], index[1], index[2], np.zeros(len(index[0]), dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0], index[1], np.zeros(len(index[0]), dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict,
                   image_color_mode="grayscale", mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    ''' Description:

        Generate image and mask at the same time
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"

        Notes:
            batch_size = 2; train_path = "data/membrane/train"; image_folder = "image"; mask_folder = "label";
            aug_dict = {'shear_range': 0.05, 'height_shift_range': 0.05, 'horizontal_flip': True, 'fill_mode': 'nearest', 'rotation_range': 0.2, 'width_shift_range': 0.05, 'zoom_range': 0.05}
    '''

    image_datagen = ImageDataGenerator(**aug_dict)  # Note: <keras.preprocessing.image.ImageDataGenerator object at 0x000001B604B0E588>
    mask_datagen = ImageDataGenerator(**aug_dict)   # Note: <keras.preprocessing.image.ImageDataGenerator object at 0x000001B604B0E588>
    image_generator = image_datagen.flow_from_directory(
        train_path,                                     # Note: 'data/membrane/train'
        classes=[image_folder],                         # Note: 'image'
        class_mode=None,                                # Note: None
        color_mode=image_color_mode,                    # Note: 'grayscale'
        target_size=target_size,                        # Note: <class 'tuple'>: (256, 256)
        batch_size=batch_size,                          # Note: 2
        save_to_dir=save_to_dir,                        # Note: None
        save_prefix=image_save_prefix,                  # Note: 'image'
        seed=seed                                       # Note: 1
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,                                     # Note: 'data/membrane/train'
        classes=[mask_folder],                          # Note: 'label'
        class_mode=None,                                # Note: None
        color_mode=mask_color_mode,                     # Note: 'grayscale'
        target_size=target_size,                        # Note: <class 'tuple'>: (256, 256)
        batch_size=batch_size,                          # Note: 2
        save_to_dir=save_to_dir,                        # Note: D:/UNET：Realising Sample/unet-master
        save_prefix=mask_save_prefix,                   # Note: 'mask'
        seed=seed                                       # Note: 1
    )

    train_generator = zip(image_generator, mask_generator)  # Note: <zip object at 0x000001B604B10288>
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)

        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask", image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


# Note: {'shear_range': 0.05, 'height_shift_range': 0.05, 'horizontal_flip': True, 'fill_mode': 'nearest', 'rotation_range': 0.2, 'width_shift_range': 0.05, 'zoom_range': 0.05}
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
""" Description: 

    Important Operation Logic Note: 
        You can see the last line of the function "trainGenerator", it says: yield (img, mask) (in a 'for' Loop)
        That means this code structure is used to produce batches of training and labeled datas
        
    Besides, notice that the variable 'myGene' is utilized by the script below: model.fit_generator(..)
"""
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

""" Attention: the following codes should be executed while hdf5 not existing
    
    >>>model = unet()
    while it is existing
    >>>model = unet(pretrained_weights="unet_membrane.hdf5")
"""
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)    # Note: the first parameter of the function: filepath: string, path to save the model file.
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])    # Note: myGnene: <generator object trainGenerator at 0x000001B67F9FD830>

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/membrane/test", results)