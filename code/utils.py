import os
from PIL import Image
import numpy as np
import cv2
from glob import glob


RESHAPE = (512, 512)


# 先提取出一些细节来
def guided_filter(data, num_patches, image_size, num_channels):
    r = 10 
    eps = 10
    batch_q = np.zeros((num_patches, image_size, image_size, 3))
    for i in range(num_patches):
        for j in range(num_channels):
            I = data[i, :, :, j]
            p = data[i, :, :, j]
            ones_array = np.ones([image_size, image_size])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) 
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            q = mean_a * I + mean_b 
            batch_q[i, :, :, j] = q 
    return batch_q


def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def load_data(path, batch_size=1):
    
    batch_images = np.random.choice(path, size=batch_size)

    imgs_hr = []
    imgs_lr = []
    for img_path in batch_images:
        img = cv2.imread(img_path)
        w = 512

        img_hr = img[:, w:]
        img_lr = img[:, 0:w] 

        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1
    imgs_lr = np.array(imgs_lr) / 127.5 - 1

    return imgs_hr, imgs_lr
