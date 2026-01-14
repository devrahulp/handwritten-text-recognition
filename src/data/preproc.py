import cv2
import numpy as np

def preprocess(img, input_size):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    new_w = int(w * input_size[1] / h)
    img = cv2.resize(img, (new_w, input_size[1]))

    if new_w > input_size[0]:
        img = img[:, :input_size[0]]
    else:
        pad = input_size[0] - new_w
        img = np.pad(img, ((0,0),(0,pad)), constant_values=255)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.transpose(img, (1, 0, 2))

    return img

def normalization(images):
    return np.array(images)

import numpy as np
import cv2

def adjust_to_see(img):
    img = img.copy()
    img = img.astype(np.uint8)
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    return img
