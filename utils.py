import cv2
import numpy as np
import os

def read_image(path):
    """ Reads an image from the given path. """
    if os.path.exists(path):
        return cv2.imread(path)
    else:
        raise FileNotFoundError(f"File not found: {path}")

def downsample_images(image, downsample_size=512):
    """ Downsamples an image until its dimensions are below the given size. """
    while image.shape[0] > downsample_size or image.shape[1] > downsample_size:
        image = cv2.pyrDown(image)
    return image

def sharpen(image):
    """ Applies a Laplacian high-pass filter to an image. """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened = cv2.addWeighted(image.astype(np.float64), 1.5, laplacian.astype(np.float64), -0.5, 0)
    return np.clip(laplacian, 0, 255).astype(np.uint8), np.clip(sharpened, 0, 255).astype(np.uint8)
