"""Utility functions for performing data augmentations on images."""

import numpy as np
import scipy.ndimage
from skimage.transform import resize

def horizontal_flip(image):
    """
    Flips an image horizontally (reflect over the y-axis).

    Args:
      image: Numpy array of shape (w, h, c) representing the image.

    Returns:
      Flipped image of the same dimensions as the input image.
    """
    return image[:,::-1,:]

def vertical_flip(image):
    """
    Flips an image vertically (reflect over the y-axis).

    Args:
      image: Numpy array of shape (w, h, c) representing the image.

    Returns:
      Flipped image of the same dimensions as the input image.
    """
    return image[::-1,:,:]

def rotate(image, angle):
    """
    Rotates an image by the given angle.

    Args:
      image: Numpy array of shape (w, h, c) representing the image.
      angle: Float angle to rotate (in degrees).

    Returns:
      Rotated image of the same dimensions as the input image.
    """
    out = scipy.ndimage.rotate(image, angle, reshape=False)

    return out

# From https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions.
def scale(img, zoom_factor):
    """
    Scales an image by a factor (zoom in or out).

    Args:
      image: Numpy array of shape (w, h, c) representing the image.
      zoom_factor: Float of amount to scale (>1 means zoom in, <1 means zoom out).

    Returns:
      Scaled image of the same dimensions as the input image.
    """

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = scipy.ndimage.zoom(img, zoom_tuple)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = scipy.ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple)
        out = resize(out, img.shape)

    # If zoom_factor == 1, just return the input array
    else:
        out = img

    return out
