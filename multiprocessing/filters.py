import numpy as np
from scipy.ndimage import convolve

def apply_grayscale(img_np):
    """1. Convert RGB to grayscale using luminance formula"""
    if len(img_np.shape) == 3:
        return np.dot(img_np[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return img_np

def apply_gaussian_blur(img_np):
    """2. Apply 3x3 Gaussian kernel for smoothing"""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    return _apply_kernel(img_np, kernel)

def apply_sobel(img_np):
    """3. Sobel filter to detect edges"""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gray = apply_grayscale(img_np)
    ix = convolve(gray.astype(float), Kx)
    iy = convolve(gray.astype(float), Ky)
    return np.clip(np.sqrt(ix**2 + iy**2), 0, 255).astype(np.uint8)

def apply_sharpen(img_np):
    """4. Image Sharpening to enhance details"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return _apply_kernel(img_np, kernel)

def apply_brightness(img_np, factor=1.2):
    """5. Increase or decrease image brightness"""
    return np.clip(img_np.astype(float) * factor, 0, 255).astype(np.uint8)

def _apply_kernel(img, kernel):
    if len(img.shape) == 3:
        return np.stack([convolve(img[:,:,i], kernel) for i in range(3)], axis=2)
    return convolve(img, kernel)
