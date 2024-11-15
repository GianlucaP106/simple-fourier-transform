import math
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

import cv2

from cv2.typing import MatLike
import numpy as np
from numpy._typing import NDArray


def dft(x: NDArray[np.float64]) -> NDArray[Any]:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(-1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


def idft(x: NDArray[np.float64]) -> NDArray[Any]:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(1j * 2 * math.pi * kxn * (1 / N))
    return np.dot((1 / N) * e, x)


def _idft(x: NDArray[np.float64]) -> NDArray[Any]:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


def fft(x: NDArray[np.float64]) -> NDArray[Any]:
    N = len(x)

    if N <= 4:
        return dft(x)

    odd = fft(x[1::2])
    even = fft(x[0::2])

    out = []
    for k in range(N // 2):
        e = np.exp(-1j * 2 * math.pi * k * (1 / N))
        out.append(even[k] + e * odd[k])
    for k in range(N // 2):
        e = np.exp(-1j * 2 * math.pi * k * (1 / N))
        out.append(even[k] - e * odd[k])

    out = np.array(out)
    return out


def ifft(x: NDArray[Any]):
    def rec(x: NDArray[Any]):
        N = len(x)

        if N <= 4:
            return _idft(x)

        odd = rec(x[1::2])
        even = rec(x[0::2])

        out = []
        for k in range(N // 2):
            e = np.exp(1j * 2 * math.pi * k * (1 / N))
            out.append((even[k] + e * odd[k]))
        for k in range(N // 2):
            e = np.exp(1j * 2 * math.pi * k * (1 / N))
            out.append((even[k] - e * odd[k]))

        return np.array(out)

    N = len(x)
    return (1 / N) * rec(x)


def fft2(x: NDArray[Any]):
    return np.transpose([fft(col) for col in np.transpose([fft(row) for row in x])])


def ifft2(x: NDArray[Any]):
    return np.transpose([ifft(col) for col in np.transpose([ifft(row) for row in x])])


def load_gray_scale(path: str) -> MatLike:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: MatLike, cutoff=0.99):
    freqs = fft2(img)

    index_cutoff = int(cutoff * freqs.shape[0] * freqs.shape[1])
    thresh = np.sort(freqs.flatten())[index_cutoff]

    freqs[freqs >= thresh] = 0

    return np.real(ifft2(freqs))


def parse_args():
    parser = argparse.ArgumentParser(description="FFT Image Processing Application")
    parser.add_argument("-m", "--mode", type=int, default=1, help="Mode: 1 for FFT display, 2 for Denoise, 3 for Compress, 4 for Runtime Plot")
    parser.add_argument("-i", "--image", type=str, default="image.png", help="Image file path")
    return parser.parse_args()


def next_power_of_2(x):
    if x == 0:
        return 1
    else:
        return 2**(x - 1).bit_length()


def load_image(path: str):
    image = load_gray_scale(path)
    h, w = image.shape
    new_size = (next_power_of_2(w), next_power_of_2(h))
    image = cv2.resize(image, new_size)
    return np.asarray(image[:, :])


def log_scale_plot(image, image_fft_form):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(np.abs(image_fft_form), norm=LogNorm(), cmap="gray")
    ax[1].set_title("Image FFT Form (log scaled)")
    plt.show()


def denoised_plot(image, denoised_image):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(denoised_image, cmap="gray")
    ax[1].set_title("Denoised Image")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    #This needs to be defaulted to args.image
    image = load_image("C:/Users/Nico/Desktop/ECSE_316/Assignments/Assignment2/simple-fourier-transform/assets/moonlanding.png")
    
    if args.mode == 1:
        image_fft_form = fft2(image)
        log_scale_plot(image, image_fft_form)

    elif args.mode == 2:
        denoised_image = denoise(image)
        denoised_plot(image, denoised_image)

    elif args.mode == 3:
        pass

    elif args.mode == 4:
        pass

    # dsize = (512, 256)
    # img = cv2.resize(img, dsize)
    # fig, ax = plt.subplots(1, 4, figsize=(10, 5))

    # qs = [0.5, 0.75, 0.99]
    # for idx, q in enumerate(qs):
    #     a = denoise(img, cutoff=q)
    #     cv2.imshow("Image " + str(q), a)
    #     ax[idx].imshow(a, cmap="gray")
    #     ax[idx].set_title(str(q))

    # s = len(qs)
    # ax[s].imshow(img, cmap="gray")
    # ax[s].set_title("Original")

    # plt.show()
