import argparse
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import os
from cv2.typing import MatLike
from matplotlib.colors import LogNorm
from numpy._typing import NDArray


def dft(x: NDArray) -> NDArray:
    """
    Naive implementation of dft.
    """
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(-1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


def idft(x: NDArray) -> NDArray:
    """
    Naive implementation of idft.
    """
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(1j * 2 * math.pi * kxn * (1 / N))
    return np.dot((1 / N) * e, x)


def _idft(x: NDArray) -> NDArray:
    """
    Modified version of idft that doesnt scale by 1/N.
    """
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


def fft(x: NDArray) -> NDArray:
    """
    Implementation of fft (Cooley-Tukey).
    """
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


def ifft(x: NDArray):
    """
    Implementation of ifft (Cooley-Tukey).
    """

    def rec(x: NDArray):
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


def dft2(x: NDArray):
    """
    Implementation of dft2 (2D) using naive dft.
    """
    return np.transpose([dft(col) for col in np.transpose([dft(row) for row in x])])


def idft2(x: NDArray):
    """
    Implementation of idft2 (2D) using naive dft.
    """
    return np.transpose([idft(col) for col in np.transpose([idft(row) for row in x])])


def fft2(x: NDArray):
    """
    Implementation of fft2 (2D) using fft.
    """
    return np.transpose([fft(col) for col in np.transpose([fft(row) for row in x])])


def ifft2(x: NDArray):
    """
    Implementation of ifft2 (2D) using ifft.
    """
    return np.transpose([ifft(col) for col in np.transpose([ifft(row) for row in x])])


def load_gray_scale(path: str) -> MatLike:
    """
    Loads an image with grayscale.
    """
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: MatLike, cutoff=0.985) -> tuple[MatLike, int]:
    """
    Denoises an image with sepecified cutoff
    """
    # to fft
    freqs = fft2(img)

    # zeroing
    index_cutoff = int(cutoff * freqs.shape[0] * freqs.shape[1])
    thresh = np.sort(np.abs(freqs.flatten()))[index_cutoff]
    freqs[freqs >= thresh] = 0

    nonzero = np.count_nonzero(freqs)

    # back to image
    return np.real(ifft2(freqs)), nonzero


def compression(img: MatLike, compression_level: float, filename: str) -> tuple[MatLike, int]:
    """
    Compresses an image with sepecified compression level.
    """
    # to fft
    freqs = fft2(img)

    # modify Fourier coefficients to compress
    index_cutoff = int(compression_level * freqs.shape[0] * freqs.shape[1])
    thresh = np.sort(np.abs(freqs.flatten()))[index_cutoff]
    freqs[np.abs(freqs) < thresh] = 0

    nonzero = np.count_nonzero(freqs)
    sparse_matrix = sp.csr_matrix(freqs)
    sp.save_npz(filename, sparse_matrix)

    # return the image to display
    return np.real(ifft2(freqs)), nonzero


def next_power_of_2(x):
    """
    Retuns the next power of 2
    """
    if x == 0:
        return 1
    else:
        return 2 ** (x - 1).bit_length()


def load_image(path: str):
    """
    Loads and image with gray scale and resizes to the next power of 2.
    """
    image = load_gray_scale(path)
    h, w = image.shape
    new_size = (next_power_of_2(w), next_power_of_2(h))
    image = cv2.resize(image, new_size)
    return np.asarray(image[:, :])


def log_scale_plot(image, image_fft_form):
    """
    Displays a log scale plot of the fft.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(np.abs(image_fft_form), norm=LogNorm(), cmap="gray")
    ax[1].set_title("Image FFT Form (log scaled)")
    plt.show()


def denoised_plot(image: MatLike, denoised_image: MatLike, count: int):
    """
    Displays the original image and a denoised version.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(denoised_image, cmap="gray")
    ax[1].set_title("Denoised Image")

    print("number of nonzero values: ", count)
    total = image.shape[0] * image.shape[1]
    print("nonzero / total: ", count, "/", total, "=", count / total)
    plt.show()


def compression_plot(image: MatLike):
    """
    Displays the original image and various compressions at different levels.
    """
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    
    compression_levels = [0, 0.2, 0.4, 0.6, 0.8, 0.999]
    file_names = [
        "./docs/sparse_matrix_0%",
        "./docs/sparse_matrix_20%",
        "./docs/sparse_matrix_40%",
        "./docs/sparse_matrix_60%",
        "./docs/sparse_matrix_80%",
        "./docs/sparse_matrix_99.9%"
    ]
    
    for idx, (level, file_name) in enumerate(zip(compression_levels, file_names)):
        compressed_image, nonzero_count = compression(image, level, file_name)
        row, col = divmod(idx, 3)  # Determine subplot position
        ax[row, col].imshow(compressed_image, cmap="gray")
        ax[row, col].set_title(f"Compression level {level*100:.1f}%")
        
        print(f"{level*100:.1f}% nonzeros count: ", nonzero_count)
        print(f"{level*100:.1f}% sparse matrix size: ", os.path.getsize(f"{file_name}.npz"))
    
    plt.show()


def runtime_plot(sizes: NDArray):
    """
    Displayes a runtime plot of dft2 vs fft2
    """
    powers_2 = 2**sizes
    futs = [dft2, fft2]
    titles = ["DFT2", "FFT2"]

    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    for fut_idx, fut in enumerate(futs):
        average_times = []
        stds = []
        for size in powers_2:
            times = []
            input = np.random.rand(size, size)
            for _ in range(10):
                start = time.time()
                fut(input)
                end = time.time()
                times.append(end - start)
            stds.append(np.std(times))
            average_times.append(np.average(times))

        stds = 3 * np.array(stds)
        ax[fut_idx].errorbar(powers_2, average_times, yerr=stds)

        ax[fut_idx].set_title(titles[fut_idx])
        ax[fut_idx].set_xlabel("Input size")
        ax[fut_idx].set_ylabel("Runtime")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="FFT Image Processing Application")
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        default=1,
        help="Mode: 1 for FFT display, 2 for Denoise, 3 for Compress, 4 for Runtime Plot",
    )

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="Image file path (Default is ./assets/moonlanding.png)",
        default="./assets/moonlanding.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image = load_image(args.image)

    if args.mode == 1:
        image_fft_form = fft2(image)
        log_scale_plot(image, image_fft_form)

    elif args.mode == 2:
        denoised_image, count = denoise(image)
        denoised_plot(image, denoised_image, count)

    elif args.mode == 3:
        compression_plot(image)

    elif args.mode == 4:
        runtime_plot(np.arange(9)[5:])


if __name__ == "__main__":
    main()
