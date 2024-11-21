import argparse
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from matplotlib.colors import LogNorm
from numpy._typing import NDArray


def dft(x: NDArray) -> NDArray:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(-1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


def idft(x: NDArray) -> NDArray:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(1j * 2 * math.pi * kxn * (1 / N))
    return np.dot((1 / N) * e, x)


def _idft(x: NDArray) -> NDArray:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


def fft(x: NDArray) -> NDArray:
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
    return np.transpose([dft(col) for col in np.transpose([dft(row) for row in x])])


def fft2(x: NDArray):
    return np.transpose([fft(col) for col in np.transpose([fft(row) for row in x])])


def ifft2(x: NDArray):
    return np.transpose([ifft(col) for col in np.transpose([ifft(row) for row in x])])


def load_gray_scale(path: str) -> MatLike:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: MatLike, cutoff=0.99) -> tuple[MatLike, int]:
    # to fft
    freqs = fft2(img)

    # zeroing
    index_cutoff = int(cutoff * freqs.shape[0] * freqs.shape[1])
    thresh = np.sort(np.abs(freqs.flatten()))[index_cutoff]
    freqs[freqs >= thresh] = 0

    nonzero = np.count_nonzero(freqs)

    # back to image
    return np.real(ifft2(freqs)), nonzero


def compress(img: MatLike):
    freqs = fft2(img)
    index_cutoff = int(0.95 * freqs.shape[0] * freqs.shape[1])
    thresh = np.sort(np.abs(freqs.flatten()))[index_cutoff]
    freqs[freqs >= thresh] = 0
    return np.real(ifft2(freqs))


def next_power_of_2(x):
    if x == 0:
        return 1
    else:
        return 2 ** (x - 1).bit_length()


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


def denoised_plot(image: MatLike, denoised_image: MatLike, count: int):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(denoised_image, cmap="gray")
    ax[1].set_title("Denoised Image")

    print("number of nonzero values: ", count)
    total = image.shape[0] * image.shape[1]
    print("nonzero / total: ", count, "/", total, "=", count / total)
    plt.show()


def compress_plot(image: MatLike, compressed: MatLike):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(compressed, cmap="gray")
    ax[1].set_title("Compressed Image")
    plt.show()


def runtime_plot(sizes: NDArray):
    powers_2 = 2**sizes
    futs = [dft2, fft2]

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

        stds = 2 * np.array(stds)
        ax[fut_idx].errorbar(powers_2, average_times, yerr=stds)

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
        help="Image file path",
        required=True,
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
        compressed = compress(image)
        pass

    elif args.mode == 4:
        # TODO: fixe sizes according to what will take a a reasonable amount of time
        runtime_plot(np.arange(11)[5:])


if __name__ == "__main__":
    main()
