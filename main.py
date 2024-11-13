from typing import Any
import numpy as np
from numpy._typing import NDArray
import math

np.random.seed(10)
SAMPLE = np.random.rand(8)


def dft1(x: NDArray[np.float64]) -> list[float]:
    N = len(x)
    out = []
    for k in range(N):
        xk = 0
        for n in range(N):
            e = np.exp(-1j * 2 * math.pi * k * n * (1 / N))
            xk += e * x[n]
        out.append(xk)
    return out


def dft(x: NDArray[np.float64]) -> NDArray[Any]:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(-1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


def idft1(x: NDArray[np.float64]):
    N = len(x)
    out = []
    for k in range(N):
        xk = 0
        for n in range(N):
            e = np.exp(1j * 2 * math.pi * k * n * (1 / N))
            xk += e * x[n]
        out.append((1/N)*xk)
    return out

def idft(x: NDArray[np.float64]) -> NDArray[Any]:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(1j * 2 * math.pi * kxn * (1 / N))
    return np.dot((1/N)*e, x)

def ifft_helper(x: NDArray[np.float64]) -> NDArray[Any]:
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


def ifft(x: NDArray[np.float64]):
    N = len(x)

    if N <= 4:
        return ifft_helper(x)

    odd = ifft(x[1::2])
    even = ifft(x[0::2])

    out = []
    for k in range(N // 2):
        e = np.exp(1j * 2 * math.pi * k * (1 / N))
        out.append((even[k] + e * odd[k]))
    for k in range(N // 2):
        e = np.exp(1j * 2 * math.pi * k * (1 / N))
        out.append((even[k] - e * odd[k]))

    out = (1/N)*np.array(out)
    return out


def fft2():
    pass


def ifft2():
    pass


if __name__ == "__main__":
    r = np.fft.ifft(SAMPLE)
    r2 = ifft(SAMPLE)
    print(np.allclose(r, r2))
    pass
